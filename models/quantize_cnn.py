'''
量化器quantizer的定义：
最基本的功能：forward、量化（用code找codeID）、解码（用codeID找code）
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizeEMAResetL2(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.reset_codebook()
        self.reset_count = 0
        self.usage = torch.zeros((self.nb_code, 1))

    def l2norm(self, t):
        return F.normalize(t, p = 2, dim = -1)
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = self.l2norm(out[:self.nb_code])       # 初始化的时候进行norm
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # code_rand：使用 self._tile 对 x 进行扩展和扰动后，从中随机选择 nb_code 个向量，形成一个新的随机向量集。
	    # 目的：在嵌入向量表需要重置时，未被频繁使用的嵌入向量会被随机向量替代。通过随机扰动的方式，使未被使用的向量重新分配，确保嵌入向量表的多样性。
        out = self._tile(x)
        code_rand = out[torch.randperm(out.shape[0])[:self.nb_code]]

        # Update centres
        '''
        EMA 更新 code_sum 和 code_count
        '''
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code 
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        '''
        Reset 机制 - 判断 usage 并应用重置逻辑
        重置机制：
	    每经过 20 次更新周期后，如果某个嵌入向量在所有周期中都未被频繁使用（即 self.usage 为 0），则允许将其随机替换。
	    在没有达到 20 次更新之前，所有嵌入向量保持更新，不会被替换。
        '''
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        self.usage = self.usage.to(usage.device)
        if self.reset_count >= 20:
            self.reset_count = 0
            usage = (usage + self.usage >= 1.0).float()
        else:
            self.reset_count += 1
            self.usage = (usage + self.usage >= 1.0).float()
            usage = torch.ones_like(self.usage, device=x.device)

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * code_rand
        self.codebook = self.l2norm(self.codebook)  # l2norm来保证每轮更新之后的codebook也能保证dim的归一化

        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)
        # 进行l2norm
        x = self.l2norm(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.reset_codebook()
        self.reset_count = 0
        self.usage = torch.zeros((self.nb_code, 1))
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[torch.randperm(out.shape[0])[:self.nb_code]]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        self.usage = self.usage.to(usage.device)
        if self.reset_count >= 20:
            self.reset_count = 0
            usage = (usage + self.usage >= 1.0).float()
        else:
            self.reset_count += 1
            self.usage = (usage + self.usage >= 1.0).float()
            usage = torch.ones_like(self.usage, device=x.device)
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim      # 每个嵌入向量的维度
        self.n_e = n_e          # 嵌入字典的大小
        self.beta = beta        # 控制嵌入损失的权重

        # 定义codebook，包含 `n_e` 个嵌入向量，每个向量维度为 `e_dim`
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 初始化codebook的权重
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        
        N, width, T = z.shape # 也就是(bs, Jx3, T)
        z = self.preprocess(z) # (bs, T, Jx3)
        assert z.shape[-1] == self.e_dim # 要确保codebook的code的维度和pose的维度是一致的
        # 将 `z` 扁平化为二维张量，形状为 (N * T, e_dim)
        # 将 z 变成 (N * T, e_dim) 的二维张量主要是为了简化距离计算并提升计算效率，使得可以一次性计算所有向量与嵌入向量之间的距离
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        # 计算 `z_flattened` 与嵌入向量之间的距离【（a-b)^2=a^2+b^2-2*a*b的向量形式】
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        # 获取距离最近的嵌入向量索引
        min_encoding_indices = torch.argmin(d, dim=1)
        # 使用最近的嵌入向量对 `z` 进行量化
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding 计算嵌入损失，包括对输入的重构误差和量化误差
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients 保持梯度传递，将 `z_q` 的梯度传递回 `z`
        z_q = z + (z_q - z).detach()
        # 恢复维度顺序为 (N, e_dim, T)
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)

        # 创建 one-hot 编码表示的最小编码向量
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        # 计算困惑度，衡量嵌入字典的利用率
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):
        # 量化函数：输入 z，返回与嵌入最近的索引
        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):
        # 解码函数：输入嵌入索引，返回对应的嵌入向量
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def preprocess(self, x):
        # 预处理函数：将输入张量从 (N, width, T) 转为 (N * T, width)
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x


class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))
        
    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = nn.Parameter(out[:self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        '''
        与EMA更新的主要不同就在这个地方！！！
        '''
        self.code_count = code_count  # nb_code
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape
        # Preprocess
        x = self.preprocess(x)
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)
        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity

    
class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99 # 用于EMA更新的衰减系数
        self.reset_codebook() # 初始化嵌入向量表和统计变量
        
    def reset_codebook(self):
        # 初始化嵌入向量表和统计变量
        self.init = False
        # 通过 code_sum / code_count 可以得到每个嵌入向量的 更新位置。这相当于对每个嵌入向量计算其分配到的输入向量的平均值，使得嵌入向量的位置逐渐靠近编码器输出的分布。
        # 使用指数移动平均进行更新可以使嵌入向量在训练过程中逐渐适应新的输入分布，但不会因为一次输入的变化而剧烈改变。
        self.code_sum = None # code_sum 是一个保存每个嵌入向量的加权和的张量。
        self.code_count = None # code_count 是一个计数器，记录每个嵌入向量被使用的频率。
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        # 使用输入向量来初始化嵌入向量（codebook），是为了让嵌入向量更好地适应数据分布
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:# 如果输入向量x的数量不够codebook的size那就用x多样化填充一下。如果输入向量 x 的数量不足以填充整个嵌入向量表，它会对 x 进行重复和随机扰动，生成足够多的向量。
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        # 初始化codebook以及相应辅助变量
        out = self._tile(x)
        self.codebook = out[:self.nb_code]# 从 out 中截取前 self.nb_code 行，将它们赋值给 self.codebook
        self.code_sum = self.codebook.clone()# self.code_sum 是一个辅助变量，用于记录嵌入向量的加权和。初始化为 self.codebook 的副本，意味着在初始化阶段，嵌入向量的加权和就是嵌入向量本身
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)# 初始化为1，表示每个嵌入向量初始时都被“使用”了一次，防止在更新时出现除以零的情况
        self.init = True # 将 self.init 标志设置为 True，表示嵌入向量表已经初始化。该标志可以在其他方法中使用，以检查嵌入向量表是否已经初始化，避免重复初始化。
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # 用于衡量离散化结果的有效性
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        '''
        更新嵌入向量表：使用EMA更新 code_sum 和 code_count，并计算新的嵌入中心 code_update。
	    EMA更新：使用 mu 衰减率，使嵌入向量缓慢地适应新的输入分布。
	    返回混杂度：衡量更新后的嵌入空间的均匀性。
        '''
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # 结果的形状: (nb_code, code_dim)，就是针对这一次输入x的每个codebook中的向量对应的所有输入的toekn的求和
        code_count = code_onehot.sum(dim=-1)  # 结果的形状: (nb_code） 每一维度就是输入x中有多少个token对应当前的这个codebook中的token

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # （code_dim, nb_code）
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # （nb_code）

        # 作用：通过 code_sum / code_count 计算新的嵌入向量。
		# self.code_sum / self.code_count 相当于每个嵌入向量的加权平均值，这个平均值可以更好地代表其被分配到的输入特征的中心位置。
		# view 用于调整 code_count 的形状，以便可以进行广播相除。
		# 更新后的 codebook 即为新的嵌入向量表，反映了当前的输入分布。
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = code_update

        # prob 是每个嵌入向量的使用频率
        prob = code_count / torch.sum(code_count)  
        # perplexity 表示嵌入空间中每个嵌入向量的使用分布情况，值越高表示嵌入向量分布越均匀
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity

    def preprocess(self, x):
        # 将输入张量从 (N, C, T) 转换为 (NT, C)，以便后续量化操作
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # 计算输入 x 到 codebook 中每个嵌入向量的距离，并找到最近的嵌入向量的索引 code_idx
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        # 通过 code_idx 索引到嵌入向量表 codebook 中的向量
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        '''
        完整前向过程：
	    1.	预处理：将输入调整为适合量化的形状。
	    2.	初始化嵌入向量表：在训练模式且未初始化时，初始化嵌入向量。
	    3.	量化和去量化：通过 quantize 和 dequantize 将输入映射到嵌入向量空间并恢复。
	    4.	更新嵌入向量表：在训练时，通过 update_codebook 更新嵌入表；在推理时，计算混杂度。
	    5.	计算损失：commit_loss 用于约束编码器输出向量接近其对应的嵌入向量。
	    6.	梯度分离：使用 x + (x_d - x).detach() 防止量化误差影响编码器的更新。
	    7.	后处理：调整输出形状为 (N, C, T)。
        '''
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity
