"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.

Refer to 
https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py
https://github.com/theAdamColton/ijepa-enhanced/blob/7edef5f7288ae8f537f0db8a10044a2a487f70c9/ijepa_enhanced/lfq.py
"""

from math import log2, ceil
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce

def exists(v):# 看看v是不是None，是none就返回false，不是none就返回true
    return v is not None

def default(*args):# 就是默认前面的有值，就用前面的（如果是对象九创建对象，只是个变量就直接返回变量）
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

# 计算概率分布的熵
# prob 是一个张量，通常表示概率分布，且每一项的值介于 0 和 1 之间。比如，如果 prob 是一个二维张量（如 batch_size x num_classes），其中每行表示一个类别的概率分布
def entropy(prob):
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)

# class

def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y

def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))

def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """

    '''
    avg_probs是每个codebook token对应的样本概率的平均值，然后计算一些这个熵，得到整个批次的熵。要最大化这个熵，是为了让概率更平均，也就是每个token被更均匀的使用
    '''
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        # avg_probs = probs[mask].mean(tuple(range(probs.ndim - 1)))
        # avg_probs = einx.mean("... D -> D", probs[mask])

        avg_probs = masked_mean(probs, mask)
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # sample_entropy = sample_entropy[mask].mean()
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss

class LFQ(Module):
    '''
    注意这里面的bits是{0,1}的，而token的每一维是{-1,1}的，注意转换
    '''
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        num_codebooks = 1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        token_factorization = False,
        factorized_bits = [9, 9]
    ):
        super().__init__()

        # some assert validations
        # 'LFQ 需要指定 dim 或 codebook_size 之一'
        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        # 如果提供了 codebook_size，确保它是 2 的幂，因为后续的二进制操作需要这个条件
        # 如果 codebook_size 已提供（即不是 None），则使用提供的 codebook_size。
		# 如果 codebook_size 未提供（即为 None），则计算 2 ** dim 作为默认值。
        self.codebook_size = default(codebook_size, lambda: 2 ** dim)
        
        self.codebook_dim = int(log2(codebook_size))

        # 总的码本维度是单个码本维度乘以码本数量
        codebook_dims = self.codebook_dim * num_codebooks
        # 如果未提供 dim，则设置为总的码本维度
        dim = default(dim, codebook_dims)

        # 检查是否需要投影层，如果输入维度与码本维度不同，则需要
        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        # 保存 dim、codebook_dim 和 num_codebooks 到实例变量
        self.dim = dim
        self.codebook_dim = self.codebook_dim
        self.num_codebooks = num_codebooks

        '''
        解释：
	    1.	参数验证：
	    •	确保至少提供了 dim 或 codebook_size 之一。
	    •	如果提供了 codebook_size，确保它是 2 的幂，因为后续的二进制操作需要这个条件。
	    2.	设置 codebook_size：
	    •	使用 default 函数，如果 codebook_size 已提供，则使用其值。
	    •	如果未提供 codebook_size，则计算 2 ** dim 作为默认值。
	    •	这样，codebook_size 与 dim 保持一致性，确保可以进行正确的量化操作。
	    3.	计算 codebook_dim：
	    •	通过 codebook_dim = int(log2(self.codebook_size)) 计算码本维度。
	    •	这表示需要多少位来表示所有的码本索引。
	    4.	调整 dim：
	    •	计算总的码本维度：codebook_dims = self.codebook_dim * num_codebooks。
	    •	如果未提供 dim，则默认使用 codebook_dims。
	    •	这样可以确保输入维度与码本维度匹配，或者在需要时添加投影层。
	    5.	确定是否需要投影层：
	    •	has_projections = dim != codebook_dims。
	    •	如果输入维度 dim 与总的码本维度 codebook_dims 不同，则需要一个投影层来调整维度。
	    6.	保存参数：
	    •	将 dim、codebook_dim 和 num_codebooks 保存为实例变量，供后续使用。

        举例说明
        假设以下情况：
        •	情况 1：提供了 dim，未提供 codebook_size
        •	例如，dim = 16，codebook_size = None。
        •	由于未提供 codebook_size，则计算 self.codebook_size = 2 ** dim = 2 ** 16 = 65536。
        •	然后，self.codebook_dim = int(log2(65536)) = 16。
        •	如果 num_codebooks = 1，则 codebook_dims = 16 * 1 = 16。
        •	因为 dim 已经是 16，与 codebook_dims 相等，所以不需要投影层。

        •	情况 2：提供了 codebook_size，未提供 dim
        •	例如，codebook_size = 512，dim = None。
        •	确认 codebook_size 是 2 的幂，log2(512) = 9，是整数。
        •	计算 self.codebook_dim = int(log2(512)) = 9。
        •	如果 num_codebooks = 2，则 codebook_dims = 9 * 2 = 18。
        •	由于未提供 dim，则 dim = codebook_dims = 18。
        •	因为 dim 与 codebook_dims 相等，不需要投影层。

        •	情况 3：同时提供 dim 和 codebook_size，且不匹配
        •	例如，dim = 32，codebook_size = 256。
        •	计算 self.codebook_dim = int(log2(256)) = 8。
        •	如果 num_codebooks = 2，则 codebook_dims = 8 * 2 = 16。
        •	由于 dim = 32，与 codebook_dims = 16 不同，需要投影层将输入维度从 32 投影到 16。
        '''
        
        # for entropy loss
        # 保存熵损失的权重参数
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # for no auxiliary loss, during inference
        # 这个mask要在计算indices的时候使用
        self.token_factorization = token_factorization
        if not self.token_factorization: #for first stage model
            '''
            解释：
		    torch.arange(self.codebook_dim)：生成一个从 0 到 self.codebook_dim - 1 的整数序列。例如，如果 self.codebook_dim 为 4，则生成 tensor([0, 1, 2, 3])。
		    2 ** torch.arange(self.codebook_dim)：对序列中的每个元素计算 2 的相应幂次方，结果为 tensor([1, 2, 4, 8])。
		    self.register_buffer('mask', ...)：将计算得到的张量注册为模型的缓冲区 mask，用于后续的位操作。
            用途：
		    生成一个权重序列，用于将二进制位转换为整数索引。在位操作中，每个位代表一个权重，2 ** n 表示第 n 位的权重。
            '''
            self.register_buffer('mask', 2 ** torch.arange(self.codebook_dim), persistent=False)
            # mask：tensor([     1,      2,      4,      8,     16,     32,     64,    128,    256,
            #            512,   1024,   2048,   4096,   8192,  16384,  32768,  65536, 131072])
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer("pre_mask", 2** torch.arange(factorized_bits[0]), persistent=False)
            self.register_buffer("post_mask", 2**torch.arange(factorized_bits[1]), persistent=False)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes

        all_codes = torch.arange(codebook_size)# 【0,1,2,3,...,codebook_size-1】所有的十进制索引
        bits = self.indices_to_bits(all_codes)# 将所有索引都获得一个二进制的索引
        codebook = bits * 2.0 - 1.0 # 将二进制中的0变成-1，1还是1.这样就获得了codebooksize个token了

        self.register_buffer('codebook', codebook, persistent = False)# 将codebook存在模型里

    @property
    def dtype(self):
        return self.codebook.dtype
    
    def indices_to_bits(self, x):
        '''
        将整数索引转换为对应的二进制位表示，返回一个布尔张量
        Index  0: Bits [False, False, False, False] => Binary 0000
        Index  1: Bits [False, False, False, True] => Binary 0001
        Index  2: Bits [False, False, True, False] => Binary 0010
        Index  3: Bits [False, False, True, True] => Binary 0011
        Index  4: Bits [False, True, False, False] => Binary 0100
        Index  5: Bits [False, True, False, True] => Binary 0101
        Index  6: Bits [False, True, True, False] => Binary 0110
        Index  7: Bits [False, True, True, True] => Binary 0111
        Index  8: Bits [True, False, False, False] => Binary 1000
        Index  9: Bits [True, False, False, True] => Binary 1001
        Index 10: Bits [True, False, True, False] => Binary 1010
        Index 11: Bits [True, False, True, True] => Binary 1011
        Index 12: Bits [True, True, False, False] => Binary 1100
        Index 13: Bits [True, True, False, True] => Binary 1101
        Index 14: Bits [True, True, True, False] => Binary 1110
        Index 15: Bits [True, True, True, True] => Binary 1111
        '''
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def get_codebook_entry(self, x, bhwc, order): #0610
        if self.token_factorization:
            if order == "pre":
                mask = 2 ** torch.arange(self.factorized_bits[0], device=x.device, dtype=torch.long)
            else:
                mask = 2 ** torch.arange(self.factorized_bits[1], device=x.device, dtype=torch.long)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        
        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0 #back to the float
        ## scale back to the 
        b, h, w, c = bhwc
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w, c=c)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.codebook_dim
        indices = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)
    
    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # to some sort of float
        x = x.to(self.dtype)
        # -1 or 1
        x = x * 2 - 1
        x = rearrange(x, "... NC Z-> ... (NC Z)")
        return x

    def dequantize(self, indices):
        # 解码函数：输入嵌入索引，返回对应的嵌入向量
        bits = self.indices_to_bits(indices)
        token = bits * 2.0 - 1.0
        return token


    def quantize(self, x):
        # 量化函数：输入x，返回与嵌入最近的索引
        assert x.shape[-1] == self.dim

        x = rearrange(x, 'b l (c d) -> b l c d', c = self.num_codebooks)
        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value) 

        # 应该是这么写的，得在num_codebooks大于1的时候验证一下
        if self.token_factorization:
            indices_pre = reduce((quantized[..., :self.factorized_bits[0]] > 0).int() * self.pre_mask.int(), "b n c d -> b n c", "sum")
            indices_post = reduce((quantized[..., self.factorized_bits[0]:] > 0).int() * self.post_mask.int(), "b n c d -> b n c", "sum")
            return indices_pre, indices_post
        else:
            # 就是计算出十进制的indices吧，c要是等于1的话，就是要有bxn个token的indices（c大于1就表示有多个codebook）
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')
            return indices
    
    def forward(
        self,
        x,
        inv_temperature = 100.,
        return_loss_breakdown = False,
        mask = None,
        return_loss = True,
    ):
        # 输入的motion是【batch_size，dim，sequence_length】
        print(x.shape)
        x = rearrange(x, 'b d ... -> b ... d')
        print(x.shape)
        # 把dim移到最后，那其实不用这个pack_one了
        # x, ps = pack_one(x, 'b * d')
        # split out number of codebooks

        x = rearrange(x, 'b l (c d) -> b l c d', c = self.num_codebooks)
        # print(x.shape)# torch.Size([256, 18, 1, 32])

        '''
        量化：
        codebook_value就是1.0
        在这行代码中，条件 x > 0 用于检查 x 中每个元素是否大于 0。
		如果条件为 True（即元素大于 0），相应的位置将被填充为 codebook_value（即 1.0）。
		如果条件为 False（即元素小于或等于 0），相应的位置将被填充为 -codebook_value（即 -1.0）。
        '''
        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value) # higher than 0 filled 

        print((quantized > 0).int().shape)
        print(self.mask.int().shape)

        '''
        根据量化出的token，计算indices
        '''
        # calculate indices
        if self.token_factorization:
            indices_pre = reduce((quantized[..., :self.factorized_bits[0]] > 0).int() * self.pre_mask.int(), "b n c d -> b n c", "sum")
            indices_post = reduce((quantized[..., self.factorized_bits[0]:] > 0).int() * self.post_mask.int(), "b n c d -> b n c", "sum")
        else:
            # 就是计算出十进制的indices吧，c要是等于1的话，就是要有bxn个token的indices（c大于1就表示有多个codebook）
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')
        # print("indices:", indices.shape) (bs,length,1)

        '''
        计算熵相关的loss
        '''
        # entropy aux loss
        if self.training and return_loss:# 训练阶段
            logits = 2 * einsum('... i d, j d -> ... i j', x, self.codebook)
            # the same as euclidean distance up to a constant
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits = logits,
                sample_minimization_weight = self.sample_minimization_weight,
                batch_maximization_weight = self.batch_maximization_weight
            )

            avg_probs = self.zero
        else:
            # logits = 2 * einsum('... i d, j d -> ... i j', x, self.codebook)
            # probs = F.softmax(logits / 0.01, -1)
            # avg_probs = reduce(probs, "b n c d -> b d", "mean")
            # avg_probs = torch.sum(avg_probs, 0) #batch dimension
            # if not training, just return dummy 0
            per_sample_entropy = codebook_entropy = self.zero
            ## calculate the codebook_entropy needed for one batch evaluation
            entropy_aux_loss = self.zero
            avg_probs = self.zero

        '''
        commit loss
        '''
        # commit loss
        if self.training:
            commit_loss = F.mse_loss(x, quantized.detach(), reduction = 'none')

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero


        # use straight-through gradients (optionally with custom activation fn) if training

        quantized = x + (quantized - x).detach() #transfer to quantized

        # merge back codebook dim
        '''
        把量化结果和indices变换成初始的维度（为了计算方便在最一开始变换了维度，现在要把它们变回来）
        '''
        quantized = rearrange(quantized, 'b n c d -> b n (c d)')

        # reconstitute image or video dimensions

        # quantized = unpack_one(quantized, ps, 'b * d')这步应该不用了，因为没有压缩这一步
        quantized = rearrange(quantized, 'b ... d -> b d ...')

        
        if self.token_factorization:
            indices_pre = indices_pre.flatten()
            indices_post = indices_post.flatten()
            indices = (indices_pre, indices_post)
        else:
            indices = indices.flatten()

        '''
        按照他的做法，需要计算一个困惑度：
        理想情况下，每个嵌入向量被均匀使用时，困惑度最高；如果部分嵌入向量被过多使用，困惑度会降低
        '''
        # 创建 one-hot 编码表示的最小编码向量
        min_encodings = F.one_hot(indices, self.codebook_size).type(x.dtype)
        # 计算困惑度，衡量嵌入字典的利用率
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))

        return quantized, commit_loss, perplexity, indices
        # return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss, avg_probs)

if __name__ == "__main__":
    quantizer = LFQ(
    codebook_size = 2**18,      # codebook size, must be a power of 2
    dim = 18,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
    sample_minimization_weight = 1.0,        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
    batch_maximization_weight = 1.0
)

    motion_feats = torch.randn(256, 18, 4) #18 is dim, must be power of 2 of codebook_size【code的dim必须是codebook—size的2的指数的那个数】

    quantized, commit_loss, perplexity,indices = quantizer(motion_feats, inv_temperature=100.)  # you may want to experiment with temperature
    print("____________________")
    print(indices)
    print(quantized)# torch.Size([2, 18, 16]
    print(quantizer.indices_to_bits(indices))

    # assert motion_feats.shape == quantized.shape
    # assert (quantized == quantizer.indices_to_bits(indices)).all()