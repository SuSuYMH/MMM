'''
decoder和encoder中的基本块
'''
import torch.nn as nn
import torch

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        '''
        n_in: 输入的通道数。
	    n_state: 卷积核的输出通道数。
	    dilation: 膨胀率，控制卷积的膨胀。
	    activation: 激活函数类型，可以是 relu、silu 或 gelu。
	    norm: 选择的归一化层，可以是 LayerNorm (LN)、GroupNorm (GN)、BatchNorm (BN) 或不归一化（None）。
        '''
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            
        
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        # 首先保存输入 x 为 x_orig，用于残差连接
        x_orig = x
        if self.norm == "LN":
            # -1表示最后一个维度，-2表示倒数第二个维度
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

# Resnet1D类由多个ResConv1DBlock块组成，形成深层的残差网络
# dim都是一样的（也就是输入和输出channel都是一样的），就表示融合了一下特征呗
# 通过卷积操作提取局部和全局特征，使网络能够学习到输入数据的丰富表示；膨胀卷积允许在不增加参数的情况下扩大感受野，使模型适用于需要长时间依赖的任务
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        '''
        dilation_growth_rate: 控制膨胀率的增长，使每层的膨胀倍数为指数增长。
	    reverse_dilation: 如果为真，反转块的顺序。
	    self.model: 使用nn.Sequential将所有残差块串联成一个模型
        '''
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)