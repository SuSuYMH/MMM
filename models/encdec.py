'''
decoder和encoder
'''
import torch.nn as nn
from models.resnet import Resnet1D

class PrintModule(nn.Module):
    def __init__(self, me=''):
        super().__init__()
        self.me = me

    def forward(self, x):
        print(self.me, x.shape)
        return x
    
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,# 输入通道的数量,humanml3d则是263，kit则是251
                 output_emb_width = 512,# 编码后的输出通道数量
                 down_t = 3,# 下采样层的数量
                 stride_t = 2,# 时间维度下采样的步幅
                 width = 512,# 模型中的通道数量
                 depth = 3,# 每个 Resnet1D 块中的层数
                 dilation_growth_rate = 3,# 卷积中扩张率的增长率
                 activation='relu',# 模型中使用的激活函数
                 norm=None):# 归一化方法
        super().__init__()
        
        blocks = []
        # filter_t：卷积核大小，计算为步幅的两倍（stride_t * 2）
        # pad_t：填充大小，计算为步幅的一半（stride_t // 2）
        filter_t, pad_t = stride_t * 2, stride_t // 2
        # nn.Conv1d：一维卷积层，卷积核大小为3，步幅为1，填充为1
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),# Resnet1D：添加残差连接以改善梯度流和模型性能
            )
            blocks.append(block)
        # 调整输出通道：将通道数量从 width 改为 output_emb_width
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# Decoder 类从编码表示中重建输入数据，有效地逆转 Encoder 的操作
class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
