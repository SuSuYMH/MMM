import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, QuantizeEMAResetL2
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask

class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,# codebook_size
                 code_dim=512,# codebook_dim
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,# encoder和decoder模型里每层feature的dim
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim # code的维度？
        self.num_code = nb_code # codebook的大小？
        self.quant = args.quantizer
        # 这个是数据集的pose维度，起名字很奇怪，为什么叫output_dim？
        output_dim = 251 if args.dataname == 'kit' else 263

        '''定义编码器、解码器、量化器'''
        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        # Transformer Encoder
        # self.encoder = Encoder_Transformer(
        #     input_feats=output_dim,
        #     embed_dim=512, # 1024
        #     output_dim=512,
        #     block_size=4,
        #     num_layers=6,
        #     n_head=16
        # )

         # Transformer Encoder 4 frames
        # from exit.motiontransformer import MotionTransformerEncoder
        # in_feature = 251 if args.dataname == 'kit' else 263
        # self.encoder2 = MotionTransformerEncoder(in_feature, args.code_dim, num_frames=4, num_layers=2)

        self.decoder = Decoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        # self.decoder = Decoder_Transformer(
        #     code_dim=512,
        #     embed_dim=512, # 1024
        #     output_dim=output_dim,
        #     block_size=49,
        #     num_layers=6,
        #     n_head=8
        # )
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "ema_reset_l2":
            self.quantizer = QuantizeEMAResetL2(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        # 编码函数：对输入 x 进行编码，并返回其在量化字典中的索引
        # 虽然在当前代码中 encode 没有被调用，但它提供了一个方便的方法来单独获取输入的编码（即量化后的离散索引），可能在特定场景或推理过程中使用。
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        # 完整的传播过程
        x_in = self.preprocess(x) #x_in.shape: torch.Size([256, 263, 64]) motion的长度为64
        print('x_in.shape:', x_in.shape)


        # Encode
        # _x_in = x_in.reshape( int(x_in.shape[0]*4), x_in.shape[1], 16)
        # x_encoder = self.encoder(_x_in)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        # [Transformer Encoder]
        # _x_in = x_in.reshape( int(x_in.shape[0]*x_in.shape[2]/4), x_in.shape[1], 4)
        # _x_in = _x_in.permute(0,2,1)
        # x_encoder = self.encoder2(_x_in)
        # x_encoder = x_encoder.permute(0,2,1)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))
        
        x_encoder = self.encoder(x_in)# x_encoder.shape: torch.Size([256, 32, 16]) 把64降采样为16个token，每个token的dim是32
        print('x_encoder.shape:', x_encoder.shape)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)# x_quantized.shape: torch.Size([256, 32, 16])
        print('x_quantized.shape:', x_quantized.shape)

        ## decoder
        x_decoder = self.decoder(x_quantized)# x_decoder.shape: torch.Size([256, 263, 64])
        print('x_decoder.shape:', x_decoder.shape)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        # 没有encoder的前向过程，应该是直接用于AR后结果恢复成pose的作用
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 # cnmd终于找到错在哪了，原来你这个参数没用上啊？？？我*你妈了我
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        # 所以最后确实是code_dim的32维度
        self.vqvae = VQVAE_251(args, nb_code, code_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x, type='full'):
        '''type=[full, encode, decode] 表示前向传播的模式'''

        if type=='full':
            # 完整模式：执行编码、量化和解码的完整流程
            x_out, loss, perplexity = self.vqvae(x)
            return x_out, loss, perplexity
        elif type=='encode':
            # 仅编码模式：只对输入 x 编码，返回量化后的索引
            b, t, c = x.size()
            quants = self.vqvae.encode(x) # (N, T)
            return quants
        elif type=='decode':
            # 仅解码模式：输入量化索引 x，通过解码器解码为重构输出
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            # 如果 `type` 参数无效，抛出异常
            raise ValueError(f'Unknown "{type}" type')
        