import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
init_save_folder(args)

# [TODO] make the 'output/' folder as arg
args.vq_dir = f'./output/vq/{args.vq_name}' #os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
codebook_dir = f'{args.vq_dir}/codebook/'
args.resume_pth = f'{args.vq_dir}/net_last.pth'
os.makedirs(args.vq_dir, exist_ok = True)
os.makedirs(codebook_dir, exist_ok = True)
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

# 上面import models.t2m_trans as trans
trans_encoder = trans.Text2Motion_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()
trans_encoder = torch.nn.DataParallel(trans_encoder)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

##### ---- get code ---- #####
##### ---- Dataloader ---- #####
# codebook_dir就是你离散编码后的pose的token的路径
# 要用你训练得到的motion VQVAE把humanml3d的motion序列编码成离散的token
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t) # down_t是downsampling rate
    # 这个unit_length的意思就是多少个pose编码成一个token，如果是4的话，motion一共200个pose，那最后经过VQVAE之后就会得到50个token在codebook中的id
    # 这个train_loader_token是一个dataloader
    for batch in train_loader_token:
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
        # 这个net是humanVQVAE，他的encode的forward是获得pose在codebook中的索引，这个target就是索引
        target = net(pose, type='encode')
        target = target.cpu().numpy()
        # 最后每一个存在output/vq/2024-10-27-19-36-20_300stepVQ/codebook下的npy文件，就是对应的motion经过VQVAE之后得到的motion的pose个数/降采样个token索引
        np.save(pjoin(codebook_dir, name[0] +'.npy'), target)

'''
dataname如果是t2m的话，就是去加载humanml3d这个数据集（里面有关于每个motion的描述）
if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
'''
train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, codebook_dir, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid=1000 
best_iter=0 
best_div=100 
best_top1=0 
best_top2=0 
best_top3=0 
best_matching=100 
# pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)

# get_acc 是一个计算准确率的辅助函数。它用于计算在指定的掩码（mask）条件下，模型预测与真实标签的匹配情况
def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()

# while nb_iter <= args.total_iter:
for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    '''
    clip_text是文本描述
    m_tokens是填充到50长度的离散motion token的id
    m_tokens_len是motion实际的长度

    m_tokens 会被填充到相同的长度，以确保所有样本的长度一致。这在深度学习模型的训练中是非常常见的做法，因为大多数深度学习框架要求一个批次（batch）中的数据具有相同的形状。
    代码中通过 self.max_motion_length 来控制所有样本的固定长度。无论原始的 m_tokens 长度如何，都会通过以下方式填充到相同的长度：
	1.	结束标记 (mot_end_idx)：在动作序列末尾添加一个结束标记，表示动作序列的结束。
	2.	填充标记 (mot_pad_idx)：如果 m_tokens 的长度小于 max_motion_length，则在结束标记之后继续用填充标记填充，直到达到 max_motion_length。
    假设 max_motion_length 是 50，而一个样本的 m_tokens 长度为 30。填充后的序列将会变成：
	•	[token_1, token_2, ..., token_30, mot_end_idx, mot_pad_idx, ..., mot_pad_idx]，总长度为 50。
    通过这种方式，可以确保每个 m_tokens 在经过填充后都有相同的长度 max_motion_length，从而便于批量处理。
    '''
    # print('***************')
    # print(clip_text)
    # print(m_tokens.shape)
    # print(m_tokens_len.shape)
    # print('***************')
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 填充长度)
    target = target.cuda()
    batch_size, max_len = target.shape[:2]

    # Random Drop Text
    # text_mask = np.random.random(len(clip_text)) > .05
    # clip_text = np.array(clip_text)
    # clip_text[~text_mask] = ''
    
    # 分词,获得文本中每个词元的词典编号
    # text.shape 应该是 (batch_size, max_seq_length)，其中：
	# batch_size 是文本的数量（clip_text 中的句子数量）。
	# max_seq_length 是 CLIP 模型的最大序列长度，通常为 77。这意味着每个文本被转化成了一个长度为 max_seq_length 的 token 序列，超出长度的部分被截断，不足的部分被填充。
    text = clip.tokenize(clip_text, truncate=True).cuda()

    # feat_clip_text：文本的全局特征向量，通常是一个单一的向量表示整个文本的语义信息，用于与图像特征进行相似度计算或其他任务。 shape为(batch_size, embedding_dim)，其中 embedding_dim 是文本特征向量的维度（通常为 512）
	# word_emb：每个词元的嵌入向量（embedding），表示输入文本的逐词表示。这可以用于分析或使用文本的每个词语的语义信息。shape为(batch_size, max_seq_length, embedding_dim)
    feat_clip_text, word_emb = clip_model(text)


    # [INFO] Swap input tokens
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(target.shape,
                                                device=target.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(target.shape,
                                                device=target.device))
    # random only motion token (not pad token). To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)
    mask = torch.logical_or(mask, ~seq_mask_no_end).int()
    r_indices = torch.randint_like(target, args.nb_code)
    input_indices = mask*target+(1-mask)*r_indices
    # 这段有点复杂，你就先记住input_indices是mask后的动作序列

    # Time step masking
    mask_id = get_model(net).vqvae.num_code + 2
    # rand_time = uniform((batch_size,), device = target.device)
    # rand_mask_probs = cosine_schedule(rand_time)
    rand_mask_probs = torch.zeros(batch_size, device = m_tokens_len.device).float().uniform_(0.5, 1)
    # rand_mask_probs = cosine_schedule(rand_mask_probs)
    num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min = 1)
    seq_mask = generate_src_mask(max_len, m_tokens_len+1)
    batch_randperm = torch.rand((batch_size, max_len), device = target.device) - seq_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim = -1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

    # masked_target = torch.where(mask_token, input=input_indices, other=-1)
    masked_input_indices = torch.where(mask_token, mask_id, input_indices)

    att_txt = None # CFG: torch.rand((seq_mask.shape[0], 1)) > 0.1
    # 模型前向传播获得预测
    # seq_mask 是序列掩码，在序列建模任务中，用于指示序列中有效的时间步位置，并掩盖掉填充部分。它的作用是帮助模型在处理不同长度的序列时，专注于序列中的实际内容，而忽略填充的部分。这样可以避免模型在计算损失或进行预测时对无意义的填充部分进行计算
    cls_pred = trans_encoder(masked_input_indices, feat_clip_text, src_mask = seq_mask, att_txt=att_txt, word_emb=word_emb)[:, 1:]

    # [INFO] Compute xent loss as a batch
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_no_end]
    weight_seq_masked = weights[seq_mask_no_end]
    loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
    loss_cls = (loss_cls * weight_seq_masked).sum()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    # 每个间隔打印一次
    if nb_iter % args.print_iter ==  0 :
        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()

        writer.add_scalar('./Loss/all', loss_cls, nb_iter)
        writer.add_scalar('./ACC/every_token', right_seq_masked*100/seq_mask_no_end.sum(), nb_iter)
        
        # [INFO] log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), nb_iter)

        # msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        # logger.info(msg)

    # 在每隔 eval_iter 或最后一次迭代时进行评估（也就是eval，要计算fid、diversity、top等指标），调用 evaluation_transformer 计算评估指标，更新最佳结果
    if nb_iter==0 or nb_iter % args.eval_iter ==  0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        print('正在eval.......')
        # if nb_iter == args.total_iter: #最后一次，之前num_repeat都是正的1，最后一次eval改成-30,但是这么弄会有很多问题？？？很抽象
        #     num_repeat = -30
        #     rand_pos = True
        #     val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

        # evaluation_transformer
        pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, num_repeat=num_repeat, rand_pos=rand_pos)
        # for i in range(4):
        #     x = pose[i].detach().cpu().numpy()
        #     y = pred_pose_eval[i].detach().cpu().numpy()
        #     l = m_length[i]
        #     caption = clip_text[i]
        #     cleaned_name = '-'.join(caption[:200].split('/'))
        #     visualize_2motions(x, val_loader.dataset.std, val_loader.dataset.mean, args.dataname, l, y, save_path=f'{args.out_dir}/html/{str(nb_iter)}_{cleaned_name}_{l}.html')

    # 最后一次，记录结果
    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            