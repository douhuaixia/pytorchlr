''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    # Tensor.ne与Tensor.eq相反
    # 返回值尺寸64*(max_len-1)*1
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

# n_position的数值为max_token_seq_len+1, 此处为53
# d_hid的数值为d_word_vec, 此处为512
# padding_dix此处为0
# 最终是要把位置向量与原来的词向量相加，所以位置向量的大小与词向量相同, 此处为512
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''
    #  将一个词向量中编号为position的位置映射为一个d_hid维的位置向量，这个向量的第i个元素的数值就是
    # cal_angle(postion, hid_idx), 可以看到第0个位置与第一个位置数值相同，第二个位置与第三个位置
    #　数值相同....
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    # n_position表示句子的长度， pos_i表示单词的位置，get_posi_angle_vec(pos_i)得到一个
    # d_hid维(512维,我对维度的理解可能有问题，尺寸为1*512)的向量

    # 这里为什么是要53呢？
    # 尺寸 53 * 512
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    # 强！
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # 这看起来是53的原因，多出来的第0维全部置为0， 原因是在dataset.py中序列编号中出现了0，而这些0
    #　是没有意义的，所以直接置０?
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

# 这里传入的参数相同，seq_k = seq_q = seq_src， 64*(max_len-1)
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.

    ## len_q = max_len - 1
    len_q = seq_q.size(1)
    # Tensor可以通过eq函数与一个数进行比较，返回值为Tensor，大小与原Tensor相同，
    # 如果原Tensor某个位置的值与该数相等，则返回的Tensor的那个位置数值为1，否则为0
    # 由于不存在单词的位置全部被编码为0，所以padding_mask得到的会是这种形式
    # [[0,0,0,0,..1,1,1], ..., [0,0,0,0,..0,0,0]]
    padding_mask = seq_k.eq(Constants.PAD)
    # unsqueeze函数简单说来就是在已有的维度中在增加一维
    # padding_mask_unsqueeze(1)==> size: 64*1*(max_len-1)
    # .expand(-1,len_q, -1)==> size: 64*(max_len-1)*(max_len-1), max_len-1是复制
    #１那个维度的数值ｍax_len-1次所得
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

# 入口：
# n_src_vocab=n_src_vocab       2911
# len_max_seq=len_max_seq       52
# d_word_vec=d_word_vec         512
# n_layers=n_layers             6
# n_head=n_head                 8
# d_k=d_k                       64
# d_v=d_v                       64
# d_model=d_model               512
# d_inner=d_inner               2048
# dropout=dropout               0.1
class Encoder(nn.Module):
    # 带注意力机制的encoder过程
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        # max_token_seq_len
        # len_max_seq为每组数据的长度上限(每组这里指的是batch_size之一)
        # +1的目的？
        n_position = len_max_seq + 1

        # 放个关于nn.Embedding的链接:https://ptorch.com/news/12.html

        # 假设input尺寸为(M,K)(任意一个数不能超过n_src_vocab-1),
        # 则ouput尺寸为(M,K,d_word_vec),
        # 当使用src_word_emb(input)时，　只要input中的某个word含有Constants.PAD,
        # 那么与那一个word相对应的d_word_vec个数全部设置为０
        # (2911, 512)
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        #  nn.Embedding.from_pretained : 用第一个参数(Tensor)的维度与数值构造并初始化Embedding,
        # freeze = True表示不更新Embedding
        # (53, 512),唯一表示位置信息，且训练过程中不更新
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        # nn.ModuleList，去掉感觉也可以, 待查证
        # n_layers = 6, 6个encoder
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # src_seq : 64*(max_len-1), src_pos : 64*(max_len-1)  64为batch_size
        # self.position_enc是一个定值，src_pos起的是索引的作用
        # enc_output: 64*(max_len-1)*512, 512为词向量的大小
        # , 这个东西才是真正意义上的最底层encoder输入
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        # 堆叠的encoder， N = 6
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

# 入口：
#         n_src_vocab = src_vocab_size              2911
#         n_tgt_vocab = tgt_vocab_size              3149
#         len_max_seq = opt.max_token_seq_len       52
#         tgt_emb_prj_weight_sharing=opt.proj_share_weight   FALSE
#         emb_src_tgt_weight_sharing=opt.embs_share_weight   FALSE
#         d_k=opt.d_k                               64
#         d_v=opt.d_v                               64
#         d_model=opt.d_model                       512
#         d_word_vec=opt.d_word_vec, the size of each embedding vector，default:512
#         d_inner=opt.d_inner_hid                   2048
#         n_layers=opt.n_layers                     6
#         n_head=opt.n_head                         8
#         dropout=opt.dropout                       0.1
class Transformer(nn.Module):
    # 带注意力机制的seq2seq模型
    # torch.nn.Module : Base class for all neural network modules
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        # 丢弃最后一个列，tgt_seq丢掉的东西是</s>, tgt_pos大小与tgt_seq大小保持一致
        # 为什么丢弃?
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
