''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

# n_head : 8
# d_model : 512
# d_k : 64
# d_v : 64
# dropout : 0.1
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Q投影矩阵, 大小为(d_model, h*d_k)
        # w_qs, (512, 8*64)
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        # K投影矩阵, 大小为(d_model, h*d_k)
        # w_ks, (512, 8*64)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        # V投影矩阵, 大小为(d_model, h*d_v)
        # w_vs, (512, 8*64)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # 三个全连接层参数初始化
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        #
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    # q, k, v = enc_input : 64*max_len*512（batch*m*d_model）
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # 获取三个维度的尺寸
        # 此处： 64， max_len-1, 512
        # 对应： batch, m, d_model
        # batch独立于运算

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # 途中没有经过multi_head attention, 而是到达ADD&NORM的那条线
        residual = q

        # 64*max_len*(8*64)
        # w_qs(q) w_ks(k)  尺寸为(batch, m, h*d_k)
        # w_vs(v)  尺寸为(batch, m, h*d_v)

        # 重新排列不会影响batch、m的位置
        # 现在给定一个点(a, b, c, d), 表示的含义是一个batch中第a句话，第b个单词，第c次attention，结果为d维度
        # 这里画图可以理解
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # (batch, m, h, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # (batch, m, h, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # (batch, m, h, d_v)

        # view(batch, m*h*d_k)
        # https://www.zhihu.com/question/43594110, 转置与重新排列还是有区别的，
        # view函数用来重新排列，他的参数是重新排列之后每一个维度的大小
        # transpose用来转置，参数为维度, 且参数为两个，作用是转置这两个维度, 这个一般用于二维数组
        # permute也可以用来转置, 参数为维度，参数为多个，作用是转置多个维度，这个一般用于多维数组
        # 转置有什么用处？

        # contiguous为调用它的非contiguous张量分配内存空间, 官方文档没有提到，permute的
        # 结果是non-contiguous的，因而这里一定要调用contiguous方法
        # 如果不调用的话，

        # 很难理解, 唉! permute与view结合是什么操作
        #　卡在这好长时间，暂时不看了
        # 转置的话，只不过是交换一些位置，permute之后，(a, b, c, d)表示的含义是第a次attention、一个batch中的第b句话，第c个单词，结果为d维度
        # 改天把整个流程图画出来，
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        # mask:(64, max_len, max_len)
        # 执行完下面这句变为(512, max_len, max_len)
        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        # attn, output(batch*h, m, d_v), attn是没有与v相乘之前的
        # self-attention
        output, attn = self.attention(q, k, v, mask=mask)

        # view之后相对位置依旧不变，
        # (h, batch, m, d_v )
        output = output.view(n_head, sz_b, len_q, d_v)
        # (batch, m, h*d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        # 全连接层+dropout层
        # (batch, m, d_model)
        output = self.dropout(self.fc(output))
        # 反正维度不变
        output = self.layer_norm(output + residual)
        # attn的作用？
        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
