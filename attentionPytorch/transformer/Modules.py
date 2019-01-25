import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

# temperature = 8
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # nn.Softmax的dim，只需记得是在最后一个维度上做Softmax，这样得到的结果总是
        # 正确的
        self.softmax = nn.Softmax(dim=2)

    # q, k, v = (8*64)*max_len*64
    # mask = (512, max_len, max_len)
    def forward(self, q, k, v, mask=None):

        # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        # batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        # If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor,
        # out will be a (b×n×p) tensor.
        # 对于每一个batch来做类似与矩阵点乘的操作

        # q:(batch*h, m, d_k)
        # k:(batch*h, m, d_k)
        # v:(batch*h, m, d_v)
        # k.transpose(batch*h, d_k, m)
        # attn.size() : (batch*h, m, m)
        attn = torch.bmm(q, k.transpose(1, 2))

        # 除以根号...
        attn = attn / self.temperature

        # masked_fill: 如果ｍask的位置为1，那么attn的对应位置的数值为-np.inf, 即负无穷
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # 做softmax
        attn = self.softmax(attn)
        # 做dropout
        attn = self.dropout(attn)
        # 继续点乘
        output = torch.bmm(attn, v)
        # output:(batch*h, m, d_v)
        return output, attn
