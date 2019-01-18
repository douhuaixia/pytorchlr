import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    # *代表什么意思？
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    # insts应该指的是src/tgt_insts, 获取所有sentence中word数量最多的数值
    max_len = max(len(inst) for inst in insts)

    # 通过填充0来对齐多维数组
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    # 相应的word索引变为
    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos

class TranslationDataset(torch.utils.data.Dataset):
    # 这里似乎是核心结构，成员变量有src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word
    # _src_word2idx, _src_idx2word, _tgt_word2idx, _tgt_idx2word,
    # _src_insts, _tgt_insts
    def __init__(
        self, src_word2idx, tgt_word2idx,
        src_insts=None, tgt_insts=None):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        # 单词：编号
        self._src_word2idx = src_word2idx
        # 编号：单词
        self._src_idx2word = src_idx2word
        # 长度表示sentence数量， 通过索引访问每一个sentence，通过二重索引访问sentence中某个
        # word的编号
        self._src_insts = src_insts

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    # dataset是内置的变量
    def __len__(self):
        # 这个函数使得len(dataset)可以获得数据集的长度
        return self.n_insts

    def __getitem__(self, idx):
        # 这个函数使得dataset[i]可以获得(_src_insts[i], _tgt_insts[i])
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]
