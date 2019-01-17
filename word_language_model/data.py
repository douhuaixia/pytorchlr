import os
from io import open
import torch

class Dictionary(object):
    # 这和我之前尚未看完的attention有类似的数据处理操作
    # 想想这样做的好处是什么？
    def __init__(self):
        # word与index的唯一性

        # dict: word : index
        self.word2idx = {}
        # list: index : word
        self.idx2word = []

    def add_word(self, word):
        # 添加word, 包括重复性检查
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    # 这个类中最重要的东西之一是self.dictionary()记录了word与index之间的双射关系
    # 另一个是self.train、self.valid、self.test，这三个均为Tensor变量，每个Tensor都记录
    # 各自数据index的一维矩阵形式。
    def __init__(self, path):
        # self.dictionary充当下面三句代码的全局变量, 实现三个数据文件的同步
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            # tokens用来记录数据中word的数量(包括每行的<eos>)
            tokens = 0
            for line in f:
                # words是一个列表，其中的元素是sentence中的word, 以<eos>作为行与行的分割
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # 代码运行到这里可以确保数据中的所有word以及索引全部被记录在self.dictionary中，
        # 并且是唯一的

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            # torch.Tensor(n)会生成尺寸为 n 的一维矩阵, 数值是随机的
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    # 通过之前记录的word字典来为ids中的每个元素赋值
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # 到这里可以知道其实整个过程就是把sentence中的word转化为数字
        return ids
