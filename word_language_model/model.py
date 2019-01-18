import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    # 容器模块包含encoder、RNN、decoder
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        # nn.Dropout()简单来说是在forward过程中以概率p随机的把某些输入张量中的元素清0
        # 目的：regularization and preventing the co-adaptation of neurons， 有篇论文专门讲这个
        # self.drop(Tensor)==>按概率p随机清0的Tensor
        self.drop = nn.Dropout(dropout)

        # nn.Embedding : be often used to store word embeddings and retrieve them using indices.
        # 文档上写得很清楚
        # The input to the module is a list of indices,
        # and the output is the corresponding word embeddings.
        # 这里只讲前两个参数： 第一个参数为size of the dictionary of embeddings
        # 第二个参数为size of word embeddings
        # 也就是说每次从总数为ntoken的word embeddings中取ninp个word embeddings
        # 把高维向量encode为低维向量？
        # word embedding指的是：word <==> index ?
        # 33278, 200
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # RNN网络, 这里不是太懂
            # 注意是最终训练的就是这些features
            # 第一个参数为input_size – The number of expected features in the input x
            # 第二个参数为hidden_size – The number of features in the hidden state h
            # 第三个参数为num_layers – Number of recurrent layers
            # drop参数If non-zero, introduces a Dropout layer on the outputs of each RNN
            # layer except the last layer, with dropout probability equal to dropout. Default: 0

            # RNN网络输入
            # input_X格式为(seq_len, batch, input_size), input_size最后一个维度的数值为RNN的第一个参数
            # h_0格式为(num_layers * num_directions, batch, hidden_size),
            # hidden_size最后一个维度的数值为RNN的第二个参数
            # If the RNN is bidirectional, num_directions should be 2, else it should be 1.
            # RNN网络输出
            # output : (seq_len, batch, num_directions * hidden_size)
            # h_n : (num_layers * num_directions, batch, hidden_size)
            # 200, 200, 2, dropout = 0.2
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        # 全连接神经网络： 与encoder的参数相反
        # 低维恢复为高维？
        # 200, 33278
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        # encoder、decoder的权重、偏置项初始化
        self.init_weights()

        # rnn_type
        self.rnn_type = rnn_type
        # nhid为每层的隐藏单元数量
        self.nhid = nhid
        # nlayers为层数
        self.nlayers = nlayers

    # 初始化权重weight
    def init_weights(self):
        initrange = 0.1
        # encoder过程的权重初始化

        # type(encoder.weight)===>torch.nn.parameter.Parameter为Tensor的子类, 当其
        # 作为module的属性出现时会被自动加入参数列表中，通过parameters()迭代器可以访问, 而
        # Tensor不会。

        # type(encoder.weight.data)===> Tensor
        # Tensor.uniform：使用连续均匀分布填充Tensor
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # decoder过程的偏置项初始化为0

        # encoder没有bias属性, 为什么？
        self.decoder.bias.data.zero_()
        # decoder过程的权重初始化
        # 下面的类型与上面所述相同
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # 前向传播过程
    def forward(self, input, hidden):
        # encoder为第一步，其次drop为第二步
        # 需要弄清楚input的维度
        emb = self.drop(self.encoder(input))
        # rnn为第三步
        output, hidden = self.rnn(emb, hidden)
        # 对输出做概率清0
        output = self.drop(output)
        # 输出的维度应该是三，这里reshape为二之后调用decoder过程
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # reshape为
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        # self.parameters()获取该模型的所有权重以及偏置项 11
        # next从迭代器中取第一项, 问题是第一项是什么？分析可知第一项的类型为nn.parameter.Parameter
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
            # new_zeros:返回一个self.nlayers * bsz * self.nhid大小的全０ Tensor
            # 此处大小为2*20*200
            # 注意返回值类型为Tensor
            # 问题是，这里为什么需要借助于weight?
            # the returned Tensor has the same torch.dtype and torch.device as this tensor.
