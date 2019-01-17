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
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # RNN网络
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
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        # 全连接神经网络：参数为 nhid×ntoken
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

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
