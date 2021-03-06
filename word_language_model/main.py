# coding: utf-8
# Tensor中的data属性是递归的，why？
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
# 这里我为了调试把默认的改为RNN_RELU
parser.add_argument('--model', type=str, default='RNN_RELU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
# 手动设置随机数种子以获得高可重复性, 那么参数中seed的意义何在?
torch.manual_seed(args.seed)
# cuda相关
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
# 这里应该写一些代码来检验cuda是否可用

###############################################################################
# Load data
###############################################################################


# arg.data参数为corpus的目录
corpus = data.Corpus(args.data)
# corpus.dictionary : 33278对word <==> index
# corpus.train : 2088628
# corpus.valid : 217646
# corpus.test : 245569

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns(列) are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    # Tensor.size(n), 获取第n维的大小，0代表第一个维度
    # 此处数据为一维Tensor

    # bsz表示batch的大小
    # nbatch表示batch的数量
    # 注意这里用了整除符号，可能会有多余的数据，这将在下面处理
    nbatch = data.size(0) // bsz
    # Tensor.narrow(dim, start, length)函数说明：
    # 对与2*3*4*5的矩阵而言， dim范围为[-4,3], 0表示取第一维也就是2这个数字对应的
    # 1表示取第二维也就是3这个数字对应的，3表示第三维也就是4这个数字对应的...
    # 在特定维度里取的范围为[start,start+length)
    # 例如：
    # x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])为2*3的矩阵
    # torch.narrow(x, 0, 0, 2)===>
    # tensor([[ 1,  2,  3],
    #         [ 4,  5,  6]])
    # torch.narrow(x, 1, 1, 2)===>
    # tensor([[ 2,  3],
    #         [ 5,  6],
    #         [ 8,  9]])

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # 截取多余的数据，只留下 batch大小*batch数量 的数据
    # 从第一个维度截取顺序不会乱掉
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # view这个函数，基本上可以进行Tensor的reshape操作, -1表示维度自动推断
    # t()这个函数实现二维矩阵的转置功能, 注意输入必须是二维矩阵，2*3转换为3*2

    # 需要注意的是reshape与转置有着明显的顺序不同, 比如下面的例子尺寸相同，但是顺序不同
    # x = torch.Tensor(2,3)
    # x.reshape(3,2) != x.t()

    # data.view(bsz, -1)这里batch取得是列而不是行
    # 之后调用.t()函数做了一下转置batch变为行

    # 所以问题来了，为什么不直接reshape为(nbatch, bsz)?
    # 好像是刻意打乱顺序
    # 1 2 3 4 5 6 7 8 9 10 11 12
    # == == == == == == == == == == == == == ==
    # call view()
    # bsz = 4
    # nbatch = 3
    # 1   2   3
    # 4   5   6
    # 7   8   9
    # 10  11  12
    # == == == == == == == == == == == == == ==
    # call t()
    # 1   4   7   10
    # 2   5   8   11
    # 3   6   9   12
    # == == == == == == == == == == == == == ==
    # contiguous()?
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
# 对训练数据而言，cl中传入了batch_size, 默认为20
# 验证集与测试集batch大小均为10

# type(train_data)==>Tensor, size==>104431*20
train_data = batchify(corpus.train, args.batch_size)
# type(val_data)==>Tensor, size==>21764*10
val_data = batchify(corpus.valid, eval_batch_size)
# type(test_data)==>Tensor, size==>24556*10
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

# train+val+test的所有tokens(不重复word)的长度, 说白了就是word与index之间的映射的长度
ntokens = len(corpus.dictionary)

# args.model为网络类型RNN_TANH or RNN_RELU or LSTM or GRU
# args.emsize : size of word embeddings
# args.nhid : number of hidden units per layer 每层的隐藏单元数量
# args.nlayers: number of layers 网络层数
# args.dropout : dropout applied to layers (0 = no dropout) 等会查什么意思
# args.tied : tie the word embedding and softmax weights 等会查什么意思

# RNNModel就是自己构建的网络
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
# useful when training a classification problem
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        # detach()返回一个新的Tensor，新的Tensor不参与梯度计算，换言之，拷贝一个Tensor
        # 并使其从计算图中分离
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    # 用len函数来测量的是矩阵的第一个维度大小
    # 所以说每次操纵的数据为args.bptt的整倍数
    # 一般来说，seq_len == args.bptt, 但是到最后不一定
    seq_len = min(args.bptt, len(source) - 1 - i)
    # 最后取得的数据可能不是恰好为args.bptt大小的batch
    data = source[i:i+seq_len]
    # target是干嘛的？
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    # 2*10*200
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            # data:35*10, targets:350
            data, targets = get_batch(data_source, i)
            # 参数：(35*10, 2*10*200), output=(35,10,33278)  hidden=(2,10,200)
            output, hidden = model(data, hidden)
            # output_flat == > 350*33278
            output_flat = output.view(-1, ntokens)
            # len(data)=35, 称为sequence length
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    # 33278
    ntokens = len(corpus.dictionary)
    # 以batch_size大小初始化隐藏层参数
    # 以下是在RNN的条件下，当在LSTM的情况下时，hidden为tuple
    # type(hidden) == > Tensor, hidden.size() == > [2, 20, 200]
    # args.batch_size为20
    hidden = model.init_hidden(args.batch_size)
    # args.bptt: sequence length
    # train_data.size(0)表示总共有多少个batch(与下面代码中的batch不同), train_data.size(1)
    # 表示batch_size的大小, bptt是序列长度, 也就是一次取多少个batch
    count = 0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        # type(data)==>Tensor, data.size()===> 35*20, not zero
        # type(targets)==>Tensor, targets.size()===> 700, not zero
        # args.bptt is 35
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # 参数中的hidden的size表示为(nlayers, bsz, nhid)
        # 此时得到的hidden算是原来hidden的副本
        hidden = repackage_hidden(hidden)
        # Sets gradients of all model parameters to zero.
        model.zero_grad()
        # 参数：(35*20, 2*20*200), output=(35,20,33278)  hidden=(2,20,200)
        # 35被称为sequence lenegth   20被称为batch_size   33278被称为num_directions*hidden_size
        # 上面的话可能有问题
        output, hidden = model(data, hidden)
        # output.view(-1, ntokens) == > 700*33278
        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        print("Done!")
        count += 1
        if count == 10:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
# 默认学习率是20
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
# epoch指的是训练轮数
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        # 完成一轮训练
        train()
        # 在val_data上进行评估，得出损失值
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        # not None ===> True
        # 第一次直接进行保存，之后如果val_loss的值比之前的更小则保存当前训练的模型
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            # 模型效果变差则立即降低学习率为原来的1/4
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
