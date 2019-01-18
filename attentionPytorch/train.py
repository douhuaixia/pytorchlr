'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    # argparse为命令行编程封装好的库文件
    parser = argparse.ArgumentParser()

    # -data参数必须有
    parser.add_argument('-data', required=True)
    # -epoch参数可选，类型为int，缺省数值为10
    parser.add_argument('-epoch', type=int, default=10)
    # -batch_size参数可选，类型为int，缺省数值为64
    parser.add_argument('-batch_size', type=int, default=64)
    #parser.add_argument('-d_word_vec', type=int, default=512)
    # -d_model参数可选，类型为int，缺省数值为512
    parser.add_argument('-d_model', type=int, default=512)
    # -d_inner_hid参数可选，类型为int，缺省数值为2048
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    # -d_k参数可选，类型为int，缺省数值为64
    parser.add_argument('-d_k', type=int, default=64)
    # -d_v参数可选，类型为int，缺省数值为64
    parser.add_argument('-d_v', type=int, default=64)
    # -n_head参数可选，类型为int，缺省数值为8
    parser.add_argument('-n_head', type=int, default=8)
    # -n_layers参数可选，类型为int，缺省数值为6
    parser.add_argument('-n_layers', type=int, default=6)
    # -n_warmup_steps参数可选，类型为int，缺省数值为4000
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    # -dropout参数可选，类型为float，缺省数值为0.1
    parser.add_argument('-dropout', type=float, default=0.1)
    # -embs_share_weight参数可选， 有为true，无为false
    parser.add_argument('-embs_share_weight', action='store_true')
    # -proj_share_weight参数可选， 有为true，无为false
    parser.add_argument('-proj_share_weight', action='store_true')
    # -log参数可选， 缺省值为None
    parser.add_argument('-log', default=None)
    # -save_model参数可选， 缺省值为None
    parser.add_argument('-save_model', default=None)
    # -save_mode参数可选， 类型为str，缺省值为best，未省略则必须在all与best中选择
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    # -no_cuda参数可选， 有为true，无为false, 该参数指定是否使用gpu
    parser.add_argument('-no_cuda', action='store_true')
    # -label_smoothing参数可选， 有为true，无为false
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    # cuda为true表示可以使用，为false则不用gpu, 原来的no_cuda依然存在
    opt.cuda = not opt.no_cuda
    # d_word_vec 为d_model的数值
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    # 载入数据load函数使用pickle模块的反序列化功能来把文件中的对象读取到内存
    # data对象存储一个大字典，有4项
    # 1 settings: 是一个argparse中的Namespace
    # 2 dict: {src: 2911组， tgt: 3149组}, 每个组是word以及它的编号, 2911组在一个dict中
    # 3 train:{src:29000组数字,tgt：29000组数字}, 每一组表示一句sentence包含的所有word的编号, 每一组在一个list中
    # 4 valid:{src:1014组数字， tgt：1014组数字}
    data = torch.load(opt.data)
    # 获取每句话的最大token数量
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)
    # 所有不重复单词的数量
    # 2911
    opt.src_vocab_size = training_data.dataset.src_vocab_size
    # 3149
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device ,opt)

# 入口：载入的data、命令行传入的参数opt
# 出口： train_data, validation_data
def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    # num_workers表示线程数量
    # collate_fn，是用来处理不同情况下的输入dataset的封装，
    # 一般采用默认即可，除非你自定义的数据读取输出非常少见
    # 跳过collate_fn
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
