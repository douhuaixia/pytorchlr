''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        # sent为一行, 包括换行符
        for sent in f:
            # 如果不对大小写敏感的话, 则一律转化为小写
            if not keep_case:
                sent = sent.lower()
            # split()函数分割字符串，去掉了换行符号，去掉了空格，返回一个分割后的列表
            words = sent.split()
            # 每行单词的数量大于预先设置的最大值的情况下
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            # 丢弃多余的
            word_inst = words[:max_sent_len]

            # 使用<s>与</s>来作为句子的界, 嵌套列表
            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    # 输出提示信息表明有些token被丢弃了, 可能需要增大max_sen_len数值
    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''
    # 拆分为word, 变为集合去掉重复, 即src/tgt中所有不重复word的集合
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}
    # 所有word数量初始化为0
    word_count = {w: 0 for w in full_vocab}
    # 统计word数量
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0

    # word_count中存放的是word: count
    # word2idx
    for word, count in word_count.items():
        if word not in word2idx:
            # 某个word的数量大于给定的min_word_count数量
            if count > min_word_count:
                # 该word编号为当前长度，所有count > min_word_count的编号不会重复
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1
    # 所有word_count小于min_word_count的word都被丢弃掉

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    # 返回之前我需要保证word2idx的 key-value 之间是双向唯一的
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    # 获取word_insts中每个单词的编号, 没有的话编号为1
    # 也就是说所有编号为1的单词都是无效的单词
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=True)

    # 可选参数max_len, 存储时用max_word_seq_len来存储，默认是50,
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    # max_word_seq_len应该指的是每行存储word的最大数量
    # max_token_seq_len指的是每行存储token的最大数量
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    # train_src_word_insts的格式， [[], [], [], []]...   len:29000, 里面的每一个元素
    # 为一句话中的 <s> + word0 + word1 + word2 + ... + </s>
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    # train_tgt_word_insts与train_src_word_insts类似
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    # 长度应该是相等的，但对应位置元素的个数是不等的
    if len(train_src_word_insts) != len(train_tgt_word_insts):
        # 此处的处理方法是否正确？ 正确
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    # 去掉了None而且元素个数保持相等
    # zip(*zipped)是用来解压的, 仅仅用来移除None
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        # 猜测判断'dict'是否是predefined_data的关键字
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        # 取出原来的数据
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
