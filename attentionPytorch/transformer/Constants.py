
PAD = 0
UNK = 1
BOS = 2
EOS = 3

# 1. collate_fn : 填充0来对齐多维数组,使得batch内部大小一致, 就是这里直接导致3.中最后一个参数的存在
# 2. batch_pos
# 3. src_word_emb,
PAD_WORD = '<blank>'
# 1. convert_instance_to_idx_seq, 所有编号为1的单词都是数量非常非常少的单词, 全部用<unk>来代替
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
