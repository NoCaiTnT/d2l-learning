# 机器翻译与数据集

import os
import torch
from d2l import torch as d2l

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                            '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

def preprocess_nmt(text):
    """预处理“英语－法语”数据集。"""
    # 将标点符号前面加空格, 便于后面的分词
    # 全转为小写字母
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

# 将句子变成token
def tokenize_nmt(text, num_examples=None):
    """标记化文本序列。"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
print(source[:6], target[:6])

# 绘制每个文本序列所包含的标记数量的直方图
# 查看句子长度直方图
d2l.set_figsize()
_, _, patches = d2l.plt.hist([[len(l) for l in source], [len(l) for l in target]],
                            label=['source', 'target'])
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right')
d2l.plt.show()

# 词汇表
# <pad>: 用来填充短句子
# <bos>: 句子的开始
# <eos>: 句子的结束
# 一个句子的长度小于2就抛弃
src_vocab = d2l.Vocab(source, min_freq=2,
                        reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))

# 序列样本都有一个固定的长度 截断或填充文本序列
# 前面8.3的语言模型中, 通过设定num_steps来固定序列长度, 即用几个词来预测下一个词
# 但是在机器翻译中, 一个句子的长度是不固定的, 所以需要填充或截断
# 参数为: 一个句子, 序列长度, 填充的token
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列。"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

# 转换成小批量数据集用于训练
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量。"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]   # 给每个句子加上结束符
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])  # 填充
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # 计算每个句子的有效长度
    return array, valid_len

# 训练模型
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词汇表。"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                            reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                            reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

# X是原句子, 即source, 英语, Y是目标句子, 即target, 法语
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('valid lengths for Y:', Y_valid_len)
    break