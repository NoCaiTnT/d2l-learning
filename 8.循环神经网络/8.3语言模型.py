# 语言模型
#   给定文本序列x_1, x_2, ..., x_T, 语言模型的目标是估计联合概率p(x_1, x_2, ..., x_T)
#   它的应用包括:
#       1. 做预训练模型(BERT, GPT-3)
#       2. 生成文本, 给定前面几个词, 不断的使用x_t~ p(x_t | x_1, x_2, ..., x_(t-1))来生成后续文本
#       3. 判断多个序列中哪个更常见, 'to recognize speech'和'to wreck a nice beach'
# 使用计数来建模
#   假设序列长度为2, 我们预测p(x, x') = p(x)p(x' | x) = n(x)/n × n(x,x')/n(x)
#       这里n是总词数, n(x), n(x, x')是单个单词和连续单词对的出现次数
#   很容易扩展到长为3的情况
#       p(x, x', x'') = p(x)p(x' | x)p(x'' | x, x') = n(x, x', x'')/n(x, x') × n(x, x')/n(x) × n(x)/n
# N元语法(一次看N个词)
#   当序列很长时, 因为文本量不够大, 很可能n(x_1, x_2, ..., x_T) <= 1
#   使用马尔科夫假设可以缓解这个问题, 即假设当前词只和前面的n-1个词相关
#       一元语法(τ=1-1=0): p(x_1, x_2, x_3, x_4) = p(x_1)p(x_2)p(x_3)p(x_4) = n(x_1)/n × n(x_2)/n × n(x_3)/n × n(x_4)/n
#       二元语法(τ=2-1=1): p(x_1, x_2, x_3, x_4) = p(x_1)p(x_2 | x_1)p(x_3 | x_2)p(x_4 | x_3)
#                                      = n(x_1)/n × n(x_2 | x_1)/n(x_1) × n(x_3 | x_2)/n(x_2) × n(x_4 | x_3)/n(x_3)
#       三元语法(τ=3-1=2): p(x_1, x_2, x_3, x_4) = p(x_1)p(x_2 | x_1)p(x_3 | x_1, x_2)p(x_4 | x_2, x_3)
#   好处: 可以处理很长的序列, 但是参数空间会变得很大, 例如二元语法的参数空间为O(n²), 三元语法的参数空间为O(n³), 可以根据词频进行剪枝
# 总结
#   语言模型估计文本序列的联合概率
#   使用统计方法时常采用N元语法

import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

# 最流行的词 被称为 停用词 画出的词频图
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
d2l.plt.show()

# 其他的词元组合, 比如二元语法, 三元语法等等, 又会如何
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]     # 当前元素和下一个元素组成的元组   滑动窗口
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show()

# 随机地生成一个小批量数据的特征和标签以供读取, 在随机采样中, 每个样本是原始序列上任意截取的一段子序列
# num_steps是每个样本的最大长度(即取多少个token进行预测下一个 ), batch_size是小批量的样本数
# 随机从0-T之间选择一个起始点, 然后将后面的序列每T个切一次, 然后随机从这些长度为T的序列中选择num_batch个序列
# Y和X的长度都是num_steps, Y是X的后一个元素
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps    # 一共有多少个子序列
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))    # 每个子序列的起始索引
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size         # 例如: 一共10个子序列, batch_size=3, 那么就有3个batch
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 生成一个从0到34的序列, 时间步数为5, 这意味着可以生成[(35-1)/5] = 6个“特征－标签”子序列对, 如果设置小批量大小为2，我们只能得到3个小批量
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# 顺序分区:保证两个相邻的小批量中的子序列在原始序列上也是相邻的
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# 将上面的两个采样函数包装到一个类中， 以便稍后可以将其用作数据迭代器
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# 定义了一个函数load_data_time_machine， 它同时返回数据迭代器和词表
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab