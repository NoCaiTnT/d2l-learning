# 机器翻译
#   给定一个源语言的句子, 自动翻译成目标语言
#   这两个句子可以有不同的长度

# Seq2Seq
#   编码器是一个RNN, 读取输入句子
#       可以是双向(句子长度确定, 可以看到整个句子, 不需要预测)
#       将最后一个隐藏状态给解码器
#   解码器使用另一个RNN来输出
#       通过<bos>开始, 输出到<eos>结束

# 编码器-解码器细节
#   编码器是没有输出的RNN
#   编码器最后时间步的隐状态用作解码器的初始隐状态
#       Encoder                 Decoder
#                                Dense
#                                  ↑
# n × Recurrent Layer   →     Recurrent Layer × n
#          ↑                       ↑
#      Embedding                Embedding
#          ↑                       ↑
#       Sources                  Targets(训练时课件, 推理时只输入<bos>)

# 训练
#   训练时解码器使用目标句子作为输入
#   即在解码时, 即使上一个词的预测是错误的, 下一个词的输入也是正确的

# 评价函数BLEU
# 衡量生成序列的好坏的BLEU
#   p_n是预测中所有n-gram的精度(n-gram是n个连续的词)
#   例如: 标签序列为A B C D E F和预测序列为A B B C D
#       p_1 = 4/5, p_2 = 3/4, p_3 = 1/3, p_4 = 0
#       分母是预测中n-gram的数量, 分子是预测中n-gram和标签中n-gram的交集数量(和顺序无关)
# BLEU定义(越大越好, 最大为1)
#   BLEU = exp(min(0,1 - (len_label/len_pred))) * ∏_n=1^k p_n^(1/(2^n))
#                       ↑                                   ↑
#                  惩罚过短的预测                         长匹配有高权重
#       (如: 只预测一个词, 精度很高, 因此需要给大的惩罚)    (越长的词预测准确, 权重越大)

# 总结
#   Seq2Seq从一个句子生成另一个句子
#   编码器和解码器都是RNN
#   将编码器最后时间隐状态来初始化解码器隐状态来完成信息传递
#   常用BLEU来衡量生成序列的好坏

# 实现
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

# Encoder
class Seq2SeqEncoder(d2l.Encoder):
    # vocab_size: 词典大小(one-hot向量的长度)
    # embed_size: 词向量长度(将每个词映射到一个向量)
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state

# 实现
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)   # X在这里为英语
output, state = encoder(X)
print(output.shape)     # 7,4,16: 时间步(句子长度为7, 有7个时间步), 批量大小, 隐藏单元数
print(state.shape)      # 2,4,16: 隐藏层层数, 批量大小, 隐藏单元数

# Decoder
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)   # 假设编码器和解码器的隐藏单元数相同, 加法就是输入+encoder的输出
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # 拿出encoder的最后的state
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        # 最后一个时间步的最后一层的隐藏状态, 重复输入X的时间步次数
        a = state[-1]
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)   # X在这里为法语
print(output.shape)     # 4,7,10: 批量大小, 时间步, 词典大小
print(state.shape)      # 2,4,16: 隐藏层层数, 批量大小, 隐藏单元数

# 通过零值化屏蔽不相关的项
# 将填充的标记的值设置为0
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# 通过拓展softmax交叉熵损失函数来遮蔽不相关的预测
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

loss = MaskedSoftmaxCELoss()
# (3,4,10): 批量大小, 时间步, 词典大小
print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))

# 训练
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 给decoder的输入加上<bos>, 使其强制学习<bos>为第一个词
            # 相当于<bos>你好世 ->  hello world
            # 将句子的最后一个词<eos>去掉
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)   # 去掉最后一个词
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
    d2l.plt.show()

# 训练过程
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# 预测
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]  # 将输入句子转换为词元索引, 并添加<eos>
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])        # 将每个句子的长度固定为num_steps, 不足的填充<pad>
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)   # 将输入转成张量
    enc_outputs = net.encoder(enc_X, enc_valid_len)                                             # 编码器输出
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)                              # 解码器的初始状态
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0) # 解码器的输入, 指定为句子开始<bos>
    output_seq, attention_weight_seq = [], []
    # 循环预测: 预测到句子结束符<eos>结束, 或者达到最大输出长度
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """计算 BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# 将几个英语句子翻译成法语
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    bleu_score = bleu(translation, fra, k=2)
    print(f'{eng} => {translation}, bleu {bleu_score:.3f}')
    