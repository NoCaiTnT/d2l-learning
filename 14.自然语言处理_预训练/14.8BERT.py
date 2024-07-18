# NLP里的迁移学习
#   使用预训练好的模型来抽取词, 句子的特征
#       如word2vec或语言模型
#   不更新预训练好的模型
#   需要构建新的网络来抓取新任务需要的信息
#       word2vec忽略了时序信息
#       语言模型只看了一个方向

# BERT的动机
#   基于微调的NLP模型
#   预训练的模型抽取了足够多的信息
#   新的任务只需要添加一个简单的输出层

# BERT架构
#   只有编码器的Transformer
#   两个版本:
#       BERT-Base: blocks = 12, hidden_size = 768, heads = 12, parameters = 110M
#       BERT-Large: blocks = 24, hidden_size = 1024, heads = 16, parameters = 340M
#   在大规模数据上训练 > 3B 词

# 对输入的修改
#   每个样本是一个句子对(对应于训练的时候 encoder的输入 和 decoder的输入)
#   加入额外的片段嵌入
#   位置编码可学习
#   句子对开始是<cls>, 句子间分隔是<sep>, 句子对结束是<sep>
#   使用segment embedding来区分两个句子
#   position embedding         E0     E1       E2      E3     E4       E5     E6    E7      E8     E9
#                              +      +        +       +      +        +      +     +       +      +
#   segment embedding          EA     EA       EA      EA     EA       EA     EB    EB      EB     EB
#                              +      +        +       +      +        +      +     +       +      +
#   token embedding         E_<bos>  E_this  E_movie  E_is  E_great  E_<sep>  E_i  E_like  E_it  E_<sep>
#                             <cls>   this    movie    is    great    <sep>   i    like    it    <sep>

# 预训练任务1: 带掩码的语言模型(Masked Language Modeling, MLM)
#   Transformer的编码器是双向, 标准语言模型要求单向
#   带掩码的语言模型每次随机(15%概率)将一些次元替换成<mask>
#       为什么在训练的时候需要mask掉一些词?
#           BERT是一个双向Transformer模型，通过掩盖输入中的一些token，模型被迫从两侧（前后）同时考虑上下文信息。这与传统的语言模型（如GPT）只从左到右或者从右到左单向预测不同。双向的上下文信息使得BERT可以更好地理解和生成文本。
#           随机掩盖token并预测它们的位置和内容，可以使模型更好地学习词汇之间的关系，从而增强模型的泛化能力。这有助于模型在下游任务中表现得更好，因为它已经在多种上下文中学会了如何推断出缺失的信息。
#   因为微调任务中不出现<mask> (即自己的数据集中不会出现<mask>), 所以需要在微调的时候加入<mask>
#       80%概率下, 将选中的词元替换成<mask>
#       10%概率下, 将选中的词元替换成一个随机词元
#       10%概率下, 保持选中的词元不变

# 预训练任务2: 下一句子预测(Next Sentence Prediction, NSP)
#   预测一个句子对中 两个句子是不是相邻
#   训练样本中:
#       50%概率选择相邻句子对: <cls>this movie is great<sep>i like it<sep>
#       50%概率选择随机句子对: <cls>this movie is great<sep>hello world<sep>
#   将<cls>对应的输出放到一个全连接层来预测

# 总结
#   BERT针对微调设计
#   基于Transformer的编码器做了如下修改
#       模型更大, 训练数据更多
#       输入句子对, 片段嵌入, 可学习的位置编码
#       训练时使用两个任务:
#           带掩码的语言模型
#           下一句子预测

# BERT代码
import torch
from torch import nn
from d2l import torch as d2l

# 输入表示
# 给一个或者两个句子, 将其变成BERT的输入: 加上<cls>和<sep>的词元token 以及 用于区分两个句子的segment(第一个句子为0, 第二个句子为1)
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

# encoder
#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

# 假设词表大小为10000，为了演示BERTEncoder的前向推断，让我们创建一个实例并初始化它的参数。
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)

# 我们将tokens定义为长度为8的2个输入序列，其中每个词元是词表的索引。
# 使用输入tokens的BERTEncoder的前向推断返回编码结果，其中每个词元由向量表示，其长度由超参数num_hiddens定义。此超参数通常称为Transformer编码器的隐藏大小（隐藏单元数）。
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
print(encoded_X.shape)

# BERT预训练任务
# 掩蔽语言模型(Masked Language Modeling, MLM)
#@save
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # 简短点说, 就是把要预测位置的词所对应的encoder的输出拿到这里, 然后通过比较简单的线性层来预测该位置的值
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

# 为了演示MaskLM的前向推断，我们创建了其实例mlm并对其进行了初始化。
# 来自BERTEncoder的正向推断encoded_X表示2个BERT输入序列。
# 我们将mlm_positions定义为在encoded_X的任一输入序列中预测的3个指示。
# mlm的前向推断返回encoded_X的所有掩蔽位置mlm_positions处的预测结果mlm_Y_hat。对于每个预测，结果的大小等于词表的大小。
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
print(mlm_Y_hat.shape)

# 通过掩码下的预测词元mlm_Y的真实标签mlm_Y_hat，我们可以计算在BERT预训练中的遮蔽语言模型任务的交叉熵损失
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
a = mlm_Y_hat.reshape((-1, vocab_size))     # 这里是预测的结果(batch, num_pred_positions, vocab_size) 根据第三维vocab_size就可以得到每个词元的概率
b = mlm_Y.reshape(-1)                       # 这里是真实的结果(batch, num_pred_positions) 根据第二维num_pred_positions就可以得到每个词元的真实值, 用真实值 和 预测的词元的所有概率 来计算交叉熵
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
print(mlm_l.shape)

# 下一句预测任务(Next Sentence Prediction, NSP)
# 二分类问题, 句子相邻 和 句子不相邻
#@save
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)

# NextSentencePred实例的前向推断返回每个BERT输入序列的二分类预测
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
print(nsp_Y_hat.shape)

# 还可以计算两个二元分类的交叉熵损失。
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
print(nsp_l.shape)

# 整合所有BERT代码
#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        # 为什么只用<cls>而不用整个句子?
        #   [CLS]标记被设计为在输入序列的开始位置，它通过Transformer编码器的多层注意力机制汇聚了整个输入序列的信息。因此，[CLS]的隐藏状态可以视为对整个输入序列的浓缩表示，包含了整个句子对的综合特征。
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat