# 双向循环神经网络

# 未来很重要
#   I am __(happy)
#   I am __(not) very hungry
#   I am __(very) very happy, I could eat half a pig
#   取决于过去和未来的上下文, 可以填很不一样的词
#   目前为止RNN只看过去
#   在填空的时候, 我们也可以看未来

# 双向RNN
#   一个前向RNN隐层
#   一个后向RNN隐层
#   合并两个隐状态得到输出
#   rH: right_H前向隐状态
#   lH: left_H后向隐状态
#   rH_t = φ(X_t W_xh^(f) + rH_{t-1} W_hh^(f) + b_h^(f))
#   lH_t = φ(X_t W_xh^(b) + lH_{t+1} W_hh^(b) + b_h^(b))
#   H_t = [rH_t, lH_t]
#   O_t = W_hq H_t + b_q

# 不适用于推理
#   因为看不到后面的信息
# 可以对句子做特征提取

# 总结
#   双向循环神经网络通过反向更新的隐藏层来利用方向时间信息
#   通常用来对序列抽取特征, 填空, 而不是预测未来

# 实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)