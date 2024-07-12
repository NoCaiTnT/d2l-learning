# 回顾: 循环神经网络
#   更新隐藏状态: h_t = φ(W_hh * h_{t-1} + W_hx * x_t-1 + b_h)
#   输出的方式: o_t = W_hq * h_t + b_q

# 更深
#   浅RNN
#       输入
#       隐层
#       输出
#   深RNN
#       输入
#       隐层1
#       隐层2
#       ...
#       隐层n
#       输出
#   H_t^1 = f_1(H_{t-1}^1, X_t)
#   ...
#   H_t^j = f_2(H_{t-1}^2, H_t^{j-1})
#   O_t = g(H_t^L)

# 总结
#   深度循环神经网络使用多个隐藏层来获得更多的非线性性

import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 通过num_layers指定隐藏层的层数
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

# 训练
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)