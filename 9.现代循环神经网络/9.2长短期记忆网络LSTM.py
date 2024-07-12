# 长短期记忆网络
#   忘记门: 将值朝0方向缩小
#   输入门: 决定是不是忽略掉输入数据
#   输出门: 决定是不是使用隐状态
#   注意: 和GRU思想差不多: 要不要不看现在的输入 尽量去用过去的信息; 要不要不看过去的信息 尽量去用现在的信息

# 门
#   忘记门: F_t = σ(X_t W_xf + H_{t-1} W_hf + b_f)
#   输入门: I_t = σ(X_t W_xi + H_{t-1} W_hi + b_i)
#   输出门: O_t = σ(X_t W_xo + H_{t-1} W_ho + b_o)

# 候选记忆单元 (和GRU的候选隐状态类似)
#   C~_t = tanh(X_t W_xc + H_{t-1} W_hc + b_c)

# 记忆单元 (和GRU的隐状态类似)
#   C_t = F_t ⊙ C_{t-1} + I_t ⊙ C~_t
#   由于C~_t的范围是[-1, 1], 而F_t和I_t的范围是[0, 1], 所以C_t的范围是[-2, 2]
#   因此, 为了使C_t的范围在[-1, 1], 我们需要对C~_t进行缩放, 使其范围在[-1, 1], 即下面的隐状态

# 隐状态
#   H_t = O_t ⊙ tanh(C_t)
#   通过tanh函数将记忆单元的范围缩放到[-1, 1], 然后通过输出门决定输出多少
#   当O_t全为0, 则相当于抛弃了所有记忆, 重置了状态, 当全为1, 则相当于没有输出

# 总结
#   F_t = σ(X_t W_xf + H_{t-1} W_hf + b_f)
#   I_t = σ(X_t W_xi + H_{t-1} W_hi + b_i)
#   O_t = σ(X_t W_xo + H_{t-1} W_ho + b_o)
#   C~_t = tanh(X_t W_xc + H_{t-1} W_hc + b_c)
#   C_t = F_t ⊙ C_{t-1} + I_t ⊙ C~_t
#   H_t = O_t ⊙ tanh(C_t)

import torch
from torch import nn
from d2l import torch as d2l

# 加载数据
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 忘记门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆单元参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 综合所有参数
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    # 开启梯度
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化函数
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),   # 隐藏状态
            torch.zeros((batch_size, num_hiddens), device=device))   # 记忆单元

# 定义模型
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i)
        F = torch.sigmoid(torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f)
        O = torch.sigmoid(torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# 训练模型
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 简洁实现
num_inputs = vocab_size
lstm_layer = nn.LSTM(len(vocab), num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
