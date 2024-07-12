# 关注一个序列
#   不是每个观察值都是同等重要的
#   想只记住相关的观察需要:
#       能关注的机制(更新门)
#       能遗忘的机制(重置门)

# 门
#   重置门: R_t = σ(X_t W_xr + H_{t-1} W_hr + b_r), 要不要将过去的信息遗忘
#   更新门: Z_t = σ(X_t W_xz + H_{t-1} W_hz + b_z), 要不要用当前的输入更新

# 候选隐状态
#   hat(H_t) = tanh(X_t W_xh + (R_t ⊙ H_{t-1}) W_hh + b_h)
#   R_t ⊙ H_{t-1} 是按元素乘法
#   若R_t全为0, 则相当于抛弃了所有过去的信息, 若全为1, 则相当于没有重置
#   相当于是保留了多少过去的信息H_{t-1}
#   R_t为可学习参数

# 隐状态
#   H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ hat(H_t)
#   若Z_t全为0, 则相当于抛弃了所有过去的信息, 若全为1, 则相当于没有更新
#   相当于是忽略了多少新的信息X_t

# 总结
#   R_t = σ(X_t W_xr + H_{t-1} W_hr + b_r)
#   Z_t = σ(X_t W_xz + H_{t-1} W_hz + b_z)
#   hat(H_t) = tanh(X_t W_xh + (R_t ⊙ H_{t-1}) W_hh + b_h)
#   H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ hat(H_t)

import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 综合所有参数
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    # 开启梯度
    for param in params:
        param.requires_grad_(True)

    return params

# 定义隐藏状态的初始化函数
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 定义GRU模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
        R = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.mm(X, W_xh) + torch.mm(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 训练
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 简洁实现
num_inputs = vocab_size
gru_layer = nn.GRU(len(vocab), num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)