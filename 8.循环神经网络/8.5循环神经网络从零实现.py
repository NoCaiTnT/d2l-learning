
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 独热编码
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

# 小批量数据形状是(批量大小, 时间步数)
X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape)         # 为什么要转置? 把时间步放到第一维度, 在访问不同时间时就会变成连续的

# 初始化循环神经网络模型的模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size   # 输入的是词表中的一个词, 输出也是词表中的一个词, 所以输入输出大小相同

    def normal(shape):      # 随机生成一个形状为shape的张量
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))    # 输入到隐藏层
    W_hh = normal((num_hiddens, num_hiddens))   # 隐藏层到隐藏层
    b_h = torch.zeros(num_hiddens, device=device)   # 隐藏层的偏置
    W_hq = normal((num_hiddens, num_outputs))   # 隐藏层到输出层
    b_q = torch.zeros(num_outputs, device=device)   # 输出层的偏置
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)   # 开启梯度
    return params

# 定义一个初始化隐藏状态的函数
# 因为在第一个时间步, 没有隐藏状态, 所以需要初始化一个隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 下面的rnn函数定义了如何在一个时间步内计算隐藏状态和输出
def rnn(inputs, state, params):
    # `inputs`的形状：(`时间步数量`, `批量大小`, `词表大小`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state                                  # H为上一个时间步的隐藏状态
    outputs = []
    # `X`的形状：(`批量大小`, `词表大小`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)      # 输出变为2维张量, 第一维是时间步×批量大小(从上往下拼接), 第二维是词表大小(即预测值)

# 创建一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数
class RNNModelScratch:  #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)     # t*批量大小*词表大小
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 样例, 检查输出是否具有正确的形状
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)

# 首先定义预测函数来生成用户提供的prefix(前缀)之后的新字符
# num_preds: 预测的字符数
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在`prefix`后面生成新字符。"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]                # 将字符串转换为词元索引
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))  # 当前预测的出来的词, 是下一次预测的输入
    for y in prefix[1:]:  # 预热期, 用前缀生成隐藏状态, 有真实值, 不需要预测
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测`num_preds`步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))     # 通过argmax获得具有最高概率的词元索引, 并附加到输出列表
    return ''.join([vocab.idx_to_token[i] for i in outputs])        # 将下标转换为真实字符

# 测试一下
print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

# 梯度裁剪 g ← min(1, θ/|g|) g
def grad_clipping(net, theta):  #@save
    """裁剪梯度。"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# 定义一个函数在一个迭代周期内训练模型
# use_random_iter: 是否使用随机采样, 即下一个批量的第i个样本和当前批量的第i个样本是否连续
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):  #@save
    """训练模型一个迭代周期（定义见第8章）。"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和, 词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:        # 若是第一个时间步, 则需要初始化隐藏状态; 若用的是随机采样, 则也需要初始化隐藏状态, 因此上一个批量和当前批量没有关系, 所以需要重新初始化隐藏状态
            # 在使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 所选代码片段是神经网络模型训练循环的一部分，特别是在反向传播过程中处理模型的状态。在递归神经网络（RNN）中，状态非常重要，因为它承载着从序列中的一个步骤到下一个步骤的信息。 代码首先会检查模型网是否是 nn.Module 的实例（nn.Module 是 PyTorch 中所有神经网络模块的基类），以及状态是否不是元组：
            # 如果该条件为真，则意味着模型是 PyTorch 模型，状态是张量（如注释中所述，nn.GRU 就是这种情况）。在这种情况下，代码会调用状态的 detach_() 方法：
            # detach_() 方法会将张量从计算图中分离出来。这意味着该状态没有梯度，因此在反向传播过程中不会更新。当我们想在下一次迭代中保留状态，但又不想让它影响梯度时，这个方法就非常有用。 如果条件不为真，则表示状态是一个元组。nn.LSTM 或自定义模型就是这种情况。在这种情况下，代码会遍历状态元组中的每个张量并将其分离：
            # 这样做同样是为了防止在反向传播过程中更新状态。总之，这段代码负责在训练过程中处理模型的状态，确保状态被带到下一次迭代，但不会影响梯度。
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU`是个张量
                state.detach_()     # 反向传播的时候, 不改掉`state`的值, 但是会将前面的计算图分离, 使得梯度不会传到前面去(只更新当前值产生的梯度, 不更新前面的值产生的梯度)
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)    # Y的形状是(时间步数, 批量大小), 转置后变为(批量大小*时间步数)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)  # 因为已经调用了mean函数
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()    # 困惑度, 词元数量/时间

# 循环神经网络模型的训练函数即支持从零开始实现的模型，也支持使用高级API实现的模型
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):  #@save
    """训练模型（定义见第8章）。"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.Adam(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# 开始训练
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), False)  # 使用顺序采样
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), True)   # 使用随机采样