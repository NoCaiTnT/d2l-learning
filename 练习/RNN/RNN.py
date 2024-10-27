import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super(MyRNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device

        # 定义模型参数
        self.W_xh = nn.Parameter(torch.randn(vocab_size, num_hiddens) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * 0.01)
        self.B_h = nn.Parameter(torch.randn(num_hiddens,) * 0.01)
        self.W_ho = nn.Parameter(torch.randn(num_hiddens, vocab_size) * 0.01)
        self.B_o = nn.Parameter(torch.randn(vocab_size) * 0.01)

    def getInitState(self, batch_size):
        return torch.zeros(size=(batch_size, self.num_hiddens))

    def forward(self, inputs, state):
        H = state
        outputs = []

        seq_len = inputs.size(1)  # 获取时间步
        for t in range(seq_len):
            X = inputs[:, t, :].float()
            H = torch.tanh(torch.mm(X, self.W_xh) + torch.mm(H, self.W_hh) + self.B_h)
            Y = torch.mm(H, self.W_ho) + self.B_o
            outputs.append(Y.unsqueeze(1))
        return torch.cat(outputs, dim=1), H


def train(device, net, inputs, targets, loss_func, optimizer, epochs, batch_size):
    net.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    state = net.getInitState(batch_size).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = net(inputs, state)
        loss = loss_func(outputs.view(-1, net.vocab_size), targets.view(-1))  # Reshape for CrossEntropyLoss
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch + 1}, loss {loss.item()}")
    torch.save(net.state_dict(), "rnn.pth")


def predict(device, net, inputs, batch_size):
    net.to(device)
    inputs = inputs.to(device)
    state = net.getInitState(batch_size).to(device)

    with torch.no_grad():  # 在推理时禁用梯度计算
        outputs, new_state = net(inputs, state)
    print(outputs.shape)
    return outputs, new_state


# 示例用法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 100  # 词汇表大小
num_hiddens = 128  # 隐藏层大小
batch_size = 2  # 批处理大小
seq_len = 10  # 序列长度

inputs = torch.randint(vocab_size, (batch_size, seq_len, vocab_size), device=device)
targets = torch.randint(vocab_size, (batch_size, seq_len), device=device)

net = MyRNN(vocab_size, num_hiddens)
net.load_state_dict(torch.load("rnn.pth"))

train(device, net, inputs, targets, nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters(), lr=0.01), 10, batch_size)

predict(device, net, inputs, batch_size)