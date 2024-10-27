import torch
import torch.nn as nn

class MyGRU(nn.Module):
    def __init__(self, vocab_size, num_hiddens, device):
        super(MyGRU, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device

        self.w_xr = nn.Parameter(torch.randn(size=(vocab_size, num_hiddens)) * 0.01)
        self.w_hr = nn.Parameter(torch.randn(size=(num_hiddens, num_hiddens)) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(size=(num_hiddens,)))

        self.w_xz = nn.Parameter(torch.randn(size=(vocab_size, num_hiddens)) * 0.01)
        self.w_hz = nn.Parameter(torch.randn(size=(num_hiddens, num_hiddens)) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(size=(num_hiddens,)))

        self.w_xh = nn.Parameter(torch.randn(size=(vocab_size, num_hiddens)) * 0.01)
        self.w_hh = nn.Parameter(torch.randn(size=(num_hiddens, num_hiddens)) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(size=(num_hiddens,)))

        self.w_ho = nn.Parameter(torch.randn(size=(num_hiddens, vocab_size)) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(size=(vocab_size,)))

    def getInitState(self, batch_size):
        return torch.zeros(size=(batch_size, self.num_hiddens))

    def forward(self, X, state):
        H = state
        setp_size = X.size(1)
        outputs = []

        for t in range(setp_size):
            x_t = X[:, t, :]
            r_t = torch.sigmoid(torch.mm(x_t, self.w_xr) + torch.mm(H, self.w_hr) + self.b_r)
            z_t = torch.sigmoid(torch.mm(x_t, self.w_xz) + torch.mm(H, self.w_hz) + self.b_z)
            h_bar = torch.tanh(torch.mm(x_t, self.w_xh) + torch.mm(r_t * H, self.w_hh) + self.b_h)
            H = H * z_t + h_bar * (1 - z_t)
            o_t = torch.mm(H, self.w_ho) + self.b_o
            outputs.append(o_t)

        return torch.cat(outputs, dim=0), H

def predict(inputs, net, device):
    net = net.to(device)
    inputs = inputs.to(device)
    init_state = net.getInitState(inputs.size(0)).to(device)

    outputs, final_state = net(inputs, init_state)

    print(outputs.shape, final_state.shape)

    return outputs, final_state


vocab_size = 100
num_hiddens = 128
batch_size = 32
step_size = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = MyGRU(vocab_size, num_hiddens, device)
inputs = torch.randn(size=(batch_size, step_size, vocab_size))

_, _ = predict(inputs, net, device)

