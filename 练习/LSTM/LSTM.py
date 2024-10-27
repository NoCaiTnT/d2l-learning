import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, device):
        super(MyLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device

        self.w_xf, self.w_hf, self.b_f = self.getParameter(vocab_size, num_hiddens)
        self.w_xi, self.w_hi, self.b_i = self.getParameter(vocab_size, num_hiddens)
        self.w_xo, self.w_ho, self.b_o = self.getParameter(vocab_size, num_hiddens)
        self.w_xc, self.w_hc, self.b_c = self.getParameter(vocab_size, num_hiddens)
        self.w_hy = nn.Parameter(torch.randn(size=(num_hiddens, vocab_size)) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(size=(vocab_size,)))

    def getParameter(self, in_channel, out_channel):
        w_x_ = nn.Parameter(torch.randn(size=(in_channel, out_channel)) * 0.01)
        w_h_ = nn.Parameter(torch.randn(size=(out_channel, out_channel)) * 0.01)
        b_ = nn.Parameter(torch.zeros(size=(out_channel,)))
        return w_x_, w_h_, b_

    def getInitState(self, batch_size):
        return torch.zeros(size=(batch_size, self.num_hiddens)), torch.zeros(size=(batch_size, self.num_hiddens))

    def forward(self, X, state, candidate):
        H = state
        C = candidate
        outputs = []
        step_size = X.size(1)

        for t in range(step_size):
            X_t = X[:, t, :].to(self.device)
            F_t = torch.sigmoid(torch.mm(X_t, self.w_xf) + torch.mm(H, self.w_hf) + self.b_f)
            I_t = torch.sigmoid(torch.mm(X_t, self.w_xi) + torch.mm(H, self.w_hi) + self.b_i)
            O_t = torch.sigmoid(torch.mm(X_t, self.w_xo) + torch.mm(H, self.w_ho) + self.b_o)
            C_hat_t = torch.tanh(torch.mm(X_t, self.w_xc) + torch.mm(H, self.w_hc) + self.b_c)
            C = F_t * C + I_t * C_hat_t
            H = O_t * torch.tanh(C)
            Y_t = torch.mm(H, self.w_hy) + self.b_y
            outputs.append(Y_t)

        return torch.cat(outputs, dim=0), H, C

def predict(inputs, net, device, batch_size):
    net = net.to(device)
    inputs = inputs.to(device)
    state, candidate = net.getInitState(batch_size)
    state = state.to(device)
    candidate = candidate.to(device)

    outputs, H, C = net(inputs, state, candidate)
    print(outputs.shape, H.shape, C.shape)
    return outputs, H, C


vocab_size = 100
num_hiddens = 128
batch_size = 64
time_step = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.randn(size=(batch_size, time_step, vocab_size))

net = MyLSTM(vocab_size, num_hiddens, device)

_, _, _ = predict(inputs, net, device, batch_size)
