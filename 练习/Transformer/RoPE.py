import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, max_len, num_hiddens):
        super(RoPE, self).__init__()
        self.max_len = max_len
        self.num_hiddens = num_hiddens
        self.P = self.getP()

    def getP(self):
        P = torch.zeros(self.max_len, self.num_hiddens)
        pos = torch.arange(0, self.max_len, dtype=torch.float32).reshape(-1, 1)
        div = torch.exp(torch.arange(0, self.num_hiddens, 2, dtype=torch.float32) / self.num_hiddens * -1.0 * torch.log(torch.tensor(10000.0, dtype=torch.float32)))
        P[:, 0::2] = torch.sin(pos * div)
        P[:, 1::2] = torch.cos(pos * div)
        return P.unsqueeze(0)

    def forward(self, inputs):
        # 获取cos和sin矩阵，重复并对偶数索引和奇数索引的维度进行扩展
        sin = self.P[:inputs.size(1), 0::2].expand(inputs.size(0), -1, -1)
        cos = self.P[:inputs.size(1), 1::2].expand(inputs.size(0), -1, -1)

        # 应用RoPE到输入张量x
        x_out = inputs * cos + torch.cat((-inputs[..., 1::2], inputs[..., 0::2]), dim=-1) * sin
        return x_out

# 测试代码
max_len = 100
num_hiddens = 64
batch_size = 2
seq_len = 50

rope = RoPE(max_len, num_hiddens)
inputs = torch.randn(batch_size, seq_len, num_hiddens)

outputs = rope(inputs)

print("输入形状:", inputs.shape)
print("输出形状:", outputs.shape)