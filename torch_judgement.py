# 判断是否安装了cuda
import torch
from torch import nn

print(torch.cuda.is_available())  # 返回True则说明已经安装了cuda
# 判断是否安装了cuDNN
from torch.backends import cudnn

print(cudnn.is_available())  # 返回True则说明已经安装了cuDNN
# 判断cuda版本
print(torch.__version__)
print(torch.version.cuda)

# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.ones(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
print(output.shape)  # torch.Size([20, 5, 10, 10]