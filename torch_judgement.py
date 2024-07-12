# 判断是否安装了cuda
import torch

print(torch.cuda.is_available())  # 返回True则说明已经安装了cuda
# 判断是否安装了cuDNN
from torch.backends import cudnn

print(cudnn.is_available())  # 返回True则说明已经安装了cuDNN
# 判断cuda版本
print(torch.__version__)
print(torch.version.cuda)
