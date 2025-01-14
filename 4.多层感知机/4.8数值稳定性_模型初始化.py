# 神将网络的梯度
#   考虑如下有d层的神经网络: h^t = f_t(h^(t-1)) 并且 优化的目标函数 y = l·f_d·...·f_1(x)
#   计算损失l关于参数W^t的梯度为:
#        ∂l      ∂l     ∂h^d      ∂h^(t+1)   ∂h^t
#       ---- = ------ --------...---------- ------
#       ∂W^t    ∂h^d   ∂h^d-1       ∂h^t     ∂W^t
#   这里的h都是向量, 向量对向量求导得到矩阵, 因此对个矩阵相乘会出现问题(太多次矩阵乘法)
# 数值稳定性的常见两个问题: 梯度爆炸, 梯度消失
#   梯度爆炸: 1.5^100 ≈ 4 × 10^17
#   梯度消失: 0.8^100 ≈ 2 × 10^-10
# 梯度爆炸的问题:
#   值超出值域(infinity)
#       对于16位浮点数尤为严重(数值区间6e-5 - 6e4)
#   对学习率敏感
#       如果学习率太大 -> 大参数值 -> 更大的梯度
#       如果学习率太小 -> 训练无进展
#       我们可能需要在训练过程中不断调整学习率
# 梯度消失的问题
#   梯度值变成0
#       对16位浮点数尤为严重
#   训练没有进展
#       不管如何选择学习率
#   对于底层尤为严重(底层为输入层附近的层, 顶层为输出层附近的层)
#       仅仅顶层训练的较好
#       无法让神经网络更深(底层不更新, 相当于没有用到)
# 总结
#   当数值过大或者过小时会导致数值问题(梯度不能太大或太小)
#   常发生在深度模型中, 因为其会对n个数进行累乘
# 让训练更加稳定
#   目标: 让梯度值在合理的范围内
#       例如[1e-6, 1e3]
#   方法:
#       1.将乘法变为加法: ResNet, LSTM
#       2.归一化: 梯度归一化, 梯度裁剪
#       3.合理的权重初始和激活函数
# 让每层的方差是一个常熟
#   将每层的输出和梯度都看做随机变量
#   让他们的均值和方差都保持一致(例如正向传播过程中, 每一层输出的元素的均值为0, 方差为a, 反向传播中, 损失对每一层元素的梯度的均值为0, 方差为b)
# 权重初始化
#   在合理值区间里随机初始参数
#   训练开始的时候更容易有数值不稳定
#       远离最优解的地方损失函数表面可能很复杂
#       最优解附近表面会比较平
#   使用均值为0, 方差为0.01的正态分布来初始可能对小网络没问题, 但不能保证深度神经网络
# Xavier初始
#   难以同时满足n_(t-1)γ_t = 1 和 n_(t)γ_t = 1     其中n_(t-1)为第t层的输入维度, n_t是第t层的输出维度, 这两个一般不会相等, γ_t为第t层的权重的方差、
#   Xavier使得其进行折中, 即γ_t(n_(t-1) + n_(t))/2 = 1  -> γ_t = 2 / (n_(t-1) + n_(t))
#       正态分布为均值为0, 方差为sqrt(γ_t) = sqrt(2 / (n_(t-1) + n_(t)))
#       均值分布为 (-sqrt(6 / (n_(t-1) + n_(t))), sqrt(6 / (n_(t-1) + n_(t))))
#   适配权重形状变换, 特别是n_t
# 假设线性的激活函数
#   假设σ(x) = αx + β
#   要保持激活函数不改变输入和输出的方差和均值, α只能取1, β只能取0(正向反向都一样)
# 检查常用激活函数
#   使用泰勒展开
#       sigmoid(x) = 0.5 + x/4 - x^3/48 + O(x^5)    不满足α=0, β=1
#       tanh(x) = 0 + x - x^3/3 + O(x^5)        输入在0附近可以看做是α=0, β=1
#       relu(x) = 0 + x  for x>= 0              输入>=0可以看做是α=0, β=1
#   调整sigmoid使其在0附近满足α=0, β=1
#       4 × sigmoid - 2
# 总结
#   合理的权重初始值和激活函数的选取可以提升数值稳定性: 使每一层的输出和梯度都是一个均值为0, 方差为一个固定数的随机变量
#       为了满足上述目标: 权重使用Xavier初始化, 激活函数使用tanh, relu或调整后的sigmoid
