# 感知机
# 给定输入x, 权重w, 偏移b, 感知机输出:
#   o = σ(<w,x>+b) σ(x) = 1 if x > 0
#                       = -1 otherwise
# 输出是一个离散的类, 只能2分类
#   与线性回归的区别: 线性回归输出的是一个实数
#   与softmax回归的区别: softmax回归输出的是类别的概率
# 训练: 分类错误时更新权重, w = w + yx, b = b + y
#   训练等价于使用批量大小维1的梯度下降, 损失函数为: l = max(0, -y<w,x>)
# 收敛定理:
#   数据在半径r内
#   有余量ρ能将所有样本分为两类, y(xw+b) >= ρ, ||w||²+b² <= 1
#   感知机保证在(r²+1) / ρ²步后收敛, r为样本的范围
# XOR问题: 一三象限为一类, 二四象限为一类
#   感知机不能拟合XOR函数, 只能产生线性分割面
# 总结(感知机):
#   感知机是一个二分类模型, 是最早的AI模型之一
#   它的求解算法等价于使用批量大小为1的梯度下降
#   它不能拟合XOR函数, 导致第一次AI寒冬
# 多层感知机
# 单隐藏层:
#   input: x1-x4, hidden layer: h1-h5, output layer: o1-o3
#   超参数: 隐藏层的大小(神经元个数)
# 单隐藏层 - 单分类
#   输入x, n维
#   若隐藏层有m个神经元, 隐藏层的权重w1为m×n维, 偏置b1为m维
#   输出单分类, 因此输出层权重w2为m维, 偏置b2为1维
#   h = σ(w1x + b1), o = w2h + b2
#   σ为激活函数, 非线性函数, 按元素计算
#   为什么非线性? 多个线性等价于一个线性
#   激活函数就是为了引入非线性
# Sigmoid激活函数: 将输入投影到(0,1)的开区间内
#   sigmoid(x) = 1 / (1 + exp(-x))
# Tanh激活函数: 将输入投影到(-1, 1)的开区间内
#   tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
# ReLU(rectified linear unit)激活函数:
#   ReLU(x) = max(x, 0)
# 单隐藏层 - 多分类, 相当于在softmax回归中加入了一层隐藏层
#   k分类: 对k个输出做一个softmax操作
#   输入x, n维
#   若隐藏层有m个神经元, 隐藏层的权重w1为m×n维, 偏置b1为m维
#   输出k分类, 因此输出层权重w2为m×k维, 偏置b2为k维
#   h = σ(w1x + b1), o = w2h + b2, y = softmax(o)
# 多隐藏层
#   h1 = σ(w1x + b1), h2 = σ(w2h1 + b2), h3 = σ(w3h2 + b3), o = w4h3 + b4
#   超参数: 隐藏层的层数, 每层隐藏层的大小
# 总结(多层感知机)
# 多层感知机使用隐藏层和激活函数来得到非线性模型
# 常用的激活函数是: Sigmoid, Tanh, ReLU
# 使用softmax来处理多类分类
# 超参数为隐藏层数, 和各个隐藏层大小
