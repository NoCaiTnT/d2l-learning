# 回归和分类的区别:
# 回归是估计一个连续值, 分类是预测一个/多个离散类别
# 对类别进行一位有效编码
# 最大值预测为置信度最大值
# softmax将其归一化, 全大于0
# 使用真实概率和预测概率计算损失
# 交叉熵常用来衡量两个概率的区别
# 常用损失函数:
#   1. L2 Loss(平方误差): l = 0.5 * (y - y')²
#   2. L1 Loss(绝对值误差): l = |y - y'|   (0点不可导, 变化剧烈)
#   3. Huber's Robust Loss: l = |y - y'| - 0.5     if |y - y'| > 1
#                             = 0.5 * (y - y')²    otherwise
