# 权重衰退 / L2正则化, 解决过拟合
# 使用均方范数作为硬性限制
#   通过限制参数值的选择范围来控制模型容量
#       在min l(w,b)时, 使 ||w||² <= θ, 即w中的每一个元素值都要小于sqrt(θ)
#       通常不限制偏移b(限制不限制都差不多)
#       小的θ意味着更强的正则项
# 使用均方范数作为柔性限制： 防止过拟合, 相当于加入的惩罚值用来对最优解进行拉扯, 让模型拟合的结果偏离最优解, 向惩罚项偏移
#   (其实就是新的损失函数由两项组成，此时求导后，梯度有两项了，一项将w向绿线中心l拉，一项将w向原点λ拉，最后将在w*点达到一个平衡)
#   对于每个θ, 都可以找到λ使得之前的目标函数等价于下面
#       min l(w,b)+λ/2 ||w||²       可以通过拉格朗日乘子证明
#   超参数λ控制了正则项的重要程度
#       λ = 0: 无作用
#       λ -> 无穷大, w* -> 0
#       一般取e的-2, -3, -4
# 参数更新法则
# 计算梯度
#   loss = l(w,b)+λ/2 ||w||²
#   ∂loss / ∂w = ∂(l(w,b) + λ/2 ||w||²) / ∂w = ∂l(w,b)/∂w + λw
#   时间t更新参数: w_t+1 = (1-ηλ)w_t - η*(∂l(w_t,b_t) / ∂w_t)   (将上述表达式带入参数更新公式, w_t+1 = w_t - η∂loss/∂w)
#   通常ηλ < 1, 在深度学习中通常叫做权重衰退, 相当于在更新前, 先对当前的权重进行了缩小
# 总结
#   权重衰退通过L2正则项使得模型参数不会过大, 从而控制模型复杂度
#   正则项权重λ是控制模型复杂度的超参数
#   因为实际的数据集有噪音, 模型不可能拟合到真正的最优解, 因此要加入正则项

# 高维线性回归演示权重衰退
import torch
from torch import nn
from d2l import torch as d2l

# 从零实现
if __name__ == "__main__":
    # 人工生成数据, y = 0.05 + \sum_{i=1}^{d} 0.01x_i + ε  ε符合均值为0, 方差维0.01²的正态分布, 这里取输入x的维数为200
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5           # 训练集大小20, 测试集大小100, 输入维度200, 数据集划分大小5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05           # 真实的权重为0.01, 偏差为0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)            # 根据wx + b生成n_train个数据
    train_iter = d2l.load_array(train_data, batch_size)                 # 将数据按batch_size划分并转为迭代器
    test_data = d2l.synthetic_data(true_w, true_b, n_test)              # 同上
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)   # 同上

    # 从零实现
    # 初始化模型参数
    def init_params():
        w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)    # 通过正态分布生成
        b = torch.zeros(1, requires_grad=True)
        return [w, b]

    #L2惩罚: w²/2
    def l2_penalty(w):
        return torch.sum(w.pow(2)) / 2

    # 训练
    def train(lambd):
        w, b = init_params()
        net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
        num_epochs, lr = 100, 0.003
        animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
        for epoch in range(num_epochs):
            for X, y in train_iter:
                # 增加了L2范数惩罚项，
                # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
                l = loss(net(X), y)  + lambd * l2_penalty(w)
                l.sum().backward()
                d2l.sgd([w, b], lr, batch_size)     # 在这个sgd函数里面清掉了梯度
            if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                         d2l.evaluate_loss(net, test_iter, loss)))
        print('w的L2范数是：', torch.norm(w).item())
        d2l.plt.show()

    # 忽略正则化直接训练
    # train(lambd=0)

    # 使用权重衰退
    train(lambd=3)
    # def train_concise(wd):
    #     net = nn.Sequential(nn.Linear(num_inputs, 1))
    #     for param in net.parameters():
    #         param.data.normal_()
    #     loss = nn.MSELoss()
    #     num_epochs, lr = 100, 0.003
    #     # 偏置参数没有衰减
    #     trainer = torch.optim.SGD([
    #         {"params": net[0].weight, 'weight_decay': wd},
    #         {"params": net[0].bias}], lr=lr)
    #     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    #     for epoch in range(num_epochs):
    #         for X, y in train_iter:
    #             trainer.zero_grad()
    #             l = loss(net(X), y)
    #             l.mean().backward()
    #             trainer.step()
    #         if (epoch + 1) % 5 == 0:
    #             animator.add(epoch + 1,
    #                          (d2l.evaluate_loss(net, train_iter, loss),
    #                           d2l.evaluate_loss(net, test_iter, loss)))
    #     print('w的L2范数：', net[0].weight.norm().item())
    #     d2l.plt.show()
    #
    # # 不使用L2正则化
    # train_concise(0)
    #
    # # 使用L2正则化
    # train_concise(3)