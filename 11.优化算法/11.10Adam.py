# Adam
#   非常平滑, 导致对 学习率 不敏感
#   记录v_t = β_1 * v_{t-1} + (1 - β_1) * g_t, 通常β_1 = 0.9
#   展开v_t = (1 - β_1) * (g_t + β_1 * g_{t-1} + β_1^2 * g_{t-2} + ...)
#   因为 Σ_{i=0}^∞ β_1^i = 1 / (1 - β_1), 所以权重和为1
#   在前几个t会进行修正
#       由于v_0 = 0,, 且 Σ_{i=0}^t β_1^i = 1-β_1^t / (1-β_1), 修正hat{v}_t = v_t / (1 - β_1^t)
#   类似记录s_t = β_2 * s_{t-1} + (1 - β_2) * g_t^2, 通常β_2 = 0.999, 且修正hat{s}_t = s_t / (1 - β_2^t)
#   计算重新调整后的梯度g'_t = hat{v}_t / sqrt(hat{s}_t) + ε
#   最后更新w_t = w_{t-1} - η * g'_t

import torch
from d2l import torch as d2l

def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim)
d2l.plt.show()

trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
d2l.plt.show()