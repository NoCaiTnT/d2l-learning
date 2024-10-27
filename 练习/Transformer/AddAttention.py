import torch
import torch.nn as nn
from MaskSoftmax import MaskSoftmax

class AddAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, drop_out):
        super(AddAttention, self).__init__()
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(drop_out)
        self.mask_soft = MaskSoftmax()

    def forward(self, query, key, value, valid_len):
        q = self.w_q(query)
        k = self.w_k(key)

        t = q.reshape(q.size(0), q.size(1), 1, -1) + k.unsqueeze(1)
        t = torch.tanh(t)

        scores = self.w_v(t).squeeze(-1)
        scores = self.mask_soft(scores, valid_len)
        scores = self.dropout(scores)
        target = torch.bmm(scores, value)
        return target


if __name__ == "__main__":
    batch_size = 2
    num_hiddens = 12
    max_len = 5
    query_size = 10
    kv_pairs = 7
    key_size = 8
    value_size = 9
    query = torch.randn(size=(batch_size, max_len, query_size))
    valid_len = torch.tensor([2, 3])
    key = torch.randn(size=(batch_size, kv_pairs, key_size))
    value = torch.randn(size=(batch_size, kv_pairs, value_size))
    add_attention = AddAttention(key_size, query_size, num_hiddens, 0.5)
    outputs = add_attention(query, key, value, valid_len)
    print(outputs)