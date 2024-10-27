import torch
import torch.nn as nn
from MaskSoftmax import MaskSoftmax

class DotAttention(nn.Module):
    def __init__(self, dropout):
        super(DotAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.mask_softmax = MaskSoftmax()

    def forward(self, query, key, value, valid_len):
        key = key.permute(0, 2, 1)
        score = torch.bmm(query, key) / torch.sqrt(torch.tensor(query.size(2)))
        score = self.mask_softmax(score, valid_len)
        target = torch.bmm(self.dropout(score), value)
        return target

if __name__ == "__main__":
    batch_size = 2
    max_size = 5
    qk_dim = 10
    kv_pairs = 7
    value_dim = 8
    query = torch.randn(size=(batch_size, max_size, qk_dim))
    key = torch.randn(size=(batch_size, kv_pairs, qk_dim))
    value = torch.randn(size=(batch_size, kv_pairs, value_dim))
    valid_len = torch.tensor([2, 3])

    dot_attention = DotAttention(0.3)
    outputs = dot_attention(query, key, value, valid_len)
    print(outputs)