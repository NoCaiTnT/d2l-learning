import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, ffn_num_hiddens, num_hiddens):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.linear2 = nn.Linear(ffn_num_hiddens, num_hiddens)

    def forward(self, X):
        return self.linear2(torch.relu(self.linear1(X)))

class AddNorm(nn.Module):
    def __init__(self, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))