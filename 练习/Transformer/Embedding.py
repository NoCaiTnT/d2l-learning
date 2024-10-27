import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, vocab_size, num_hiddens, max_len, device):
        super(PositionEmbedding, self).__init__()
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.P = self.getPosition()

    def getPosition(self):
        P = torch.zeros(self.max_len, self.num_hiddens)

        position = torch.arange(0, self.max_len, dtype=torch.float32).reshape(-1, 1)
        div = torch.exp(torch.arange(0, self.num_hiddens, 2, dtype=torch.float32) / self.num_hiddens * -torch.log(torch.tensor(10000.0, dtype=torch.float32)))

        P[:, 0::2] = torch.sin(position * div)
        P[:, 1::2] = torch.cos(position * div)

        P = P.unsqueeze(0)
        return P

    def forward(self, X):
        X = X.to(device)
        X = self.embedding(X)
        return X + self.P.expand(X.size(0), -1, -1).to(self.device)

if __name__ == "__main__":
    batch_size = 64
    vocab_size = 100
    num_hiddens = 128
    squence_max_len = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = PositionEmbedding(vocab_size, num_hiddens, squence_max_len, device)
    p.to(device)

    inputs = torch.randint(vocab_size, size=(batch_size, squence_max_len))
    outpus = p(inputs)

    print(outpus.shape)