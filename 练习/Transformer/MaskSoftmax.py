import torch
import torch.nn as nn

class MaskSoftmax(nn.Module):
    def __init__(self):
        super(MaskSoftmax, self).__init__()

    def forward(self, X, valid_len = None):
        if valid_len is None:
            return torch.softmax(X, dim=-1)
        else:
            mask = torch.zeros(size=(X.size(0), X.size(1), X.size(2)), dtype=torch.float32)
            for t in range(X.size(0)):
                mask[t, :, valid_len[t]:] = -1e6
            X += mask
            return torch.softmax(X, dim=-1)


if __name__ == "__main__":
    mask_softmax = MaskSoftmax()
    inputs = torch.randint(0, 5, size=(2, 5, 10), dtype=torch.float32)
    valid_len = torch.tensor([2, 3])
    outputs = mask_softmax(inputs, valid_len)
    print(outputs)