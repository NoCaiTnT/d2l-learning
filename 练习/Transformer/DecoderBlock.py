import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from AddNorm_FFN import AddNorm, FFN

class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, ffn_num_hiddens):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadAttention(num_heads, num_hiddens, dropout, num_hiddens, num_hiddens, num_hiddens)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = MultiHeadAttention(num_heads, num_hiddens, dropout, num_hiddens, num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = FFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, decoder, decoder_valid_len, encoder_outputs, encoder_valid_len):
        y = self.attention1(decoder, decoder, decoder, decoder_valid_len)
        x = self.addnorm1(decoder, y)

        y = self.attention2(x, encoder_outputs, encoder_outputs, encoder_valid_len)
        x = self.addnorm2(x, y)

        y = self.ffn(x)
        x = self.addnorm3(x, y)

        return x
