# 重新考察CNN
#   编码器：将输入编程成中间表达形式(特征)   --- 隐藏层特征提取
#   解码器：将中间表达形式解码成输出         --- softmax回归

# 重新考察RNN
#   编码器: 将文本表示成向量
#   解码器: 将向量表示成输出

# 编码器-解码器架构
#   一个模型被分为两块
#       编码器处理输入
#       解码器生成输出
#   Input → Encoder → State → Decoder → Output
#                                ↑
#                              Input

# 总结
#   使用编码器-解码器架构的模型, 编码器负责表示输出, 解码器负责输出

# 实现
from torch import nn

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)