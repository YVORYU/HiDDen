import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    译码器模块。接收带有水印的图像并提取水印。输入图像可能有各种各样的噪声应用于它，如裁剪、jpeg压缩等。
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels
        #定义第一层卷积层输入为3通道，输出为decoder_channels通道数
        layers = [ConvBNRelu(3, self.channels)]
        #定义后续卷积层，输入为decoder_channels通道数，输出为decoder_channels通道数
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        #定义最后一层卷积层，输入为decoder_channels通道数，输出为message_length通道数
        layers.append(ConvBNRelu(self.channels, config.message_length))
        #自适应平均池化层，将特征图的大小池化为1x1，输出为message_length通道数的张量
        """
        前面卷积层输出 [batch_size, message_length, H, W]
        自适应平均池化层输出 [batch_size, message_length, 1, 1]
        线性层将 [batch_size, message_length, 1, 1] 转换为 [batch_size, message_length]
        """
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        #将所有卷积层打包成为一个序列模型，它会按照传入的顺序依次执行每一层。
        self.layers = nn.Sequential(*layers)
        #定义线性层，输入为message_length通道数，输出为message_length通道数
        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, image_with_wm):
        #提取message，输出为[batch_size, message_length, 1, 1]
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        #将[batch_size, message_length, 1, 1]转换为[batch_size, message_length]
        x.squeeze_(3).squeeze_(2)
        #全连接层，输入为message_length通道数，输出为message_length通道数，
        # 由于经过layers层提取，message可能会出现“串扰”或“偏差”,经过线性层可以对message进行修正
        x = self.linear(x)
        return x
