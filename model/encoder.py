import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    用于隐藏网络的编码器。将水印插入到图像中。
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        #定义第一层卷积层输入为3通道，输出为conv_channels通道数
        layers = [ConvBNRelu(3, self.conv_channels)]
        #定义后续卷积层，输入为conv_channels通道数，输出为conv_channels通道数
        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)
        #将所有卷积层组合起来，打包为一个序列模型，它会按照传入的顺序依次执行每一层。
        self.conv_layers = nn.Sequential(*layers)
        #定义合并层，输入为conv_channels通道数+3通道数+message_length通道数，输出为conv_channels通道数
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        #定义最终卷积层，输入为conv_channels通道数，输出为3通道数，卷积核大小为1*1
        #3*3的卷积核用于特征提取，1*1的卷积核用于通道数升降维，特征融合
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        #由于图像是4维张量([batch_size,3, H, W]),所以要把message也变成4维
        expanded_message = message.unsqueeze(-1) #将原message张量扩展一个维度
        expanded_message.unsqueeze_(-1)#将扩展后的message张量原地扩展一个维度
        
        #将扩展后的message张量扩展为与图像相同的高度和宽度
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        #对图像进行卷积层编码，输出为conv_channels通道数的特征图
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        #将扩展后的message张量、编码后的图像特征图、原始图像张量按通道维度拼接起来，输出为conv_channels通道数的张量
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        #对拼接后的张量进行卷积层编码，输出为conv_channels通道数的特征图
        im_w = self.after_concat_layer(concat)
        #对特征图进行卷积层编码，输出为3通道数的图像
        im_w = self.final_layer(im_w)
        return im_w
