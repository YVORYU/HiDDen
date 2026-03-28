import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    判别器，判断图像是否包含水印。
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()
        #定义判积层，输入为3通道图像，输出为discriminator_channels通道数
        layers = [ConvBNRelu(3, config.discriminator_channels)]
        #定义判积器的剩余层，每个层的输入和输出通道数相同
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))
        #定义判积器的输出层，输入为discriminator_channels通道数，输出为1通道数
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        #将所有卷积层打包成为一个序列模型，它会按照传入的顺序依次执行每一层。
        self.before_linear = nn.Sequential(*layers)
        #定义全连接层

        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        #将图像输入到判积器中，输出为[batch_size, discriminator_channels, 1, 1]
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        #将[batch_size, discriminator_channels, 1, 1]转换为[batch_size, discriminator_channels]
        X.squeeze_(3).squeeze_(2)
        #由于经过layers层提取，图像可能会出现“串扰”或“偏差”,经过全性层可以对图像进行修正
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X