import torch.nn as nn

class ConvBNRelu(nn.Module):
    """
    CNN基础模块构建
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    用于隐藏网络的构建块。是一个序列的卷积，批归一化，和ReLU激活
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        # 定义卷积层:批归一化层,ReLU激活层
        # 卷积层:3x3,步长为stride,填充为1,卷积核大小为3x3
        # 批归一化层:输出通道数为channels_out
        # ReLU激活层:inplace新张量
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
