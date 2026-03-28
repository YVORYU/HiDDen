import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization

"""
实现噪声层组合器，根据配置参数随机选择噪声层
"""
class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        #初始化噪声层列表，默认包含Identity层(无噪声)
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            if type(layer) is str:
                #根据字符串配置创建噪声层
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        #随机选择一个噪声层
        #并将其应用到编码图像和封面图像上
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        #返回应用噪声层后的编码图像和封面图像
        return random_noise_layer(encoded_and_cover)

