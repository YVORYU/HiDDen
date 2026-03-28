import numpy as np
import os
import re
import csv
import time
import pickle
import logging

import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden

# 图片转张量
def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor

# 张量转图片
def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)

"""
保存验证的图片
# original_images: 原始图片张量，形状为(batch_size, channels, height, width)
# watermarked_images: 隐藏写入后的图片张量，形状为(batch_size, channels, height, width)
# epoch: 当前训练轮次
# folder: 保存图片的文件夹路径
# resize_to: 指定将图片缩放为指定大小
"""
def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    # 提取第一个batch的图片
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    # 将值从原始范围[-1,1]缩放到[0,1]范围，值域归一化
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    #如果传入的参数不为None，将图片缩放为指定大小
    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)
    #拼接原始图片和隐藏写入后的图片，维度为0，即按行拼接
    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    # 保存拼接后的图片
    torchvision.utils.save_image(stacked_images, filename, nrow=original_images.shape[0], normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file

"""
保存模型检查点
# model: 要保存的模型对象
# experiment_name: 实验名称
# epoch: 当前训练轮次
# checkpoint_folder: 保存检查点的文件夹路径
"""
def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


"""
加载最新的模型检查点
# checkpoint_folder: 检查点文件夹路径
# return: 最新的检查点对象和检查点文件路径
"""
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(options_file_name) -> (TrainingOptions, HiDDenConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)