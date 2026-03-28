import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter


def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """
    #创建数据加载器
    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        #创建一个字典，用于存储每个损失的平均值
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            #随机生成消息，消息长度为hidden_config.message_length
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            #训练模型，返回值为loss字典和(encoded_images, noised_images, decoded_messages)元组
            losses, _ = model.train_on_batch([image, message])
            #遍历loss字典中的每个损失，更新training_losses字典中的平均值
            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                #将损失写入日志
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1
        #计算单轮训练的时间消耗
        train_duration = time.time() - epoch_start
        #写日志
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        #写损失到csv文件
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)
        #每轮训练结束后，验证模型
        first_iteration = True
        #创建一个字典，用于存储每个损失的平均值同training_losses
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            #随机生成消息，消息长度为hidden_config.message_length
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            #验证模型，返回值为loss字典和(encoded_images, noised_images, decoded_messages)元组
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])
            #遍历loss字典中的每个损失，更新validation_losses字典中的平均值
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            #将这一批次的图片与隐写后图片的对比图保存，每轮只保存第一批次的images_to_save张图片
            if first_iteration:
                #如果启用了半精度训练，将图片转换为float类型，否则会报错
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False
        #将损失写入日志
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        #保存模型检查点
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        #写损失到csv文件
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
