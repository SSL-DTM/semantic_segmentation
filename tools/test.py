# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import pandas as pd



from semseg import models
from semseg.config import config
from semseg import data
from semseg.config import update_config
from torch.nn import MSELoss
from semseg.utils.modelsummary import get_model_summary
from semseg.utils.utils import create_logger, FullModel
from semseg.core.function import train, validate, testval, test
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME +
                 '.get_model')(config)

    if config.MODEL.NAME != 'swin':
        dump_input = torch.rand(
            (1, config.DATASET.INPUT_CHANNELS, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    test_dataset = eval('data.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        split=config.DATASET.TEST_DIR,
        img_dirs=config.DATASET.IMG_DIRS,
        lbl_dir=config.DATASET.LBL_DIR,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        img_ext=config.DATASET.IMG_EXT,
        lbl_ext=config.DATASET.LBL_EXT)

    logger.info('len test data: {}'.format(len(test_dataset)))

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)


    sv_dir = None

    start = timeit.default_timer()
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
                                                       test_dataset,
                                                       testloader,
                                                       model,
                                                       sv_dir=sv_dir,
                                                       sv_pred=False)

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
                                                pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Test prediction took Mins: %d' % np.int((end - start) / 60))

    logger.info('Done')

    if config.TEST_RESULTS_CSV is not None:
        pth, file = os.path.split(config.TEST_RESULTS_CSV)
        os.makedirs(pth, exist_ok=True)
        with open(config.TEST_RESULTS_CSV, 'a') as writer:
            writer.writelines('{},{},{}\n'.format(final_output_dir, mean_IoU, IoU_array))



if __name__ == '__main__':
    main()