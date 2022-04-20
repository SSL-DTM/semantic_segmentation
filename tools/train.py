# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
# import random
# import torch
# import numpy as np
#
# random.seed(42)     # python random generator
# np.random.seed(42)  # numpy random generator
#
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)


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

from semseg import models
from semseg.config import config
from semseg import data
from semseg.config import update_config
from torch.nn import MSELoss
from semseg.utils.modelsummary import get_model_summary
from semseg.utils.utils import create_logger, FullModel
from semseg.core.function import train, validate
import math
from semseg.core.criterion import CrossEntropy, OhemCrossEntropy


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from semseg.utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # save config to experiment directory
    with open(os.path.join(final_output_dir, 'config.yaml'), 'w') as f:
        print(config, file=f)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

        # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_model')(config)

    if config.MODEL.NAME != 'swin':
        dump_input = torch.rand(
            (1, config.DATASET.INPUT_CHANNELS, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # copy model file
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        # if os.path.exists(models_dst_dir):
        #     shutil.rmtree(models_dst_dir)
        # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    # labeled data!!
    train_dataset = eval('data.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        split=config.DATASET.TRAIN_DIR,
        img_dirs=config.DATASET.IMG_DIRS,
        lbl_dir=config.DATASET.LBL_DIR,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        img_ext=config.DATASET.IMG_EXT,
        lbl_ext=config.DATASET.LBL_EXT,
        class_weights=config.DATASET.CLASS_WEIGHTS,
        scale_factor=config.TRAIN.SCALE_FACTOR)
    logger.info('Got {} labeled examples!'.format(len(train_dataset)))

    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('data.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        split=config.DATASET.VALID_DIR,
        img_dirs=config.DATASET.IMG_DIRS,
        lbl_dir=config.DATASET.LBL_DIR,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        img_ext=config.DATASET.IMG_EXT,
        class_weights=config.DATASET.CLASS_WEIGHTS,
        lbl_ext=config.DATASET.LBL_EXT)

    test_sampler = get_sampler(test_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    model = FullModel(model, criterion)
    if distributed:
        print('distributed!!!!!')
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    # if config.TRAIN.OPTIMIZER == 'SGD':

    params_dict = dict(model.named_parameters())
    if config.TRAIN.NONBACKBONE_KEYWORDS:
        bb_lr = []
        nbb_lr = []
        nbb_keys = set()
        for k, param in params_dict.items():
            if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                nbb_lr.append(param)
                nbb_keys.add(k)
            else:
                bb_lr.append(param)
        print(nbb_keys)
        params = [{'params': bb_lr, 'lr': config.TRAIN.LR},
                  {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
    else:
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

    if config.TRAIN.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    elif config.TRAIN.OPTIMIZER == 'RMSProp':
        optimizer = torch.optim.RMSprop(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD
                                    )
    else: #ADAM
        optimizer = torch.optim.Adam(params,
                                    lr=config.TRAIN.LR,
                                    weight_decay=config.TRAIN.WD
                                    )
    # else:
    #     raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    print('NUM ITERS: {}'.format(epoch_iters))

    if config.TRAIN.REDUCE_LR_ON_PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10) # max because we are looking for miou!

    last_epoch = 0
    best_mIoU = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            last_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            dct = checkpoint['state_dict']

            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train_loss = train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters, config.TRAIN.LR, num_iters, trainloader,
                           optimizer, model, writer_dict)

        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)

        if config.TRAIN.REDUCE_LR_ON_PLATEAU:
            scheduler.step(mean_IoU)

        if args.local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            logging.info('Train Loss: {:.3f}'.format(train_loss))
            msg = 'Val Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict(),
                   os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end - start) / 3600))
        logger.info('Done')


if __name__ == '__main__':
    main()
