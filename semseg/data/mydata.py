# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from glob import glob

import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset

from tifffile import imread

import pandas as pd
import logging


class MyData(BaseDataset):
    def __init__(self,
                 root=None,
                 split=None,
                 img_dirs=None,
                 lbl_dir=None,
                 num_classes=5,
                 multi_scale=True,
                 base_size=256,
                 crop_size=(128, 128),
                 img_ext='.tif',
                 lbl_ext='.tif',
                 class_weights=None,
                 ignore_label=-1,
                 just_predict=False,
                 scale_factor=11):

        super(MyData, self).__init__(base_size, crop_size, scale_factor, ignore_label, class_weights)

        self.root = root
        self.num_classes = num_classes
        self.img_dirs = img_dirs
        self.lbl_dir = lbl_dir
        self.img_ext = img_ext
        self.lbl_ext = lbl_ext
        self.split = split
        self.just_predict = just_predict
        self.multi_scale = multi_scale
        self.files = self.read_files()

    def read_files(self):
        files = []

        img_files = sorted(glob(os.path.join(self.root, self.split, self.img_dirs[0],  '*{}'.format(self.img_ext))))
        for img_file in img_files:
            name = os.path.basename(img_file)
            sample = {
                'name': name
            }
            files.append(sample)

        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def normalize(self, image):
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + np.finfo(np.float32).eps)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        return image

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = [self.normalize(imread(os.path.join(self.root, self.split, im_dir, name))) for im_dir in self.img_dirs]
        image = np.concatenate(image, -1)
        # image = np.array(image)
        # if image.ndim == 4: # RGB has 3 channels!!!
        #     image = np.squeeze(image, 0)
        # image = np.transpose(image, (1, 2, 0))

        size = image.shape[:2]

        if self.just_predict:
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )

            if image.ndim == 2:
                image = np.expand_dims(image, -1)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label_path = os.path.join(self.root, self.split, self.lbl_dir, os.path.splitext(name)[0]+self.lbl_ext)
        label = imread(label_path)

        label[label == 255] = 0

        if self.split in ['valid', 'test']:
            # image, label = self.resize_short_length(
            #     image,
            #     label=label,
            #     short_length=self.base_size,
            #     fit_stride=8
            # )

            if image.ndim == 2:
                image = np.expand_dims(image, -1)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        # image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale)

        return image.copy(), label.copy(), np.array(size), name