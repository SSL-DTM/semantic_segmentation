# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data
import albumentations as A


class BaseDataset(data.Dataset):
    def __init__(
            self,
            base_size=1024,
            crop_size=(1024, 1024),
            scale_factor=16,
            ignore_label=-1,
            class_weights=None
    ):
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.class_weights = class_weights

        self.scale_factor = scale_factor

        self.files = []
        self.aug = A.Compose(
            [
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.Transpose(p=0.3),
                A.RandomRotate90(p=0.3),
                A.OneOf(
                    [
                        A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05, alpha_affine=120 * 0.03),
                        A.GridDistortion(p=0.3),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3)
                    ],
                    p=0.3
                ),
                A.PadIfNeeded(min_height=self.crop_size[0], min_width=self.crop_size[1], p=1.0),
                A.RandomCrop(width=self.crop_size[1], height=self.crop_size[0], p=1.0)
            ]
        )

    def __len__(self):
        return len(self.files)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:2]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):

        h, w = image.shape[:2]
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = np.int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int(h * short_length / w + 0.5)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=0
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image

    def gen_sample(self, image, label,
                   multi_scale=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        augmented = self.aug(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        image = image.transpose((2, 0, 1))

        return image, label

    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        return pred.exp()

    def multi_scale_inference(self, config, model, image, scales=[1]):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)

        final_pred = torch.zeros([1, self.num_output_channels ,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:2]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, 0)
                if new_img.ndim == 2:
                    new_img = np.expand_dims(new_img, -1)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width,
                                             self.crop_size, 0)
                new_h, new_w = new_img.shape[:2]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_output_channels,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img,
                                                      h1-h0,
                                                      w1-w0,
                                                      self.crop_size,
                                                      0)
                        if crop_img.ndim == 2:
                            crop_img = np.expand_dims(crop_img, -1)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    # BY BASHIR
    def simple_inference(self, config, model, image):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        new_img = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)

        height, width = new_img.shape[:2]

        if max(height, width) <= np.min(self.crop_size):
            new_img = self.pad_image(new_img, height, width,
                                     self.crop_size, 0)
            if new_img.ndim == 2:
                new_img = np.expand_dims(new_img, -1)
            new_img = new_img.transpose((2, 0, 1))
            new_img = np.expand_dims(new_img, axis=0)
            new_img = torch.from_numpy(new_img)
            preds = self.inference(config, model, new_img)
            preds = preds[:, :, 0:height, 0:width]
        else:
            if height < self.crop_size[0] or width < self.crop_size[1]:
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, 0)
            new_h, new_w = new_img.shape[:2]
            rows = np.int(np.ceil(1.0 * (new_h -
                                         self.crop_size[0]) / stride_h)) + 1
            cols = np.int(np.ceil(1.0 * (new_w -
                                         self.crop_size[1]) / stride_w)) + 1
            preds = torch.zeros([1, self.num_output_channels,
                                 new_h, new_w]).cuda()
            count = torch.zeros([1, 1, new_h, new_w]).cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + self.crop_size[0], new_h)
                    w1 = min(w0 + self.crop_size[1], new_w)
                    crop_img = new_img[h0:h1, w0:w1]
                    if h1 == new_h or w1 == new_w:
                        crop_img = self.pad_image(crop_img,
                                                  h1-h0,
                                                  w1-w0,
                                                  self.crop_size,
                                                  0)
                    if crop_img.ndim == 2:
                        crop_img = np.expand_dims(crop_img, -1)
                    crop_img = crop_img.transpose((2, 0, 1))
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.from_numpy(crop_img)
                    pred = self.inference(config, model, crop_img, False)
                    preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                    count[:, :, h0:h1, w0:w1] += 1
            preds = preds / count
            preds = preds[:, :, :height, :width]

        preds = F.interpolate(
            preds, (ori_height, ori_width),
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        return preds
