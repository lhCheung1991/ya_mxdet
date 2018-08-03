#!/usr/bin/python3
# Copyright 2017, Linghan Zhang<lhcheung1991@gmail.com>

import random
import cv2
import numpy as np
import mxnet as mx
import json
from imgaug import augmenters as iaa


class TUPUFaceDataset(mx.gluon.data.Dataset):
    """
    Wrapper of TUPU Face Dataset in json file.
    """
    tupuface_class_name = ['__background__', 'face']

    def __init__(self, input_json_files, transform=None, resize_func=None, shuffle=True, imgaug_seq=None, random_crop=False, **kwargs):
        """
        Args:
            annotation_dir: a string describing the path of annotation XML files.
            img_dir: a string describing the path of JPEG images.
            dataset_index: filename of a file containing the IDs of all images used to constructing this dataset
        """
        super(TUPUFaceDataset, self).__init__(**kwargs)
        self.json_line_lst = []

        json_line_lst = []
        for f_name in input_json_files:
            with open(f_name, "r") as f:
                json_line_lst += f.readlines()[:]

        for k, v in enumerate(json_line_lst):
            line_cont_lst = v.strip().split()
            if len(line_cont_lst) < 3:
                continue
            self.json_line_lst.append(v)

        if shuffle is True:
            random.shuffle(self.json_line_lst)
        self.transform = transform
        self.resize_func = resize_func
        self.random_crop = random_crop

        # data augmentation
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        if imgaug_seq is None:
            self.seq = iaa.Sequential([])
        else:
            self.seq = imgaug_seq
   
    def __getitem__(self, idx):

        try:
            _str = self.json_line_lst[idx]
            _str_lst = _str.split()
            img_path, json_str = (_str_lst[0], _str_lst[1])
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            json_obj = json.loads(json_str)

        except (ValueError, Exception):
            raise ValueError

        gt = []
        for i in json_obj:
            bbx = []
            bbx.append(i[u"xmin"])
            bbx.append(i[u"ymin"])
            bbx.append(i[u"xmax"])
            bbx.append(i[u"ymax"])
            bbx.append(1)
            gt.append(bbx)
        gt = np.asarray(gt, dtype=np.float32)

        # augmentation
        area = (gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2])
        if self.random_crop == True and np.random.uniform(0, 4) < 1 and np.max(area) > 6400:
            t = np.argmax(area)
            xmin_t, ymin_t, xmax_t, ymax_t = (gt[t, 1], gt[t, 2], gt[t, 3], gt[t, 4])
            xmin_t = max(0, xmin_t - 20)
            ymin_t = max(0, ymin_t - 20)
            xmax_t = min(xmax_t + 20, img_w)
            ymax_t = min(ymax_t + 20, img_h)

            img = img[int(ymin_t):int(ymax_t), int(xmin_t):int(xmax_t), :]
            gt = np.asarray([[1, gt[t, 1]-xmin_t, gt[t, 2]-ymin_t, gt[t, 3]-xmin_t, gt[t, 4]-ymin_t]], dtype=np.float32)

        seq_det = self.seq.to_deterministic() # call this for each batch again, NOT only once at the start
        images_aug = seq_det.augment_images(np.expand_dims(img, axis=0))
        img = images_aug[0]

        # resize
        if self.resize_func is not None:
            # img, scale = self.resize_func(img)
            img, scale = self.resize_func(img, target_h=600, target_w=1066)
        else:
            scale = 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        gt[:, 0:4] *= scale

        if self.transform is not None:
            img, gt = self.transform(img, gt)

        gt_pad = np.ones((30 - gt.shape[0], 5)) * -1
        gt = np.concatenate([gt, gt_pad], axis=0)

        return mx.nd.array(img), mx.nd.array(gt)

    def __len__(self):
        return len(self.json_line_lst)
