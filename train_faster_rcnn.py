#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

from faster_rcnn.config import cfg
# from VOCDataset import VOCDataset
from TUPUFaceDataset import TUPUFaceDataset
from faster_rcnn.faster_rcnn import FasterRCNN
import mxnet as mx
from faster_rcnn.utils import random_flip, imagenetNormalize, img_resize, img_resize_fix, random_square_crop, select_class_generator, bbox_inverse_transform, softmax_celoss_with_ignore
from faster_rcnn.rpn_gt_opr import rpn_gt_opr
from faster_rcnn.rpn_proposal import proposal_train
import os
import argparse
import logging
from faster_rcnn_benchmark import *
from imgaug import augmenters as iaa
from mxboard import SummaryWriter

import sys;sys.path.append("/home/zhanglinghan/face-detection-benchmark-pytorch")
from evals import evals
import numpy as np
import glob


def logging_system():
    global args
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_path, args.logger), 'w')
    formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s'
        )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def train_transformation(data, label):
    data, label = random_flip(data, label)
    data = imagenetNormalize(data)
    return data, label


def train_dataset():
    '''
    prepare a custom dataset
    return: train_dataset
    '''
    # train_dataset = VOCDataset(annotation_dir=cfg.annotation_dir,
    #                           img_dir=cfg.img_dir,
    #                           dataset_index=cfg.dataset_index,
    #                           transform=train_transformation,
    #                           resize_func=img_resize)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes(iaa.AdditiveGaussianNoise(scale=(10, 0.1)))
    ])
    training_dataset = TUPUFaceDataset(
        cfg.input_json_files,
        transform=train_transformation,
        # resize_func=img_resize,
        resize_func=img_resize_fix,
        shuffle=True,
        imgaug_seq=seq
    )

    return training_dataset


def main():
    global args, logger
    train_ds = train_dataset()

    CTX = [mx.gpu(int(i)) for i in args.gpus.split(",")]
    n_gpus = len(CTX)
    assert args.batch_size == n_gpus
    logger.info(CTX)
    logger.info("batch_size = {}, 1 images per GPU".format(n_gpus))

    train_datait = mx.gluon.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            last_batch="discard"
            )

    net = FasterRCNN(
            len(cfg.anchor_ratios) * len(cfg.anchor_scales),
            cfg.num_classes,
            pretrained_model=args.model,
            feature_name=args.feature_name,
            ctx=CTX)
    net.init_params(CTX)
    if args.pretrained_model != "":
        net.collect_params().load(args.pretrained_model, CTX)
        logger.info("loading {}".format(args.pretrained_model))

    lr_schdl = mx.lr_scheduler.FactorScheduler(step=80000, factor=0.5)
    trainer = mx.gluon.trainer.Trainer(net.collect_params(), 'sgd',
                                        optimizer_params={
                                            'learning_rate': args.learning_rate,
                                            'wd': args.weight_decay,
                                            "lr_scheduler": lr_schdl,
                                            'momentum': 0.9
                                        })
    anchors_count = len(cfg.anchor_ratios) * len(cfg.anchor_scales)

    files = glob.glob(os.path.join(args.save_path, "logs", "*"))
    for f in files:
        os.remove(f)

    for epoch in range(0, args.epochs):

        loss_interval = 0.0
        rpn_loss_cls_interval = 0.0
        rpn_loss_reg_interval = 0.0
        rcnn_loss_cls_interval = 0.0
        rcnn_loss_reg_interval = 0.0

        for it, (datas, labels) in enumerate(train_datait):
            datas_lst = mx.gluon.utils.split_and_load(datas, CTX)
            labels_lst = mx.gluon.utils.split_and_load(labels, CTX)

            with mx.autograd.record():
                losses = []
                for data, label in zip(datas_lst, labels_lst):
                    _n, _c, h, w = data.shape

                    rpn_cls, rpn_reg, f = net.rpn(data)
                    f_height = f.shape[2]
                    f_width = f.shape[3]
                    rpn_cls_gt, rpn_reg_gt = rpn_gt_opr(rpn_reg.shape, label, data.context, h, w)
                    rpn_bbox_sampled, rcnn_reg_target, rcnn_cls_target = proposal_train(rpn_cls, rpn_reg, label, f.shape, data.shape, data.context)
    
                    # RPN Loss part
                    # Reshape and transpose to the shape of gt
                    rpn_cls = rpn_cls.reshape((1, -1, 2, f_height, f_width))
                    rpn_cls = mx.nd.transpose(rpn_cls, (0, 1, 3, 4, 2))
                    rpn_reg = mx.nd.transpose(rpn_reg.reshape((1, -1, 4, f_height, f_width)), (0, 1, 3, 4, 2))
                    mask = (rpn_cls_gt==1).reshape((1, anchors_count, f_height, f_width, 1)).broadcast_to((1, anchors_count, f_height, f_width, 4))
                    rpn_loss_reg = mx.nd.sum(mx.nd.smooth_l1((rpn_reg - rpn_reg_gt) * mask, 3.0)) / mx.nd.sum(mask)
                    rpn_loss_cls = softmax_celoss_with_ignore(rpn_cls.reshape((-1, 2)), rpn_cls_gt.reshape((-1, )), -1)
    
                    # RCNN part
                    # add batch dimension
                    rpn_bbox_sampled = mx.nd.concatenate([mx.nd.zeros((rpn_bbox_sampled.shape[0], 1), data.context), rpn_bbox_sampled], axis=1)
                    f = mx.nd.ROIPooling(f, rpn_bbox_sampled, (7, 7), 1.0/16) # VGG16 based spatial stride=16
                    rcnn_cls, rcnn_reg = net.rcnn(f)
                    mask = (rcnn_cls_target > 0).reshape((rcnn_cls_target.shape[0], 1)).broadcast_to((rcnn_cls_target.shape[0], 4*cfg.num_classes))
                    rcnn_loss_reg = mx.nd.sum(mx.nd.smooth_l1((rcnn_reg - rcnn_reg_target) * mask, 1.0)) / mx.nd.sum(mask)
                    rcnn_loss_cls = mx.nd.softmax_cross_entropy(rcnn_cls, rcnn_cls_target) / rcnn_cls.shape[0]
                    
                    loss = rpn_loss_cls + rpn_loss_reg + rcnn_loss_cls + rcnn_loss_reg
                    losses.append(loss)
                    loss_interval += loss.asscalar()
                    rpn_loss_cls_interval += rpn_loss_cls.asscalar()
                    rpn_loss_reg_interval += rpn_loss_reg.asscalar()
                    rcnn_loss_cls_interval += rcnn_loss_cls.asscalar()
                    rcnn_loss_reg_interval += rcnn_loss_reg.asscalar()
    
            for loss in losses:
                loss.backward()
            trainer.step(datas.shape[0])
            mx.nd.waitall()

            if it % args.log_train_freq == 0:

                logger.info("Epoch {:=2d}, train [{:>6d}/{:<6d}], loss={:>6.5f}, rpn_loss_cls={:>6.5f}, rpn_loss_reg={:>6.5f}, rcnn_loss_cls={:>6.5f}, rcnn_loss_reg={:>6.5f}, lr={:>6.5f}".format(
                            epoch,
                            it,
                            int(train_ds.__len__() / args.batch_size),
                            loss_interval / args.log_train_freq / args.batch_size,
                            rpn_loss_cls_interval / args.log_train_freq / args.batch_size,
                            rpn_loss_reg_interval / args.log_train_freq / args.batch_size,
                            rcnn_loss_cls_interval / args.log_train_freq / args.batch_size,
                            rcnn_loss_reg_interval / args.log_train_freq / args.batch_size,
                            trainer.learning_rate
                            )
                        )

                # tensorboard info
                with SummaryWriter(logdir=os.path.join(args.save_path, "logs")) as sw:
                    sw.add_scalar(tag='Loss', value=(loss_interval / args.log_train_freq / args.batch_size), global_step=it)
                    sw.add_scalar(tag='LR', value=trainer.learning_rate, global_step=it)

                loss_interval = 0.0
                rpn_loss_cls_interval = 0.0
                rpn_loss_reg_interval = 0.0
                rcnn_loss_cls_interval = 0.0
                rcnn_loss_reg_interval = 0.0


            if it % args.save_interval == 0:
                save_schema = os.path.split(args.save_path)[1] + "-{}"
                net.collect_params().save(os.path.join(args.save_path, save_schema.format(it) + ".gluonmodel"))
                benchmark(net, CTX[0], os.path.join(args.save_path, save_schema.format(it) + ".benchmark"))
                pr_curve = evals(
                    cfg.test_dataset_json_lst[0],  
                    os.path.join(args.save_path, save_schema.format(it) + ".benchmark"),
                    topk=1000,
                    reset=True)
                ap = np.trapz(pr_curve[:, 0], pr_curve[:, 1])

                # tensorboard info
                with SummaryWriter(logdir=os.path.join(args.save_path, "logs")) as sw:
                    sw.add_scalar(tag='AP', value=ap, global_step=it)


if __name__ == "__main__":
    global args, logger
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--gpus', default='0', type=str, help='identify gpus')
    parser.add_argument('--batch_size', '-b', default=1, type=int, help='mini batch')
    parser.add_argument('--learning_rate', '-lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', '-wd', default=0.000005, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--logger', default='training.log', type=str, help='')
    parser.add_argument('--log_train_freq', default=10, type=int, help='print log')
    parser.add_argument('--root_dir', default='./', type=str, help='path to root path of data')
    parser.add_argument('--save_path', default='', type=str, help='path to save checkpoint and log')
    parser.add_argument('--save_interval', default=10000, type=int, help='')
    parser.add_argument('--model', default='vgg16', type=str, help='path to backbone network')
    parser.add_argument('--pretrained_model', default='', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--feature_name', default='vgg0_conv12_fwd_output', type=str, help='feature to be extracted')
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    logger = logging_system()
    logger.info(vars(args))

    main()
