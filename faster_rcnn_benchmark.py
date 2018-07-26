#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
import os
import argparse
from faster_rcnn.config import cfg
from TUPUFaceDataset import TUPUFaceDataset
from faster_rcnn.faster_rcnn import FasterRCNN
import mxnet as mx
from faster_rcnn.utils import imagenetNormalize, img_resize, bbox_inverse_transform, bbox_clip
from faster_rcnn.rpn_proposal import proposal_test
from faster_rcnn.nms import nms


def parse_args():
    parser = argparse.ArgumentParser(description="Test Faster RCNN")
    parser.add_argument('model_file', metavar='model_file', type=str)
    return parser.parse_args()


def test_transformation(data, label):
    data = imagenetNormalize(data)
    return data, label


def benchmark(net, ctx, benchmark_save_path):

    global f_path

    test_dataset = TUPUFaceDataset(
        cfg.test_dataset_json_lst,
        transform=test_transformation,
        resize_func=img_resize,
        shuffle=False
    )
    
    test_datait = mx.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    ctx = ctx
    with open(cfg.test_dataset_json_lst[0], "r") as f:
    # with open(f_path, "r") as f:
       test_data_lines = f.readlines()
    
    with open(benchmark_save_path, "w") as out_file:
    
        for it, (data, label) in enumerate(test_datait):
            if it >= 1000:
               break
            _str_lst = test_data_lines[it].split()
            file_path, _, _ = (_str_lst[0], _str_lst[1], _str_lst[2])
            data = data.as_in_context(ctx)
            _n, _c, h, w = data.shape
            label = label.as_in_context(ctx)
            rpn_cls, rpn_reg, f = net.rpn(data)
            f_height = f.shape[2]
            f_width = f.shape[3]
            rpn_bbox_pred = proposal_test(rpn_cls, rpn_reg, f.shape, data.shape, ctx)
        
            # RCNN part 
            # add batch dimension
            rpn_bbox_pred_attach_batchid = mx.nd.concatenate([mx.nd.zeros((rpn_bbox_pred.shape[0], 1), ctx), rpn_bbox_pred], axis=1)
            f = mx.nd.ROIPooling(f, rpn_bbox_pred_attach_batchid, (7, 7), 1.0/16) # VGG16 based spatial stride=16
            rcnn_cls, rcnn_reg = net.rcnn(f)
            rcnn_bbox_pred = mx.nd.zeros(rcnn_reg.shape)
            for i in range(len(test_dataset.tupuface_class_name)):
                rcnn_bbox_pred[:, i*4:(i+1)*4] = bbox_clip(bbox_inverse_transform(rpn_bbox_pred, rcnn_reg[:, i*4:(i+1)*4]), h, w)
            rcnn_cls = mx.nd.softmax(rcnn_cls)
        
            bboxes = rcnn_bbox_pred.asnumpy()
            cls_scores = rcnn_cls.asnumpy()
        
            # NMS by class
            for cls_id in range(1, len(test_dataset.tupuface_class_name)):
                cur_scores = cls_scores[:, cls_id]
                bboxes_pick = bboxes[:, cls_id * 4: (cls_id+1)*4]
                cur_scores, bboxes_pick = nms(cur_scores, bboxes_pick, cfg.rcnn_nms_thresh)
                bboxes_pick[:, 0] = bboxes_pick[:, 0] / w
                bboxes_pick[:, 2] = bboxes_pick[:, 2] / w
                bboxes_pick[:, 1] = bboxes_pick[:, 1] / h
                bboxes_pick[:, 3] = bboxes_pick[:, 3] / h
                bboxes_pick[:, 2] = bboxes_pick[:, 2] - bboxes_pick[:, 0]
                bboxes_pick[:, 3] = bboxes_pick[:, 3] - bboxes_pick[:, 1]
                # for i in range(len(cur_scores)):
                #     if cur_scores[i] >= cfg.rcnn_score_thresh:
                #         bbox = bboxes_pick[i]
                mask = cur_scores > cfg.rcnn_score_thresh
                cur_scores = cur_scores[mask]
                bboxes_pick = bboxes_pick[mask]

                pred_bbxs_num = len(cur_scores)
                if pred_bbxs_num != 0:
                    cur_rec = "{} {} {} ".format(file_path, pred_bbxs_num, "")
                    for i in range(pred_bbxs_num):
                        str_out = "{} {} {} {} {} ".format(
                                    bboxes_pick[i, 0],
                                    bboxes_pick[i, 1],
                                    bboxes_pick[i, 2],
                                    bboxes_pick[i, 3],
                                    cur_scores[i]
                                )
                        cur_rec += str_out
                    cur_rec += "\n"
                else:
                    cur_rec = "{} {}\n".format(file_path, 0)
                out_file.write(cur_rec)


if __name__ == "__main__":
    ctx = mx.gpu(7)
    net = FasterRCNN(
        len(cfg.anchor_ratios) * len(cfg.anchor_scales), 
        cfg.num_classes, 
        pretrained_model="vgg16",
        feature_name="vgg0_conv12_fwd_output",
        ctx=ctx)
    net.init_params(ctx)
    net.collect_params().load("/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/faster-rcnn-vgg16-9anchors/faster-rcnn-vgg16-9anchors-140000.gluonmodel", ctx)
    
    '''
    global f_path
    path_lst = os.listdir("/world/data-c40/wuhuimin/data/live_faces/detection_input")
    path_lst.sort()
    for f_name in path_lst: 
        f_path = os.path.join("/world/data-c40/wuhuimin/data/live_faces/detection_input", f_name)
        print("processing {}".format(f_path))
        f_path_out = os.path.join("/world/data-c40/wuhuimin/data/live_faces/detection_output", f_name.strip())
        benchmark(net, ctx, f_path_out)
    '''
    benchmark(net, ctx, "/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/faster-rcnn-vgg16-9anchors/faster-rcnn-vgg16-9anchors-140000.benchmark")
