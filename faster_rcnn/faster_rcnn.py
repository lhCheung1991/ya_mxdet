#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import mxnet as mx
from .config import cfg
from .rpn import RPNBlock

def _set_dense_weights(lv: mx.gluon.nn.Dense, rv: mx.gluon.nn.Dense):
    lv.weight.set_data(rv.weight.data())
    lv.bias.set_data(rv.bias.data())


class RCNNBlock(mx.gluon.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(RCNNBlock, self).__init__(**kwargs)
        self.fc6 = mx.gluon.nn.Dense(units=4096, activation='relu')
        self.fc7 = mx.gluon.nn.Dense(in_units=4096, units=4096, activation='relu')
        self.cls_fc = mx.gluon.nn.Dense(in_units=4096, units=num_classes, activation=None)
        self.reg_fc = mx.gluon.nn.Dense(in_units=4096, units=num_classes*4, activation=None)
    
    def hybrid_forward(self, F, f, **kwargs):
        f = self.fc6(f)
        f = self.fc7(f)
        cls_output = self.cls_fc(f)
        reg_output = self.reg_fc(f)
        return cls_output, reg_output
    
    def init_by_vgg(self, ctx):
        self.collect_params().initialize(mx.init.Normal(), ctx=ctx)
        vgg16 = mx.gluon.model_zoo.vision.vgg16(pretrained=True)
        # _set_dense_weights(self.fc6, vgg16.features[31])
        # _set_dense_weights(self.fc7, vgg16.features[33])


class LightHeadRCNNBlock(mx.gluon.HybridBlock):
    '''
    Li, Z., Peng, C., Yu, G., Zhang, X., Deng, Y., & Sun, J. (2017, November 20). Light-Head R-CNN: In Defense of Two-Stage Object Detector. Arxiv.org.
    '''
    def __init__(self, num_classes, **kwargs):
        super(LightHeadRCNNBlock, self).__init__(**kwargs)
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        self.fc_new_1 = mx.gluon.nn.Dense(units=2048, activation="relu")
        self.cls_score = mx.gluon.nn.Dense(in_units=2048, units=num_classes, activation=None) 
        self.bbox_pred = mx.gluon.nn.Dense(in_units=2048, units=num_reg_classes*4, activation=None) 
    
    def hybrid_forward(self, F, f, **kwargs):
        f = self.fc_new_1(f)
        cls_output = self.cls_score(f)
        reg_output = self.bbox_pred(f)
        return cls_output, reg_output
    
    def init_by_vgg(self, ctx):
        self.collect_params().initialize(mx.init.Normal(), ctx=ctx)


class FasterRCNN(mx.gluon.HybridBlock):
    def __init__(self, num_anchors, num_classes, ctx=mx.cpu(), **kwargs):
        super(FasterRCNN, self).__init__()
        self.rpn = RPNBlock(
                num_anchors,
                pretrained_model=kwargs["pretrained_model"],
                feature_name=kwargs["feature_name"],
                ctx=ctx)

        # standard Faster R-CNN
        self.rcnn = RCNNBlock(num_classes)

        # Light Head R-CNN
        # self.rcnn = LightHeadRCNNBlock(num_classes)
        
    def hybrid_forward(self, F, x, **kwargs):
        raise NotImplementedError
    
    def init_params(self, ctx):
        self.rpn.init_params(ctx)
        self.rcnn.init_by_vgg(ctx)
