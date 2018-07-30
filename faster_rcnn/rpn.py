#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

"""
RPN: Region Proposal Network
"""

import mxnet as mx
from gluoncv.model_zoo import model_zoo as models

def setConvWeights(lv: mx.gluon.nn.Conv2D, rv: mx.gluon.nn.Conv2D):
    lv.weight.set_data(rv.weight.data())
    lv.bias.set_data(rv.bias.data())

class DetectorHead(mx.gluon.HybridBlock):
    def __init__(self, num_anchors, **kwargs):
        super(DetectorHead, self).__init__(**kwargs)
        self.conv1 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=(1,1), activation='relu', weight_initializer=mx.init.Normal(0.01))
        self.conv_cls = mx.gluon.nn.Conv2D(channels=2*num_anchors, kernel_size=(1, 1),padding=(0, 0), weight_initializer=mx.init.Normal(0.01))
        self.conv_reg = mx.gluon.nn.Conv2D(channels=4*num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=mx.init.Normal(0.01))
    
    def hybrid_forward(self, F, feature, *args):
        f = self.conv1(feature)
        f_cls = self.conv_cls(f)
        f_reg = self.conv_reg(f)
        return f_cls, f_reg

    def init_params(self, ctx):
        self.collect_params().initialize(ctx=ctx)


class LightHead(mx.gluon.HybridBlock):
    '''
    Li, Z., Peng, C., Yu, G., Zhang, X., Deng, Y., & Sun, J. (2017, November 20). Light-Head R-CNN: In Defense of Two-Stage Object Detector. Arxiv.org.
    '''
    def __init__(self, num_cmid, **kwargs):
        super(LightHead, self).__init__(**kwargs)
        self.conv_new_1 = mx.gluon.nn.Conv2D(channels=num_cmid, kernel_size=(15, 1), padding=(7, 0), activation="relu", weight_initializer=mx.init.Normal(0.01))
        self.conv_new_2 = mx.gluon.nn.Conv2D(channels=10*7*7, kernel_size=(1, 15), padding=(0, 7), activation="relu", weight_initializer=mx.init.Normal(0.01))
        self.conv_new_3 = mx.gluon.nn.Conv2D(channels=num_cmid, kernel_size=(1, 15), padding=(0, 7), activation="relu", weight_initializer=mx.init.Normal(0.01))
        self.conv_new_4 = mx.gluon.nn.Conv2D(channels=10*7*7, kernel_size=(15, 1), padding=(7, 0), activation="relu", weight_initializer=mx.init.Normal(0.01))
    
    def hybrid_forward(self, F, feature, *args):
        f1 = self.conv_new_1(feature)
        f1 = self.conv_new_2(f1)
        f2 = self.conv_new_3(feature)
        f2 = self.conv_new_4(f2)
        f = f1 + f2
        return f

    def init_params(self, ctx):
        self.collect_params().initialize(ctx=ctx)


class RPNBlock(mx.gluon.HybridBlock):
    """ RPNBlock: region proposal network block

    Attributes:
      num_anchors: The number of anchors this RPN should predict.
      pretrained_model: A pretrained model as the base architecture of this
                        region proposal network. The default value is VGG16.
                        You may choose other models in gluon model zoo.
      feature_name: The name of feature in pretrained model. The name varies
                    for different models and different stages. 
    """
    def __init__(self, num_anchors, pretrained_model="vgg16", feature_name='vgg0_conv12_fwd_output', ctx=mx.cpu(), **kwargs):
        super(RPNBlock, self).__init__(**kwargs)
        # self.feature_extractor = None
        self.feature_model = pretrained_model
        self.feature_name = feature_name
        # get feature exactor
        feature_model = models.get_model(self.feature_model, pretrained=True, ctx=ctx)
        input_var = mx.sym.var('data')
        out_var = feature_model(input_var)
        internals = out_var.get_internals()
        feature_list = internals.list_outputs()
        # make sure the feature user want exists
        assert self.feature_name in feature_list
        feature_requested = internals[self.feature_name]
        self.feature_exactor = mx.gluon.SymbolBlock(feature_requested, input_var, params=feature_model.collect_params())

        self.head = DetectorHead(num_anchors)
        self.light_head = LightHead(num_cmid=64)
    
    def hybrid_forward(self, F, data, *args):
        # standard Faster R-CNN
        # f = self.feature_exactor(data)
        # f_cls, f_reg = self.head(f)
        # return f_cls, f_reg, f

        # Light Head R-CNN
        f = self.feature_exactor(data)
        f_cls, f_reg = self.head(f)
        f = self.light_head(f)
        return f_cls, f_reg, f
    
    def init_params(self, ctx):
        self.head.init_params(ctx)
        self.light_head.init_params(ctx)
