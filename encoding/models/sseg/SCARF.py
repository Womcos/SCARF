
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.parameter import Parameter

from .base import BaseNet
from .fcn import FCNHead

class SCARF(BaseNet):
    r"""DeepLabV3_plus

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).

    """
    def __init__(self, nclass, backbone, DS=False, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SCARF, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = SCARFHead(nclass, norm_layer, self._up_kwargs)
        self.DS = DS

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.base_forward(x)    # [c1, c2, c3, c4]
        x = self.head(x)
        outputs = []
        if self.DS:
            for output in x:
                outputs.append(interpolate(output, (h, w), **self._up_kwargs))
        else:
            outputs.append(interpolate(x[0], (h, w), **self._up_kwargs))
        return tuple(outputs)

def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

def one_conv(in_channels, out_channels, kernel_size, padding, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class SCARFHead(nn.Module):
    def __init__(self, nclass, norm_layer, up_kwargs, atrous_rates=[12, 24, 36], **kwargs):
        super(SCARFHead, self).__init__()
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        inter_channels = 256
        decoder_filters = 128
        self.aspp = ASPP_Module(2048, atrous_rates, norm_layer, up_kwargs, **kwargs)
        self.weight = Parameter(torch.ones([nclass, 1]) * 0.04)
        new_feat_conv_list = []
        cat_merge_list = []     # cat in low resolutions
        ACF_decoder_list = []
        aggregation_logit_list = []     # logit in all resolutions
        for i in range(3):
            new_feat_conv_list.append(
                one_conv(int(1024 * ((1 / 2)**i)), decoder_filters, kernel_size=1, padding=0, norm_layer=norm_layer)
            )
            if i == 0:
                cat_filters = inter_channels + decoder_filters
            else:
                cat_filters = decoder_filters * 2
            cat_merge_list.append(nn.Sequential(
                one_conv(cat_filters, decoder_filters, 3, 1, norm_layer),
                one_conv(decoder_filters, decoder_filters, 3, 1, norm_layer)
            ))
            ACF_decoder_list.append(
                one_conv(decoder_filters, decoder_filters, kernel_size=1, padding=0, norm_layer=norm_layer)
            )
        for i in range(4):
            if i == 0:
                aggregatation_filters = inter_channels
            else:
                aggregatation_filters = decoder_filters * 2
            aggregation_logit_list.append(nn.Sequential(
                one_conv(aggregatation_filters, decoder_filters, 1, 0, norm_layer),
                nn.Conv2d(decoder_filters, nclass, kernel_size=3, padding=1)
            ))

        self.new_feat_conv_list = nn.ModuleList(new_feat_conv_list)
        self.cat_merge_list = nn.ModuleList(cat_merge_list)
        self.ACF_decoder_list = nn.ModuleList(ACF_decoder_list)
        self.aggregation_logit_list = nn.ModuleList(aggregation_logit_list)

    def forward(self, x):
        c1, c2, c3, c4 = x
        new_feat_list = [c3, c2, c1]
        outputs = []
        weight = torch.sigmoid(100 * self.weight)    # accelerate the learning rate
        decoder_feat = self.aspp(c4)
        logit = self.aggregation_logit_list[0](decoder_feat)
        outputs.append(logit)
        for i in range(3):
            new_feat = self.new_feat_conv_list[i](new_feat_list[i])
            _, _, h, w = new_feat.size()
            decoder_feat = interpolate(decoder_feat, (h, w), **self._up_kwargs)
            logit = interpolate(logit, (h, w), **self._up_kwargs)
            decoder_feat = torch.cat([decoder_feat, new_feat], dim=1)
            decoder_feat = self.cat_merge_list[i](decoder_feat)
            SCARF_feat = self.ACF_decoder_list[i](decoder_feat)
            b, c, h, w = SCARF_feat.shape
            SCARF_feat = SCARF_feat.view([b, c, h*w]).permute(0, 2, 1)  # b, h*w, c
            SCARF_soft = torch.softmax(logit, dim=1).view(b, self.nclass, h*w)    # b, nclass, h*w
            SCARF_soft_sum = torch.sum(SCARF_soft, dim=-1, keepdim=True)  # b, nclass, 1
            SCARF_soft_weight = SCARF_soft.permute(0, 2, 1).contiguous().view(b*h*w, self.nclass)
            Lambda = torch.matmul(SCARF_soft_weight, weight).view(b, h*w, 1)    # b, h*w, 1
            term2 = SCARF_feat - Lambda * SCARF_feat

            SCARF_feat = torch.matmul(SCARF_soft, SCARF_feat)  # b, nclass, c
            SCARF_soft = SCARF_soft.permute(0, 2, 1)    # b, h*w, nclass
            SCARF_feat = torch.matmul(SCARF_soft, SCARF_feat)  # b, h*w, c
            N = torch.matmul(SCARF_soft, SCARF_soft_sum)   # b, h*w, 1
            term1 = SCARF_feat * Lambda / N
            SCARF_feat = term1 + term2
            SCARF_feat = SCARF_feat.permute([0, 2, 1]).view(b, c, h, w)
            aggregation_feat = torch.cat([decoder_feat, SCARF_feat], dim=1)
            logit = self.aggregation_logit_list[i + 1](aggregation_feat) + logit
            outputs.append(logit)
        outputs = outputs[::-1]
        return outputs

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

def get_SCARF(dataset='pascal_voc', backbone='resnet50s', pretrained=False,
            root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ...datasets import datasets, acronyms
    model = SCARF(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('deeplab_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model
