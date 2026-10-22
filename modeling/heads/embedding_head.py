# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from torch import nn
import torch
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from fastreid.modeling.heads.multi_head_module import Multi_Heads, Multi_Heads_
@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on
        self.neck_feat = neck_feat
        '''
        

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_1 = nn.Sequential(*bottleneck)

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_2 = nn.Sequential(*bottleneck)
        
        '''
        self.gmp = nn.AdaptiveMaxPool2d(1)
        '''
        # classification layer
        # fmt: off
        if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim*2, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'cosSoftmax':    self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on
        '''
        #self.bottleneck_1.apply(weights_init_kaiming)
        #self.bottleneck_2.apply(weights_init_kaiming)
        #self.classifier.apply(weights_init_classifier)
        self.multi_heads_module_1 = Multi_Heads([1501,1812,1467,11934,1816], 2048, max_pool=True)
        self.multi_heads_module_2 = Multi_Heads([1501,1812,1467,11934,1816], 2048)
        #self.conv_1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
        #self.conv_2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)

        
        #self.dy_module = Dynamic_Module()
        #self.multi_heads = Multi_Heads([1501,1812,1467,11934,1816])
    def forward(self, features, sources=None, targets=None, opt=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        #global_feat = features.unsqueeze(2).unsqueeze(3)
        #features_1 = self.conv_1(features)
        #features_2 = self.conv_2(features)
        global_feat = self.pool_layer(features)
        pooled_feature = features[..., 0, 0]
        N, C = pooled_feature.shape
        
        
        
        # mask_1 = self.mask_1(pooled_feature).view(N, C, 1, 1)
        # feature_1 = features*mask_1
        # global_feat_1 = global_feat*mask_1
        features_1 = features
        global_feat_1 = self.gmp(features)
        
        features_2 = features
        global_feat_2 = global_feat
        
        
        # bn_feat_1 = self.bottleneck_1(global_feat_1)
        # bn_feat_1 = bn_feat_1[..., 0, 0]
        
        # bn_feat_2 = self.bottleneck_2(global_feat_2)
        # bn_feat_2 = bn_feat_2[..., 0, 0]
        
        
        
        global_feat = torch.cat([global_feat_1, global_feat_2], dim=1)
        bn_feat = torch.cat([global_feat_1, global_feat_2], dim=1)[..., 0, 0]
        if not self.training: return global_feat[..., 0, 0]
        #score_1 = self.multi_heads_module_1(features_1,opt=opt)
        #score_2 = self.multi_heads_module_2(features_2,opt=opt)
        score_1 = self.multi_heads_module_1(features_1)
        score_2 = self.multi_heads_module_2(features_2)
        score_list = [score_1, score_2]
        #score_list = []
        
        
        
        #out_list = self.dy_module(features)
        #output_feats, output_score = self.multi_heads(out_list, sources)
        # Evaluation
        # fmt: off
        #if not self.training: return global_feat[..., 0, 0]
        #if not self.training: return global_feat[..., 0, 0]+global_feat_2[..., 0, 0]
        # if not self.training: return global_feat[..., 0, 0]

        # fmt: on

        # Training
        '''
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            cls_outputs = F.softmax(cls_outputs, dim=1)
            #print(cls_outputs.shape)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            cls_outputs = F.softmax(cls_outputs, dim=1)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                             F.normalize(self.classifier.weight))
        '''
        cls_outputs = score_1
        pred_class_logits = score_1.cpu().cuda()
        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
            "score_list":score_list
        }
