# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)






class Mask_Module(nn.Module):
    def __init__(self, inplanes=2048, heads=5, with_mask=False, max_pool=False):
        super(Mask_Module, self ).__init__()
        self.feat_dim = inplanes
        self.embedding_dim = int(inplanes/4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.heads = heads
        self.max_pool = max_pool
        self.Masker = nn.ModuleDict() 
        for step in range(self.heads):
            self.Masker[f'step:{step}'] = nn.Sequential(nn.Linear(self.feat_dim, self.embedding_dim),
                                                                 nn.BatchNorm1d(self.embedding_dim),
                                                                 nn.ReLU(inplace=True),
                                                                 nn.Linear(self.embedding_dim, self.feat_dim),
                                                                 nn.Sigmoid()
                                                                )
        self.with_mask = with_mask
    def forward(self, input):
        
        N, C, H, W = input.shape
        output_list = []
        for c_s in range(self.heads):
            #output = self.conv2(feat)
            if self.max_pool:
                feat = self.gmp(input).view(N, C)
            else:
                feat = self.gap(input).view(N, C)
            mask = self.Masker[f'step:{c_s}'](feat)
            if self.with_mask:
                masked_feat = feat*mask
                output_list.append(masked_feat)
            else:
                output_list.append(feat)
        return output_list
        
        

class Multi_Heads(nn.Module):
    def __init__(self, num_list, inplanes=2048, with_mask=False, max_pool=False):
        super(Multi_Heads, self ).__init__() 

        self.num_list = num_list
        self.down_sample = nn.ModuleDict()
        self.feat_dim = inplanes
        self.embedding_dim = int(inplanes/4)
        self.heads = 5
        self.max_pool = max_pool
        self.gmp = nn.AdaptiveMaxPool2d(1)  
        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.mask_module = Mask_Module(inplanes=self.feat_dim, heads=self.heads, with_mask=with_mask, max_pool=max_pool)


        
        
        self.classifier_dict = nn.ModuleDict()
        for step, num in enumerate(num_list):
            self.classifier_dict[f'step:{step}'] = nn.Sequential(nn.BatchNorm1d(self.feat_dim),
                                                                    nn.Linear(self.feat_dim, num, bias=False))
            self.classifier_dict[f'step:{step}'][0].bias.requires_grad_(False)
            self.classifier_dict[f'step:{step}'][0].apply(weights_init_kaiming)
            self.classifier_dict[f'step:{step}'][1].apply(weights_init_classifier)
        

        
       
    def forward(self, input, source=None):
    
        N, C, H, W = input.shape
        feat_list = self.mask_module(input)
        cls_score_list = []
        for c_s in range(self.heads):
            #cls_score = self.classifier_dict[f'step:{c_s}'](feat)
            feat_ = feat_list[c_s]
            cls_score = self.classifier_dict[f'step:{c_s}'](feat_)
            cls_score = F.softmax(cls_score, dim=1)
            cls_score_list.append(cls_score)
        output_score = torch.cat(cls_score_list, dim=1) 
        return output_score
       






















       
        
class Multi_Heads_(nn.Module):
    def __init__(self, num_list, inplanes=2048):
        super(Multi_Heads_, self ).__init__() 

        self.num_list = num_list
        self.down_sample = nn.ModuleDict()
        self.feat_dim = inplanes
        self.embedding_dim = int(inplanes/4)
        self.heads = 5

        self.gmp = nn.AdaptiveMaxPool2d(1)  
        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.mask_module = Mask_Module(inplanes=self.feat_dim, heads=self.heads)


        
        
        self.classifier_dict = nn.ModuleDict()
        for step, num in enumerate(num_list):
            self.classifier_dict[f'step:{step}'] = nn.Sequential(nn.BatchNorm1d(self.feat_dim),
                                                                    nn.Linear(self.feat_dim, num, bias=False))
            self.classifier_dict[f'step:{step}'][0].bias.requires_grad_(False)
            self.classifier_dict[f'step:{step}'][0].apply(weights_init_kaiming)
            self.classifier_dict[f'step:{step}'][1].apply(weights_init_classifier)
        

        
       
    def forward(self, input, source=None):
    
        N, C, H, W = input.shape
        pool_feat = self.gap(input).view(N, C)
        feat_list = [pool_feat, pool_feat, pool_feat, pool_feat, pool_feat]
        cls_score_list = []
        ori_score_list = []
        bn_feat_list = []
        for c_s in range(self.heads):
            #cls_score = self.classifier_dict[f'step:{c_s}'](feat)
            feat_ = feat_list[c_s]
            cls_score = self.classifier_dict[f'step:{c_s}'](feat_)
            ori_score_list.append(cls_score)
            #cls_score = F.softmax(cls_score, dim=1)
            cls_score_list.append(cls_score)

        output_score = torch.cat(cls_score_list, dim=1) 
        ori_score = torch.cat(ori_score_list, dim=1)
        return output_score, ori_score
        
        
        
        
        
        