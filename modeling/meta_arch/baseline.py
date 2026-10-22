# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)

        # head
        self.heads = build_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, tsne=False):
        images = self.preprocess_image(batched_inputs)
        sources = batched_inputs["sourceid"].to(self.device) if self.training else None
        
        features, score_list, kd_losses, tsne_feat, attn_maps = self.backbone(images, sources)
        if tsne:
            return tsne_feat, attn_maps
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)
            #print(targets)
            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, sources, targets)
            score_list = outputs["score_list"]
            losses = self.losses(outputs, targets, score_list, kd_losses)
            return losses
        else:
            outputs = self.heads(features)
            return outputs
            #return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels, score_list=None, kd_losses=None):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME
        '''
        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.0
        '''
        if isinstance(score_list,list):
            for i, score in enumerate(score_list):
                
                scale = 1.0
                if self._cfg.DATALOADER.SPECIFIC_WAY ==False: scale=0.0
                if i==0:
                    loss_dict["loss_cls"] = cross_entropy_loss(
                        score,
                        gt_labels,
                        self._cfg.MODEL.LOSSES.CE.EPSILON,
                        self._cfg.MODEL.LOSSES.CE.ALPHA,
                    ) * scale
                else:
                    loss_dict["loss_cls"] += cross_entropy_loss(
                        score,
                        gt_labels,
                        self._cfg.MODEL.LOSSES.CE.EPSILON,
                        self._cfg.MODEL.LOSSES.CE.ALPHA,
                    ) * scale
        if isinstance(kd_losses, torch.Tensor):
            loss_dict["loss_kd"] = kd_losses * 0.0
        else:
            loss_dict["loss_kd"] = loss_dict["loss_cls"] * 0.0
        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE 

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        if "Cosface" in loss_names:
            loss_dict["loss_cosface"] = pairwise_cosface(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.COSFACE.MARGIN,
                self._cfg.MODEL.LOSSES.COSFACE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.COSFACE.SCALE

        return loss_dict
