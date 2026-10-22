# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager

import torch

from fastreid.utils.logger import log_every_n_seconds


def extract_tsne_feature(model, data_loader):
    extracted_features = []
    attn_map_list = []
    source_ids = []
    pid_list = []
    im_pth_list = []
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            sources = inputs["sourceid"]
            pids = inputs["targets"]
            im_path = inputs["img_paths"]
            pid_list.extend(pids)
            im_pth_list.extend(im_path)
            outputs, attn_maps = model(inputs, tsne=True)
            extracted_features.append(outputs.cpu())
            attn_map_list.append(attn_maps.cpu())
            source_ids.append(sources.cpu())
    #extracted_features = torch.cat(extracted_features, dim=0)
    #source_ids = torch.cat(source_ids, dim=0)
    return extracted_features, attn_map_list, source_ids, pid_list, im_pth_list
    
    
    
    
    
    
    

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)