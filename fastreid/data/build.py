# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os

import torch
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader
import numpy as np
from fastreid.utils import comm
from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms
from collections import defaultdict
__all__ = [
    "build_reid_train_loader",
    "build_reid_test_loader"
]

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader(cfg, mapper=None, **kwargs):
    """
    Build reid train loader

    Args:
        cfg : image file path
        mapper : one of the supported image modes in PIL, or "BGR"

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """
    cfg = cfg.clone()

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL, **kwargs)
        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=True)

    train_set = CommDataset(train_items, transforms, relabel=True, training=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()
    same_running = cfg.DATALOADER.SAME_RUNNING
    if cfg.DATALOADER.PK_SAMPLER:
        if cfg.DATALOADER.SPECIFIC_WAY:
            data_sampler = samplers.Source_Specific_Sampler(train_set.img_items, mini_batch_size, num_instance, same_running=same_running)
        elif cfg.DATALOADER.NAIVE_WAY:
            data_sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
        else:
            data_sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader
    
    
def instance_sampler(s_num, dataset, tsne_show):
    sampled_list = list()
    pid_list = list()
    selected_cam = {}
    for tup in dataset.train:
        img_path, pid, camid, source_id = tup
        pid_list.append(pid)
        if pid not in selected_cam.keys():
            selected_cam[pid]=camid
        
    pid_set = set(pid_list)
    selected_pids = np.random.choice(list(pid_set), size=s_num, replace=False)
    selected_pids = comm.all_gather(selected_pids)[0]
    selected_ids_dict = defaultdict(list)
    for tup in dataset.train:
        pid = tup[1]
        camid = tup[2]
        if tsne_show:
            if pid in selected_pids and camid==selected_cam[pid]:
                if len(selected_ids_dict[pid]) > 10:
                    continue
                sampled_list.append(tup)
                selected_ids_dict[pid].append(tup)
        else:
            if pid in selected_pids:
                if np.random.random()<0.5 and camid==selected_cam[pid]:
                    continue
                if len(selected_ids_dict[pid]) > 10:
                    continue
                sampled_list.append(tup)
                selected_ids_dict[pid].append(tup)
    return sampled_list
    
def build_reid_tsne_loader(cfg, instance_num = 100, **kwargs):
    cfg = cfg.clone()

    tsne_items = list()
    for i, d in enumerate(cfg.DATASETS.NAMES):
        #if i not in (0,2):
        #    continue
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL, **kwargs)
        #if comm.is_main_process():
        #    dataset.show_train()
        selected_tups = instance_sampler(5, dataset, cfg.TEST.TSNE_SHOW)
        tsne_items.extend(selected_tups) 
    transforms = build_transforms(cfg, is_train=False)
    tsne_set = CommDataset(tsne_items, transforms, relabel=False, training=True)
    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(tsne_set))
    
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    tsne_loader = DataLoader(
        tsne_set,
        batch_sampler=batch_sampler,
        num_workers=4,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return tsne_loader

def build_reid_test_loader(cfg, dataset_name, mapper=None, **kwargs):
    """
    Build reid test loader

    Args:
        cfg:
        dataset_name:
        mapper:
        **kwargs:

    Returns:

    """

    cfg = cfg.clone()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    
    
    
    if dataset_name in['GRID', 'PRID', 'VIPeR', 'iLIDS']:
        loader_list = []
        for id in range(10):
            dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, split_id=id, **kwargs)
            if comm.is_main_process():
                dataset.show_test()
            test_items = dataset.query + dataset.gallery

            if mapper is not None:
                transforms = mapper
            else:
                transforms = build_transforms(cfg, is_train=False)

            test_set = CommDataset(test_items, transforms, relabel=False, training=False)

            mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
            data_sampler = samplers.InferenceSampler(len(test_set))
            batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
            test_loader = DataLoader(
                test_set,
                batch_sampler=batch_sampler,
                num_workers=4,  # save some memory
                collate_fn=fast_batch_collator,
                pin_memory=True,
            )
            loader_list.append(test_loader)
        return loader_list, len(dataset.query)
    
    
    
    
    
    
    
    
    
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query + dataset.gallery

    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=False)

    test_set = CommDataset(test_items, transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=4,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
