# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import os.path as osp
import random

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import pickle as pickle

def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and 
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret
  
def get_pid_camid(name):
    if name[8]!='_' or name[13]!='_':
       return -1,-1
    else:
       pid = int(name[:8])
       camid = int(name[9:13]) 
       return pid, camid
  
@DATASET_REGISTRY.register()
class PRAI(ImageDataset):

    dataset_dir = "PRAI-1581"
    dataset_name = "PRAI-1581"

    def __init__(self, root='datasets', test_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        
        pkl_path = osp.join(self.dataset_dir, 'partitions.pkl')
        #['trainval_im_names', 'trainval_ids2labels', 'val_marks', 'test_marks', 'name2tracks', 'train_ids2labels', 'val_im_names', 'train_im_names', 'test_im_names']

        partitions = load_pickle(pkl_path)
        
        train_im_names = partitions['train_im_names']
        test_im_names = partitions['test_im_names']
        test_marks = partitions['test_marks']
        
        train = self._process_train_dir(train_im_names, relabel=True)
        query, gallery = self._process_test_dir(test_im_names, test_marks)
        
        #self.check_before_run(required_files)
        super(PRAI, self).__init__(train, query, gallery, **kwargs)

    def _process_train_dir(self, im_names, relabel=True):
        pid_container = set()
        for name in im_names:
            pid, camid = get_pid_camid(name)
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for name in im_names:
            pid, camid = get_pid_camid(name)   
            if pid == -1: continue  # junk images are just ignored
            #if relabel: pid = pid2label[pid]
            img_path = osp.join(self.dataset_dir, 'images_index_f/'+name)
            pid = self.dataset_name+'_'+str(pid)
            camid = self.dataset_name+'_'+str(camid)
            dataset.append((img_path, pid, camid))
        return dataset
        
    def _process_test_dir(self, im_names, marks):
        assert len(im_names)==len(marks)
        query = []
        gallery = []
        for name, mark in zip(im_names,marks):
            pid, camid = get_pid_camid(name)
            img_path = osp.join(self.dataset_dir, 'images_index_f/'+name)
            if mark==0:
               query.append((img_path, pid, camid))
            elif mark==1:
               gallery.append((img_path, pid, camid))
            else:
               print('ERROR in VeRi-776 test set.')
               assert 0==1
        return query, gallery  
