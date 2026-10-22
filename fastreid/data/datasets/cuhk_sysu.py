import glob
import re
import pdb
import os.path as osp
from .bases import ImageDataset
from .utils import load_pickle, get_pid_camid
from ..datasets import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class cuhkSYSU(ImageDataset):
    dataset_dir = 'CUHK-SYSU'
    dataset_name = "cuhksysu"
    def __init__(self, root='../', verbose=True, **kwargs):

        self.dataset_dir = osp.join(root, self.dataset_dir)
        pkl_path = osp.join(self.dataset_dir, 'partitions.pkl')
        #['trainval_im_names', 'trainval_ids2labels', 'val_marks', 'test_marks', 'name2tracks', 'train_ids2labels', 'val_im_names', 'train_im_names', 'test_im_names']
        partitions = load_pickle(pkl_path)
        train_im_names = partitions['train_im_names']
        test_im_names = partitions['test_im_names']
        test_marks = partitions['test_marks']
        
        train = self._process_train_dir(train_im_names, relabel=True)
        query = []
        gallery = []

        super(cuhkSYSU, self).__init__(train, query, gallery)

    def _process_train_dir(self, im_names, relabel=True):
        pid_container = set()
        for name in im_names:
            pid, camid = get_pid_camid(name)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for name in im_names:
            pid, camid = get_pid_camid(name) 
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_" + str(camid)
            img_path = osp.join(self.dataset_dir, 'images/'+name)
            source_id = 3
            dataset.append((img_path, pid, camid, source_id))
        return dataset
        
    def _process_test_dir(self, im_names, marks):
        assert len(im_names)==len(marks)
        query = []
        gallery = []
        for name, mark in zip(im_names,marks):
            pid, camid = get_pid_camid(name)
            img_path = osp.join(self.dataset_dir, 'images/'+name)
            if mark==0:
               query.append((img_path, pid, camid))
            elif mark==1:
               gallery.append((img_path, pid, camid))
            else:
               print('ERROR in VeRi-776 test set.')
               assert 0==1
        return query, gallery        