# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True, training=False):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.training = training
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict()
            counter = [0, 0, 0, 0, 0]
            for i, p in enumerate(self.pids):
                if 'market' in p:
                    self.pid_dict[p] = counter[0]
                    counter[0] += 1
                elif 'duke' in p:
                    self.pid_dict[p] = counter[1] + 1501
                    counter[1] += 1
                elif 'cuhk03' in p:
                    self.pid_dict[p] = counter[2] + 3313
                    counter[2] += 1
                elif 'cuhksysu' in p:
                    self.pid_dict[p] = counter[3] + 4780
                    counter[3] += 1   
                elif 'cuhk02' in p:
                    self.pid_dict[p] = counter[4] + 4780 + 11934
                    counter[4] += 1
                else:
                    raise 'not avaiable dataset name'
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        if self.training:
            sourceid = img_item[3]
        else:
            sourceid = 5
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]

        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "sourceid": sourceid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
