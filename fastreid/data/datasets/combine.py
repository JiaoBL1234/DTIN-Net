import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMC
from .cuhk03 import CUHK03
from .cuhk02 import CUHK02
from .cuhk_sysu import cuhkSYSU
from .grid import GRID
@DATASET_REGISTRY.register()
class Combine(ImageDataset):
    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        self.market = Market1501(root)
        self.duke = DukeMTMC(root)
        self.cuhk03 = CUHK03(root, cuhk03_labeled=False)
        self.cuhk02 = CUHK02(root)
        self.cuhk_sysu = cuhkSYSU(root)
        self.grid = GRID(root)
        train = self.market.train + self.market.query + self.market.gallery +\
                self.duke.train + self.duke.gallery + self.duke.query +\
                self.cuhk03.train + self.cuhk03.query + self.cuhk03.gallery +\
                self.cuhk02.train+self.cuhk_sysu.train
        query = self.grid.query
        gallery = self.grid.gallery
        #query = self.market.query+self.duke.query+self.cuhk03.query+self.cuhk02.query+self.cuhk_sysu.query
        #gallery = self.market.gallery+self.duke.gallery+self.cuhk03.gallery+self.cuhk02.gallery+self.cuhk_sysu.gallery
        super(Combine, self).__init__(train, query, gallery, **kwargs)
        