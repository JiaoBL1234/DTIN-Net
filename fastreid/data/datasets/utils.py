import os,sys,pdb
import os.path as osp
import pickle as pickle
import numpy as np
def get_pid_camid(name):
    if name[8]!='_' or name[13]!='_':
       return -1,-1
    else:
       pid = int(name[:8])
       camid = int(name[9:13]) 
       return pid, camid

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
import json
def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    #mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))