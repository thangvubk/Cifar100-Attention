# Author: Thang Vu
# Date: 19/Apr/2017
# Description: Load datasets

import gzip
from six.moves import cPickle as pickle
import os
import platform


# load pickle based on python version 2 or 3
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    else:
        raise ValueError("invalid python version: {}".format(version))

def load_cifar100_dataset(path):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' %path)
    with open(path, 'rb') as f:
        dataset = load_pickle(f)
        inputs = dataset['data']
        labels = dataset['fine_labels']
        return inputs, labels
