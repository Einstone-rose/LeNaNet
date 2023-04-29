import os
import errno
import numpy as np
from PIL import Image
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import utils.config as config


def path_for(train=False, val=False, test=False, question=False, answer=False, knowledge=False):
    assert train + val + test == 1
    assert question + answer + knowledge == 1

    if train:
        split = 'train' if config.cp_data else 'train2014'
    elif val:
        split = 'test' if config.cp_data else 'val2014'
    else:
        split = config.test_split

    if question:
        fmt = '{0}_{1}_{2}_questions.json'
    elif knowledge:
        fmt = '{1}_{2}_facts.json'
    else:
        if test:
            # will be ignored anyway
            split = 'val2014'
        fmt = '{0}_{1}_{2}_annotations.json' if \
                config.cp_data else '{1}_{2}_annotations.json'

    if config.version == 'v2' and not config.cp_data:
        fmt = 'v2_' + fmt
    if config.cp_data and not knowledge:
        s = fmt.format(config.task, config.version, split)
    else:
        s = fmt.format(config.task, config.image_dataset, split)

    if not knowledge:
        return os.path.join(config.qa_path, s)
    else:
        return os.path.join(config.knowledge_path, s)


def assert_eq(real, expected):
    assert real == expected, '{} (true) vs {} (expected)'.format(real, expected)


def assert_array_eq(real, expected):
    EPS = 1e-7
    assert (np.abs(real-expected) < EPS).all(), \
        '{} (true) vs {} (expected)'.format(real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Tracker:
    """ Keep track of results over time, while having access to
        monitors to display information about them.
    """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
