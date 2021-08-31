import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import subprocess
import collections


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def flatten(d, root_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = root_key + sep + k if root_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)


def dict_to_str(d, sep='_'):
    return f"{sep}".join("{!s}={!r}".format(key, val)
                         for key, val in d.items())


def mod_dict(dict_to_modify, params):
    def recurse_dict(d, k, v):
        if (k in d):
            d[k] = v
            return d
        for kk, vv in d.items():
            if (type(vv) == collections.OrderedDict or type(vv) == dict):
                d[kk] = recurse_dict(vv, k, v)
        return d
    for k, v in params.items():
        if k in dict_to_modify:
            dict_to_modify[k] = v
            continue
        for kk, vv in dict_to_modify.items():
            if (type(vv) == collections.OrderedDict or type(vv) == dict):
                dict_to_modify[kk] = recurse_dict(vv, k, v)
    return dict_to_modify


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def get_gpu_memory_map():
    import torch
    """Get the current gpu usage.
    Adopted from https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    for idx in range(torch.cuda.device_count()):
        # doesn't distinguish between nvidia-smi idx and torch cuda idxs..
        mem = torch.cuda.get_device_properties(idx).total_memory / 1024.0 ** 2
        gpu_memory_map[idx] /= mem
    return gpu_memory_map

def import_module(base_name, config_name, config):
    """
    dynamic import
    """
    return getattr(__import__('{}.{}'.format(base_name, config[config_name]['module_name'])), config[config_name]['type'])
