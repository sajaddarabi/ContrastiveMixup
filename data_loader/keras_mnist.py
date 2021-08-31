import numpy as np
import torch
from keras.datasets import mnist
from torch.utils.data import Dataset

METHODS = ['', 'supervised', 'semisupervised', 'pseudolabeling']

class KerasMNIST(Dataset):
    """ Implements keras MNIST torch.utils.data.Dataset


        Args:
            train (bool): flag to determine if train set or test set should be loaded
            labeled_ratio (float): fraction of train set to use as labeled
            method (str): valid methods are in `METHODS`.
                'supervised': getitem return x_labeled, y_label
                'semisupervised': getitem will return x_labeled, x_unlabeled, y_label
                'pseudolabeling': getitem will return x_labeled, x_unlabeled, y_label, y_pseudolabel
            random_seed (int): change random_seed to obtain different splits otherwise the fixed random_seed will be used

    """
    def __init__(self,
                 train: bool,
                 labeled_ratio: float=0.0,
                 method: str='supervised',
                 random_seed: int=None):
        super().__init__()
        print('Dataloader __getitem__ mode: {}'.format(method))
        assert method.lower() in METHODS , 'Method argument is invalid {}, must be in'.format(METHODS)
        if (train):
            (self.data, self.targets), (_, _) = mnist.load_data()
        else:
            (_, _), (self.data, self.targets) = mnist.load_data()

        self.data = self.data / 255.0
        self.train = train
        self.data = np.reshape(self.data, (len(self.targets), -1)).astype(np.float32)
        self.method = method.lower()
        idx = np.arange(len(self.targets))
        self.labeled_idx = idx
        self.unlabeled_idx = idx
        self.idx = idx
        if (labeled_ratio > 0):
            if random_seed is not None:
                idx = np.random.RandomState(seed=random_seed).permutation(len(self.targets))
            else:
                idx = np.random.permutation(len(self.targets))
            self.idx = idx
            if labeled_ratio <= 1.0:
                ns = labeled_ratio * len(self.idx)
            else:
                ns = labeled_ratio
            ns = int(ns)
            labeled_idx = self.idx[:ns]
            unlabeled_idx = self.idx[ns:]
            self.labeled_idx =  labeled_idx
            self.unlabeled_idx = unlabeled_idx

        self._pseudo_labels = list()
        self._pseudo_labels_weights = list()

    def get_pseudo_labels(self):
        return self._pseudo_labels

    def set_pseudo_labels(self, pseudo_labels):
        self._pseudo_labels = pseudo_labels

    def set_pseudo_labels_weights(self, pseudo_labels_weights):
        self._pseudo_labels_weights = pseudo_labels_weights

    def __len__(self):
        if (self.method == 'pseudolabeling'):
            return len(self.idx)
        return len(self.labeled_idx)

    def _semisupervised__getitem__(self, idx):
        idx = self.labeled_idx[idx]
        uidx = np.random.randint(0, len(self.unlabeled_idx))
        uidx = self.unlabeled_idx[uidx]
        img, target = self.data[idx], int(self.targets[idx])
        uimg = self.data[uidx]
        if len(self._pseudo_labels):
            utarget = self._pseudo_labels[uidx]
            uweight = self._pseudo_labels_weights[uidx]
            return img, uimg, target, utarget, uweight
        return img, uimg, target

    def _normal__getitem__(self, idx):
        idx = self.labeled_idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        return img, target

    def _pseudolabeling__getitem__(self, idx):
        idx = self.idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        labeled_mask = np.array([False], dtype=np.bool)
        if idx in self.labeled_idx:
            labeled_mask[0] = True
        idx = np.asarray([idx])
        return img, target, labeled_mask, idx

    def __getitem__(self, idx):
        if self.method == 'semisupervised' and self.train:
            return self._semisupervised__getitem__(idx)
        if self.method == 'pseudolabeling' and self.train:
            return self._pseudolabeling__getitem__(idx)
        else:
            return self._normal__getitem__(idx)
