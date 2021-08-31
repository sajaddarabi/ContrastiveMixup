from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.keras_mnist import KerasMNIST

import copy
import torch
import numpy as np

class KerasMNISTDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, **kwargs):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.training = training
        self.validation_split = validation_split
        self.kwargs = kwargs
        self.dataset = KerasMNIST(train=training, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_pseudolabeling_loader(self):
        if hasattr(self, 'pseudolabeling_loader'):
            return self.pseudolabeling_loader

        dataset = copy.copy(self.dataset)
        dataset.method = 'pseudolabeling'
        self.pseudolabeling_loader = BaseDataLoader(dataset, self.batch_size, shuffle=True,  validation_split=0.0, num_workers=self.num_workers)

        return self.pseudolabeling_loader

    def update_pseudo_labels(self, model, pseudolabeler, device=None):

        model.eval()
        data_loader = self.get_pseudolabeling_loader()

        latents = list()
        labels = list()
        labels_mask = list()
        idxs = list()

        for i, data in enumerate(data_loader):
            x, label, labeled_mask, idx = data
            with torch.no_grad():
                output = model(x, device)
            latents.append(output['z'].detach().cpu())
            labels.append(label)
            labels_mask.append(labeled_mask)
            idxs.append(idx)

        latents = torch.cat(latents, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        labels_mask = torch.cat(labels_mask, dim=0).numpy()
        idxs = torch.cat(idxs, dim=0).numpy()
        ordered_idxs = np.arange(0, len(latents)).reshape(-1, 1)
        input_data = (latents, labels, labels_mask, ordered_idxs)
        pseudo_labels, acc = pseudolabeler(*input_data)
        idxs = idxs.squeeze()
        labels_ = np.zeros(len(data_loader.dataset.data))
        labels_[idxs] = pseudo_labels
        weights = np.zeros(len(pseudo_labels))
        weights[idxs] = pseudolabeler.p_weights
        self.dataset.set_pseudo_labels(labels_)
        self.dataset.set_pseudo_labels_weights(weights)

        return acc
