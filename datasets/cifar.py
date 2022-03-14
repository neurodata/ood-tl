import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.transform import rotate

from typing import List
from copy import deepcopy

class CIFAR10Handler:
    """
    Object for the CIFAR-10 dataset
    """
    def __init__(self, task, angle):
        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.2, 0.25, 0.25]
        vanilla_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_norm, std=std_norm)])

        trainset = torchvision.datasets.CIFAR10('data/cifar10', download=True, train=True, transform=vanilla_transform)
        testset = torchvision.datasets.CIFAR10('data/cifar10', download=True, train=False, transform=vanilla_transform)

        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        for lab_id, lab in enumerate(task):

            task_tr_ind = np.where(np.isin(trainset.targets,
                                            [lab % 10]))[0]
            task_te_ind = np.where(np.isin(testset.targets,
                                            [lab % 10]))[0]

            tr_ind.append(task_tr_ind)
            te_ind.append(task_te_ind)
            curlab = (lab_id)

            tr_vals = [curlab for _ in range(len(task_tr_ind))]
            te_vals = [curlab for _ in range(len(task_te_ind))]

            tr_lab.append(tr_vals)
            te_lab.append(te_vals)

        tr_ind, te_ind = np.concatenate(tr_ind), np.concatenate(te_ind)
        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        trainset.data = trainset.data[tr_ind]
        testset.data = testset.data[te_ind]

        trainset.targets = [it for it in tr_lab]
        testset.targets = [it for it in te_lab]

        # rotate the dataset
        rot_trainset = deepcopy(trainset)
        for i in range(len(trainset.data)):
          im = trainset.data[i]/255.0
          rot_trainset.data[i] = rotate(im, angle)*255

        self.trainset = trainset
        self.testset = testset
        self.rot_trainset = rot_trainset

    def sample_data(self, n, m):
        comb_trainset = deepcopy(self.trainset)
        data = self.trainset.data
        rot_data = self.rot_trainset.data
        targets = np.array(self.trainset.targets)
        class_0_indices = np.where(targets == 0)[0]
        class_1_indices = np.where(targets == 1)[0]

        n_indices = np.concatenate((class_0_indices[:int(n/2)], class_1_indices[:int(n/2)]))
        m_indices = np.concatenate((class_0_indices[int(n/2):int(n/2)+int(m/2)], class_1_indices[int(n/2):int(n/2)+int(m/2)]))

        comb_trainset.data = np.concatenate((data[n_indices], rot_data[m_indices]))
        comb_trainset.targets = np.concatenate((targets[n_indices], targets[m_indices])).tolist()

        self.comb_trainset = comb_trainset

    def get_data_loader(self, batch_size, train=True):
        if train:
            data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True)
        else:
            data_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        return data_loader
