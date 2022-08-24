import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datahandlers.sampler import CustomBatchSampler

from typing import List
from copy import deepcopy

class CINIC10(torchvision.datasets.VisionDataset):
    class_map = {
        0: 'airplane',
        1:'automobile',
        2:'bird',
        3:'cat',
        4:'deer',
        5:'dog',
        6:'frog',
        7:'horse',
        8:'ship',
        9:'truck'
    }
    def __init__(self, root, subdataset, task, train=True, transform=None, target_transform=None):
        if train:
            flag = 'train'
        else:
            flag = 'test'
        class_0_data = np.load(root + '/cinic-10-{}/{}/{}/data.npy'.format(subdataset, flag, self.class_map[task[0]])) / 255.0
        class_1_data = np.load(root + '/cinic-10-{}/{}/{}/data.npy'.format(subdataset, flag, self.class_map[task[1]])) / 255.0
        self.data = torch.tensor(np.vstack((class_0_data, class_1_data)), dtype=torch.float)
        self.targets = list(np.concatenate((np.zeros(len(class_0_data)), np.ones(len(class_1_data)))).astype('int'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

class SplitCINIC10Handler:
    """
    Object for CINIC-10 Dataset
    """
    def __init__(self, cfg):
        self.cfg = cfg
        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.2, 0.25, 0.25]
        vanilla_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_norm, std=std_norm)])
        augment_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean_norm, std_norm)])

        if cfg.task.augment:
            train_transform = augment_transform
        else:
            train_transform = vanilla_transform

        tmap = cfg.task.task_map
        task = tmap[cfg.task.target]

        cifar_trainset = CINIC10('data/cinic10', 'cifar', task=task, train=True, transform=train_transform)
        imagenet_trainset = CINIC10('data/cinic10', 'imagenet', task=task, train=True, transform=train_transform)
        cifar_testset = CINIC10('data/cinic10', 'cifar', task=task, train=False, transform=vanilla_transform)

        self.trainset = deepcopy(cifar_trainset)
        self.testset = deepcopy(cifar_trainset)

        self.trainset.data = np.concatenate((cifar_trainset.data, imagenet_trainset.data))
        labels = [[0, lab] for lab in cifar_trainset.targets]
        imagenet_labels = [[1, lab] for lab in imagenet_trainset.targets]
        labels.extend(imagenet_labels)
        self.trainset.targets = labels

        self.testset.data = cifar_testset.data.numpy()
        labels = [[0, lab] for lab in cifar_testset.targets]
        self.testset.targets = labels

    def sample_data(self, seed):
        ## Balanced sample for each data

        cfg = self.cfg
        comb_trainset = deepcopy(self.trainset)
        data = self.trainset.data
        targets = np.array(self.trainset.targets)
        tasks = targets[:, 0]

        num_tasks = len(cfg.task.ood) + 1
        indices = []

        for i in range(num_tasks):
            idx = (np.where(tasks == i))[0]
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

            nsamples = cfg.task.n
            if i > 0:
                nsamples *= cfg.task.m_n
            nsamples = int(nsamples)

            if nsamples > 0:
                for lb in range(len(cfg.task.task_map[i])):
                    lab_idx = np.where(targets[idx, 1] == lb)[0]
                    indices.extend(list(idx[lab_idx][:nsamples]))

        comb_trainset.data = data[indices]
        comb_trainset.targets = targets[indices].tolist()
        self.comb_trainset = comb_trainset

    def get_data_loader(self, train=True):
        def wif(id):
            """
            Used to fix randomization bug for pytorch dataloader + numpy
            Code from https://github.com/pytorch/pytorch/issues/5059
            """
            process_seed = torch.initial_seed()
            # Back out the base_seed so we can use all the bits.
            base_seed = process_seed - id
            ss = np.random.SeedSequence([id, base_seed])
            # More than 128 bits (4 32-bit words) would be overkill.
            np.random.seed(ss.generate_state(4))

        cfg = self.cfg

        kwargs = {
            'worker_init_fn': wif,
            'pin_memory': True,
            'num_workers': 4,
            'multiprocessing_context':'fork'}

        if train:
            if cfg.task.custom_sampler and cfg.task.m_n > 0:
                tasks = np.array(self.comb_trainset.targets)[:, 0]
                batch_sampler = CustomBatchSampler(cfg, tasks)

                data_loader = DataLoader(
                    self.comb_trainset, batch_sampler=batch_sampler, **kwargs)
            else:
                # If no OOD samples use naive sampler
                data_loader = DataLoader(
                    self.comb_trainset, batch_size=cfg.hp.bs,
                    shuffle=True, **kwargs)
        else:
            data_loader = DataLoader(
                self.testset, batch_size=cfg.hp.bs, shuffle=False, **kwargs)
                

        return data_loader
