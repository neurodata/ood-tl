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



class CINICHandler:
    """
    Object for the CINIC-10 dataset
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

        # CIFAR10 dataset
        if cfg.task.augment:
            train_transform = augment_transform
        else:
            train_transform = vanilla_transform

        trainset = torchvision.datasets.CIFAR10('data/cifar10', download=True,
                                                train=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10('data/cifar10', download=True,
                                               train=False, transform=vanilla_transform)

        # Load cinic dataset
        cinic_dat = np.load("./data/cifar10_neg/CIFAR10_neg.npz")
        x_cinic = cinic_dat['data']
        y_cinic = cinic_dat['labels']

        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []
        # Right now, we get all labels
        for lab in range(10):
            curlab = (0, lab)

            task_tr_ind = np.where(np.isin(trainset.targets,
                                           [lab % 10]))[0]
            tr_ind.append(task_tr_ind)
            tr_vals = [curlab for _ in range(len(task_tr_ind))]
            tr_lab.append(tr_vals)

            task_te_ind = np.where(np.isin(testset.targets,
                                           [lab % 10]))[0]
            te_ind.append(task_te_ind)
            te_vals = [curlab for _ in range(len(task_te_ind))]
            te_lab.append(te_vals)

        tr_ind, te_ind = np.concatenate(tr_ind), np.concatenate(te_ind)
        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        trainset.data = trainset.data[tr_ind]

        # Testset
        testset.data = testset.data[te_ind]
        testset.targets = [list(it) for it in te_lab]
        self.testset = testset

        ctr_ind, ctr_lab = [], []
        for lab in range(10):
            curlab = (1, lab)
            task_tr_ind = np.where(np.isin(y_cinic,
                                           [lab % 10]))[0]
            ctr_ind.append(task_tr_ind)
            ctr_vals = [curlab for _ in range(len(task_tr_ind))]
            ctr_lab.append(ctr_vals)

        ctr_ind, ctr_lab = np.concatenate(ctr_ind), np.concatenate(ctr_lab)

        ctr_dat = x_cinic[ctr_ind]
        
        # Merge CINIC dataset
        trainset.data = np.concatenate([trainset.data, ctr_dat], axis=0)
        trainset.targets = np.concatenate([tr_lab, ctr_lab], axis=0)
        trainset.targets = [list(it) for it in trainset.targets]

        self.trainset = trainset

    def sample_data(self, seed):
        ## Balanced sample for each data

        cfg = self.cfg
        comb_trainset = deepcopy(self.trainset)
        data = self.trainset.data
        targets = np.array(self.trainset.targets)
        tasks = targets[:, 0]

        indices = []

        for i in range(2):
            idx = (np.where(tasks == i))[0]
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

            nsamples = cfg.task.n
            if i > 0:
                nsamples *= cfg.task.m_n
            nsamples = int(nsamples)

            if nsamples > 0:
                for lb in range(10):
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
