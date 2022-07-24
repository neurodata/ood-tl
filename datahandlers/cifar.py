import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.transform import rotate
import random
from datahandlers.sampler import CustomBatchSampler

from typing import List
from copy import deepcopy

class RotatedCIFAR10Handler:
    """
    Object for the CIFAR-10 dataset
    """
    def __init__(self, task, angle, augment):
        # separate the selected task-data from the main dataset
        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.2, 0.25, 0.25]
        augment_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean_norm, std_norm)])
        vanilla_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_norm, std=std_norm)])

        if augment:
            print("Using data augmentations...")
            train_transform = augment_transform
        else:
            print("No data augmentations...")
            train_transform = vanilla_transform

        trainset = torchvision.datasets.CIFAR10('data/cifar10', download=True, train=True, transform=train_transform)
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

        # Rotate the selected task-data by the specified angle
        rot_trainset = deepcopy(trainset)
        for i in range(len(trainset.data)):
          im = trainset.data[i]/255.0
          rot_trainset.data[i] = rotate(im, angle)*255

        rot_testset = deepcopy(testset)
        for i in range(len(testset.data)):
          im = testset.data[i]/255.0
          rot_testset.data[i] = rotate(im, angle)*255

        # Combined the selected task-data with rotated selected task-data and add (task_id, class label) as targets
        trainset.data = np.concatenate((trainset.data, rot_trainset.data))
        train_targets = []
        for i in range(2*len(trainset.targets)):
            if i < len(trainset.targets):
                train_targets.append([0, trainset.targets[i]])
            else:
                train_targets.append([1, trainset.targets[i-len(trainset.targets)]])
        trainset.targets = train_targets
        self.trainset = trainset
        
        testset.data = np.concatenate((testset.data, rot_testset.data))
        test_targets = []
        for i in range(2*len(testset.targets)):
            if i < len(testset.targets):
                test_targets.append([0, testset.targets[i]])
            else:
                test_targets.append([1, testset.targets[i-len(testset.targets)]])
        testset.targets = test_targets
        self.testset = testset

    def sample_data(self, n, m, randomly=False, SEED=1234):
        comb_trainset = deepcopy(self.trainset)
        data = self.trainset.data
        targets = np.array(self.trainset.targets)

        # encode the (task_id, class_lable) to an integer
        encoded_targets = np.dot(np.array(targets), 2**np.arange(1, -1, -1))

        # specify the target and OOD sample sizes in each class
        sample_sizes = [int(n/2), int(n/2), int(m/2), int(m/2)]
        
        indices = []
        random.seed(SEED)
        for i, sample_size in enumerate(sample_sizes):
            if randomly:
                # select the samples randomly
                indices.extend(random.sample(list(np.where(encoded_targets == i)[0]), sample_size))
            else:
                # select the samples non-randomly (incrementally)
                if i==2 or i==3:
                    indices.extend(np.where(encoded_targets == i)[0][int(n/2):int(n/2)+sample_size])
                else:
                    indices.extend(np.where(encoded_targets == i)[0][:sample_size])
        comb_trainset.data = data[indices]
        comb_trainset.targets = np.array(targets)[indices].tolist()

        self.comb_trainset = comb_trainset

    def get_data_loader(self, batch_size, train=True, isTaskAware=True):
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
        if train:
            if not isTaskAware:
                # Use a usual dataloader if task-agnostic setting
                data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4) # original
            else:
                # Use the dataloader if task-aware setting
                targets = self.comb_trainset.targets
                task_vector = torch.tensor([targets[i][0] for i in range(len(targets))], dtype=torch.int32)
                if task_vector.sum()==0:
                    # Use a usual dataloader if there're no OOD samples in the combined dataset
                    data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4) # original
                else:
                    # Use a custom batch-sampler there're OOD samples in the combined dataset
                    batch_sampler = torch.utils.data.BatchSampler(CustomBatchSampler(task_vector, batch_size), batch_size, True)
                    data_loader = DataLoader(self.comb_trainset, worker_init_fn=wif, pin_memory=True, num_workers=4, batch_sampler=batch_sampler)
                    # data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4) # original
        else:
            data_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, worker_init_fn=wif, pin_memory=True, num_workers=4)
        return data_loader

    def get_task_data_loader(self, task, batch_size, train=False):
        """
        Get Dataloader for a specific task
        """
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
        if train:
            task_set = deepcopy(self.trainset)
        else:
            task_set = deepcopy(self.testset)

        task_ind = [task == i[0] for i in task_set.targets]

        task_set.data = task_set.data[task_ind]
        task_set.targets = np.array(task_set.targets)[task_ind, :]
        task_set.targets = [(lab[0], lab[1]) for lab in task_set.targets]

        loader = DataLoader(
            task_set, batch_size=batch_size,
            shuffle=False, num_workers=6, pin_memory=True,
            worker_init_fn=wif)

        return loader


class SplitCIFARHandler:
    """
    Object for the CIFAR-10 dataset
    """
    def __init__(self, tasks, augment):
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

        if augment:
            print("Using data augmentations...")
            train_transform = augment_transform
        else:
            print("No data augmentations...")
            train_transform = vanilla_transform

        trainset = torchvision.datasets.CIFAR10('data/cifar10', download=True, train=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10('data/cifar10', download=True, train=False, transform=vanilla_transform)

        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []
        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(trainset.targets,
                                                [lab % 10]))[0]
                task_te_ind = np.where(np.isin(testset.targets,
                                                [lab % 10]))[0]

                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)
                curlab = (task_id, lab_id)

                tr_vals = [curlab for _ in range(len(task_tr_ind))]
                te_vals = [curlab for _ in range(len(task_te_ind))]

                tr_lab.append(tr_vals)
                te_lab.append(te_vals)

        tr_ind, te_ind = np.concatenate(tr_ind), np.concatenate(te_ind)
        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        trainset.data = trainset.data[tr_ind]
        testset.data = testset.data[te_ind]

        trainset.targets = [list(it) for it in tr_lab]
        testset.targets = [list(it) for it in te_lab]

        self.trainset = trainset
        self.testset = testset

    def sample_data(self, n, m, randomly=False, SEED=1234):
        comb_trainset = deepcopy(self.trainset)
        data = self.trainset.data
        targets = np.array(self.trainset.targets)

        encoded_targets = np.dot(np.array(targets), 2**np.arange(1, -1, -1))
        sample_sizes = [int(n/2), int(n/2), int(m/2), int(m/2)]
        
        indices = []
        random.seed(SEED)
        for i, sample_size in enumerate(sample_sizes):
            if randomly:
                indices.extend(random.sample(list(np.where(encoded_targets == i)[0]), sample_size))
            else:
                indices.extend(np.where(encoded_targets == i)[0][:sample_size])
        comb_trainset.data = data[indices]
        comb_trainset.targets = np.array(targets)[indices].tolist()

        self.comb_trainset = comb_trainset

    def get_data_loader(self, batch_size, train=True, isTaskAware=True, use_custom_sampler=True):
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
        if train:
            if not isTaskAware:
                # Use a usual dataloader if task-agnostic setting
                data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4) # original
            else:
                # Use the dataloader if task-aware setting
                targets = self.comb_trainset.targets
                task_vector = torch.tensor([targets[i][0] for i in range(len(targets))], dtype=torch.int32)
                if task_vector.sum()==0:
                    # Use a usual dataloader if there're no OOD samples in the combined dataset
                    data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4) # original
                else:
                    # Use a custom batch-sampler there're OOD samples in the combined dataset
                    if use_custom_sampler:
                        batch_sampler = torch.utils.data.BatchSampler(CustomBatchSampler(task_vector, batch_size), batch_size, True)
                        data_loader = DataLoader(self.comb_trainset, worker_init_fn=wif, pin_memory=True, num_workers=4, batch_sampler=batch_sampler)
                    else:
                        data_loader = DataLoader(self.comb_trainset, batch_size=batch_size, shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4) # original
        else:
            data_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, worker_init_fn=wif, pin_memory=True, num_workers=4)
        return data_loader

    def get_task_data_loader(self, task, batch_size, train=False):
        """
        Get Dataloader for a specific task
        """
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
        if train:
            task_set = deepcopy(self.trainset)
        else:
            task_set = deepcopy(self.testset)

        task_ind = [task == i[0] for i in task_set.targets]

        task_set.data = task_set.data[task_ind]
        task_set.targets = np.array(task_set.targets)[task_ind, :]
        task_set.targets = [(lab[0], lab[1]) for lab in task_set.targets]

        loader = DataLoader(
            task_set, batch_size=batch_size,
            shuffle=False, num_workers=6, pin_memory=True,
            worker_init_fn=wif)

        return loader

