from copy import deepcopy
import torchvision
import numpy as np
from datahandlers.basehandler import DatasetHandler
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import ShuffleSplit
import deeplake


class DomainNet():
    def __init__(self, train=True):
        super().__init__()
        environments = ["clip", "info", "paint", "quick", "real", "sketch"]
        self.datasets = []
        for i, environment in enumerate(environments):
            if train:
                env_dataset = deeplake.load("hub://activeloop/domainnet-{}-{}".format(environment, "train"))
            else:
                env_dataset = deeplake.load("hub://activeloop/domainnet-{}-{}".format(environment, "test"))
            self.datasets.append(env_dataset)

class DomainNetDataset(torchvision.datasets.VisionDataset):
    def __init__(self, envs, classes, train=True, transform=None, target_transform=None):
        ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
        
        datasets = []
        for env in envs:
            if train:
                datasets.append(deeplake.load("hub://activeloop/domainnet-{}-{}".format(env, "train")))
            else:
                datasets.append(deeplake.load("hub://activeloop/domainnet-{}-{}".format(env, "test")))
    
        imgs = []
        labels = []
        for i, _ in enumerate(envs):
            task_labels = datasets[i].labels.numpy()
            labels.append(np.array([[i, int(lab)] for lab in task_labels]))
            imgs.append(np.array([[i, idx] for idx,_ in enumerate(task_labels)]))

        labels = np.concatenate(labels)
        imgs = np.concatenate(imgs)

        indices = []
        for idx, label in enumerate(labels):
            if label[1] in classes:
                indices.append(idx)
        
        imgs = imgs[indices]
        labels = labels[indices]
        cls_dict = dict(zip(classes, np.arange(0, len(classes)).tolist()))
        labels = [[lab[0], cls_dict[lab[1]]] for lab in labels]

        self.env_datasets = datasets
        self.data = imgs
        self.targets = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        task = self.data[idx][0]
        index = int(self.data[idx][1])
        image, target = self.env_datasets[task].images[index].numpy(), self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

class DomainNetHandler(DatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)

        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.20, 0.25, 0.25]
        vanilla_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_norm, std=std_norm)])
        augment_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean_norm, std_norm)])

        if cfg.task.augment:
            train_transform = augment_transform
        else:
            train_transform = vanilla_transform
        test_transform = vanilla_transform

        envs = [str(cfg.task.target_env), str(cfg.task.ood_env)]
        classes = cfg.task.task_map[0]
        self.trainset = DomainNetDataset(envs, classes, train=True, transform=train_transform)
        self.testset = DomainNetDataset([envs[0]], classes, train=False, transform=test_transform)

