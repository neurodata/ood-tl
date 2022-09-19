import torch
import numpy as np
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image


def get_cinic_imagenet():
    cinic_mean = [0.50, 0.50, 0.50]
    cinic_std = [0.25, 0.25, 0.25]

    cinic_train = torchvision.datasets.ImageFolder(
            './data/cinic-10/train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

    import ipdb;ipdb.set_trace()

    labcount = np.zeros(10)
    xdat = [[] for _ in range(10)]

    for img, lab in cinic_train.imgs:

        if "cifar10" in img:
            continue

        if labcount[lab] >= 7500:
            continue

        img_mat = Image.open(img)
        img_mat = img_mat.convert('RGB')
        img_mat = np.array(img_mat)

        xdat[lab].append(img_mat)
        labcount[lab] += 1

    xdat = np.array(xdat)
    print(xdat.shape)

    np.save("./data/cinic10_img.npy", xdat)



get_cinic_imagenet()
