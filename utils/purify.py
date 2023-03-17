import sys
sys.path.insert(1, '/cis/home/adesilva/ashwin/research/ood-tl')

from datahandlers.pacs import PACSHandler

import numpy as np
import omegaconf
from imshowtools import imshow
from PIL import Image
from copy import deepcopy

from sklearn.metrics import pairwise_kernels, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast
import torch.cuda.amp as amp

import wandb
import contextlib
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
from itertools import product
import matplotlib.pyplot as plt


# Kernel two-sample test functions

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def compute_null_distribution_given_permutations(K, m, n, permutation,
                                                 iterations=None):
    """Compute the bootstrap null-distribution of MMD2u given
    predefined permutations.
    Note:: verbosity is removed to improve speed.
    """
    if iterations is None:
        iterations = len(permutation)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = permutation[i]
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=10000,
                           verbose=False, random_state=None, **kwargs):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return mmd2u, mmd2u_null, p_value


# get feature matrix from pretrained model
def get_feature_mat(model, loader, num_features, gpu=1):
    device = torch.device("cuda:{}".format(gpu))
    model.to(device)
    model.eval()
    feature_mat = torch.zeros((len(loader.dataset), num_features))
    class_labs = torch.zeros(len(loader.dataset))
    with torch.no_grad():
        for ims, targets, ids in loader:
            ims = ims.to(device)
            _ , labs = targets
            labs = labs.to(device)
            with autocast():
                out = model(ims)
                feature_mat[ids.squeeze(-1)] = out.squeeze().float().cpu()
                class_labs[ids.squeeze(-1)] = labs.squeeze().float().cpu()
    return feature_mat, class_labs


# define the iteration
def iteration(D_1, num_samples_per_iter, sig_level=0.05):
    idx = np.random.choice(len(D_1), 2 * num_samples_per_iter, replace=False)
    idxA = idx[:num_samples_per_iter]
    idxB = idx[num_samples_per_iter:]
    A = D_1[idxA]
    B = D_1[idxB]
    sigma2 = np.median(pairwise_distances(A, B, metric='euclidean'))**2
    t, _ , p_value =  kernel_two_sample_test(A, B, kernel_function='rbf', gamma=1.0/sigma2)
    weights = np.zeros((len(D_1), len(D_1)))
    prod = list(product(idxA, idxB))
    if p_value < sig_level:
        for x in prod:
            weights[x[0], x[1]] = t / num_samples_per_iter
    return weights


# purify
def purify_data(cfg, loader, featurizer, num_features):

    # Obtain the feature matrices
    feature_mat, _ = get_feature_mat(featurizer, loader, num_features)

    # compute the weights for each iteration
    f = lambda : iteration(feature_mat, cfg.purify.num_samples_per_iter)
    all_weights = np.array(Parallel(n_jobs=-1)(delayed(f)() for i in range(cfg.purify.max_iter)))

    mean_pairwise_weights = np.mean(all_weights, axis=0)

    clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='complete', compute_distances=True).fit(mean_pairwise_weights)
    cluster_assignments = clustering.labels_

    targets = np.array(loader.dataset.targets)
    true = targets[:, 0]
    true_m_n_ratio = len(np.where(true==1)[0])/len(np.where(true==0)[0])

    # select the appropriate cluster
    pred1 = cluster_assignments
    pred1_m_n_ratio = len(np.where(pred1==1)[0])/len(np.where(pred1==0)[0])
    pred2 = abs(cluster_assignments - 1)
    pred2_m_n_ratio = len(np.where(pred2==1)[0])/len(np.where(pred2==0)[0])

    if abs(true_m_n_ratio - pred1_m_n_ratio) < abs(true_m_n_ratio - pred2_m_n_ratio):
        cluster_assignments = pred1
    elif abs(true_m_n_ratio - pred1_m_n_ratio) > abs(true_m_n_ratio - pred2_m_n_ratio):
        cluster_assignments = pred2
    else:
        cluster_assignments = pred1

    targets[:, 0] = cluster_assignments
    new_loader = deepcopy(loader)
    new_loader.dataset.targets = targets.tolist()

    return new_loader, cluster_assignments


# train
def train(cfg, net, trainloader, wandb_log=True):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    fp16 = device != 'cpu'
    net.to(device)

    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=cfg.hp.lr,
        momentum=0.9, 
        nesterov=True,
        weight_decay=1e-5
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        cfg.hp.epochs * len(trainloader)
    )
    scaler = amp.GradScaler()
    ntasks = len(cfg.task.ood) + 1

    for epoch in range(cfg.hp.epochs):
        t_train_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        batches = 0.0
    
        criterion = nn.CrossEntropyLoss(reduction='none')
        net.train()

        for dat, target, ids in trainloader:
            tasks, labels = target
            tasks = tasks.long().to(device)
            labels = labels.long().to(device)
            dat = dat.to(device)
            batch_size = int(labels.size()[0])
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=fp16):
                out = net(dat)

                if cfg.loss.group_task_loss:
                    
                    loss = criterion(out, labels)
                    wt = cfg.loss.alpha
                    wo = (1-cfg.loss.alpha)
                    loss_target = torch.nan_to_num(loss[tasks==0].mean())
                    loss_ood = torch.nan_to_num(loss[tasks==1].mean())
                    final_loss = wt*loss_target + wo*loss_ood
                    
                    # task_oh = torch.nn.functional.one_hot(tasks, ntasks)
                    # task_count = task_oh.sum(0)

                    # loss = criterion(out, labels)
                    # task_loss = (loss.view(-1, 1) * task_oh).sum(0)

                    # mask = task_count != 0
                    # task_loss[mask] /= task_count[mask]

                    # task_loss[1:] *=  (1 - cfg.loss.alpha) / (ntasks - 1)
                    # task_loss[0] *= (cfg.loss.alpha)

                    # final_loss = task_loss.sum()

                else:
                    final_loss = criterion(out, labels).mean()

            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # Compute Train metrics
            batches += batch_size
            train_loss += final_loss.item() * batch_size

            # if cfg.loss.group_task_loss:
            #     tl = task_loss.detach().to('cpu').numpy()
            #     t_train_loss += tl * batch_size  # This is not exact

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            train_acc += np.sum(labels == (np.argmax(out, axis=1)))

    
        info = {
            "epoch": epoch + 1,
            "train_loss": np.round(train_loss/batches, 4),
            "train_acc": np.round(train_acc/batches, 4)
        }

        if cfg.deploy and wandb_log:
            wandb.log(info)

        # if cfg.loss.group_task_loss:
        #     info["task_loss"] = tuple(np.round(t_train_loss/batches, 4))
    
    return net


def evaluate(cfg, net, testloader, run_num):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    net.to(device)

    net.eval()
    acc = 0
    count = 0

    with torch.no_grad():
        for dat, target, ids in testloader:

            tasks, labels = target
            dat = dat.to(device)
            tasks = tasks.long().to(device)
            labels = labels.long().to(device)
            batch_size = int(labels.size()[0])

            out = net(dat)

            out = out.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc += np.sum(labels == (np.argmax(out, axis=1)))
            count += batch_size

    error = 1 - (acc/count)
    info = {"run_num": run_num,
            "final_test_err": error}

    if cfg.deploy:
        wandb.log(info)

    return error



