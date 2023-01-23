import hydra
import wandb

import torch
import torch.nn as nn
import torchvision

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

from utils.init import set_seed, open_log, init_wandb, cleanup, update_config
from utils.purify import purify_data, train, evaluate

from datahandlers.pacs import PACSHandler

def get_data(cfg, seed):
    dataHandler = PACSHandler(cfg)
    dataHandler.sample_data(seed)
    trainloader = dataHandler.get_data_loader(train=True)
    testloader = dataHandler.get_data_loader(train=False)
    return trainloader, testloader


def get_net(cfg):
    # Obtain the featurizer from the pre-trained ResNet-18 model
    ptmodel = torchvision.models.resnet18(pretrained=True)
    num_features = ptmodel.fc.in_features
    for param in ptmodel.parameters():
        param.requires_grad = False
    ptmodel.fc = nn.Linear(num_features, len(cfg.task.task_map[0]))
    return ptmodel


def get_featurizer():
    ptmodel = torchvision.models.resnet18(pretrained=True)
    num_features = ptmodel.fc.in_features
    for param in ptmodel.parameters():
        param.requires_grad = False
    featurizer = nn.Sequential(*list(ptmodel.children())[:-1])
    return featurizer, num_features


def get_metrics(result):
    """Return recall, precision, false positive rate, false negative rate, f1-score, and accuracy"""
    tn, fp, fn, tp = result 
    recall = tp / (tp + fn)
    precision = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return [recall, precision, fpr, fnr, f1, acc]

def run(cfg, trainloader, testloader, rnum, case):
    update_config(cfg)
    net = get_net(cfg)
    net = train(cfg, net, trainloader)
    err = evaluate(cfg, net, testloader, rnum)
    info = {"{}_err".format(case) : round(err, 4)}
    if cfg.deploy:
        wandb.log(info)
    return err

@hydra.main(config_path="./config", config_name="purify.yaml")
def main(cfg):
    init_wandb(cfg, project_name="data_purification", entity_name="ashwin1996")
    fp = open_log(cfg)

    errs = []
    metrics = []
    for rnum in range(cfg.reps):
        sub_errs = []

        if cfg.random_reps:
            seed =  cfg.seed + rnum * 10
        else:
            seed = cfg.seed
        set_seed(seed)

        trainloader, testloader = get_data(cfg, seed)
        true_task_labels = np.array(trainloader.dataset.targets)[:, 0]
        featurizer, num_features = get_featurizer()

        cfg.purify.num_samples_per_iter = int(cfg.task.n * cfg.task.m_n * len(cfg.task.task_map[0]) + 2)
        purified_trainloader, pred_task_labels = purify_data(cfg, trainloader, featurizer, num_features)

        # case 1 : no purification (target/OOD samples unknown), no weighting (alpha = n/(n+m))
        cfg.loss.group_task_loss = False
        err = run(cfg, trainloader, testloader, rnum, "case1")
        sub_errs.append(err)

        # case 2 : target/OOD samples known, throw away all OOD samples (alpha = 1)
        cfg.loss.group_task_loss = True
        cfg.loss.alpha = 1
        err = run(cfg, trainloader, testloader, rnum, "case2")
        sub_errs.append(err)

        # case 3 : target/OOD samples known, optimal weighting
        cfg.loss.group_task_loss = True
        cfg.loss.alpha = 0.95
        err = run(cfg, trainloader, testloader, rnum, "case3")
        sub_errs.append(err)

        # case 4 : purification, throw away all OOD samples (alpha = 1)
        cfg.loss.group_task_loss = True
        cfg.loss.alpha = 1
        err = run(cfg, purified_trainloader, testloader, rnum, "case4")
        sub_errs.append(err)

        # case 5 : purification, optimal weighting
        cfg.loss.group_task_loss = True
        cfg.loss.alpha = 0.75
        err = run(cfg, purified_trainloader, testloader, rnum, "case5")
        sub_errs.append(err)
        
        metric = get_metrics(confusion_matrix(true_task_labels, pred_task_labels).ravel())
        info = {
            "recall": metric[0], 
            "precision": metric[1], 
            "fpr": metric[2], 
            "fnr": metric[3], 
            "f1": metric[4], 
            "acc": metric[5], 

        }
        if cfg.deploy:
            wandb.log(info)

        errs.append(sub_errs)
        metrics.append(metric)

    info = {
        "errs": errs,
        "metrics": metrics,
        "avg_err": np.mean(errs, axis=0),
        "std_err": np.std(errs, axis=0),
        "avg_mets": np.mean(metrics, axis=0),
        "std_mets": np.std(metrics, axis=0)                                       
    }
    
    if cfg.deploy:
        wandb.log(info)

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()