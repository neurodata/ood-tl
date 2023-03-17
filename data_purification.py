import hydra
import wandb

import torch
import torch.nn as nn
import torchvision

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from copy import deepcopy

from utils.init import set_seed, open_log, init_wandb, cleanup, update_config
from utils.purify import purify_data, train, evaluate

from datahandlers.pacs import PACSHandler

def get_data(cfg, seed):
    cfg.task.custom_sampler = False
    dataHandler = PACSHandler(cfg)
    dataHandler.sample_data(seed)
    trainloader_generic = dataHandler.get_data_loader(train=True)
    testloader = dataHandler.get_data_loader(train=False)

    cfg.task.custom_sampler = True
    dataHandler = PACSHandler(cfg)
    dataHandler.sample_data(seed)
    trainloader_custom = dataHandler.get_data_loader(train=True)
    return trainloader_generic, trainloader_custom, testloader


def get_net(cfg):
    # Obtain the featurizer from the pre-trained ResNet-18 model
    ptmodel = torchvision.models.resnet18(pretrained=True)
    num_features = ptmodel.fc.in_features
    for param in ptmodel.parameters():
        param.requires_grad = False
    ptmodel.fc = nn.Linear(num_features, len(cfg.task.task_map[0]))
    return ptmodel


def run(cfg, trainloader, testloader, rnum, case):
    update_config(cfg)
    net = get_net(cfg)
    net = train(cfg, net, trainloader)
    err = evaluate(cfg, net, testloader, rnum)
    if cfg.deploy:
        wandb.log({"{}_err".format(case) : round(err, 4)})
    return err


def search_alpha(cfg, trainloader, testloader):
    alpha_range = [0.55, 0.65, 0.75, 0.85, 0.95]
    scores = []
    for alpha in alpha_range:
        cfg.loss.alpha = np.float64(alpha).item()
        tune_net = get_net(cfg) # deepcopy the network architecture
        tune_net = train(cfg, tune_net, trainloader)
        err =  evaluate(cfg, tune_net, testloader, 0)
        info = {
            "current_tuning_alpha": round(alpha, 4),
            "error_at_alpha": round(err, 4)
        }
        if cfg.deploy:
            wandb.log(info)
        scores.append(err)
    return alpha_range[np.argmin(scores)], scores[np.argmin(scores)]


@hydra.main(config_path="./config", config_name="purify.yaml")
def main(cfg):
    init_wandb(cfg, project_name="data_purification", entity_name="ashwin1996")
    fp = open_log(cfg)

    errs = []
    for rnum in range(cfg.reps):
        sub_errs = []

        if cfg.random_reps:
            seed =  cfg.seed + rnum * 10
        else:
            seed = cfg.seed
        set_seed(seed)

        m_n = cfg.task.m_n
        array = np.load('notebooks/predicted_labels.npy', allow_pickle=True)
        predicted_task_labels = array[array[:, 0]==m_n][rnum][-1]

        trainloader_generic, trainloader_custom, testloader = get_data(cfg, seed)
        
        purified_trainloader = deepcopy(trainloader_custom)
        targets = np.array(purified_trainloader.dataset.targets)
        targets[:, 0] = predicted_task_labels
        purified_trainloader.dataset.targets = targets.tolist()

        # # case 1 : no purification (target/OOD samples unknown), no weighting (alpha = n/(n+m))
        cfg.loss.group_task_loss = False
        err = run(cfg, trainloader_generic, testloader, rnum, "case1")
        sub_errs.append(err)

        # case 2 : target/OOD samples known, throw away all OOD samples (alpha = 1)
        cfg.loss.group_task_loss = True
        cfg.loss.alpha = 1
        err = run(cfg, trainloader_custom, testloader, rnum, "case2")
        sub_errs.append(err)

        # # case 3 : target/OOD samples known, optimal weighting
        cfg.loss.group_task_loss = True
        _ , err = search_alpha(cfg, trainloader_custom, testloader)
        if cfg.deploy:
            wandb.log({"{}_err".format("case3") : round(err, 4)})
        sub_errs.append(err)

        # # # case 4 : purification, throw away all OOD samples (alpha = 1)
        # cfg.loss.group_task_loss = True
        # cfg.loss.alpha = 1
        # err = run(cfg, purified_trainloader, testloader, rnum, "case4")
        # sub_errs.append(err)

        # # case 5 : purification, optimal weighting
        cfg.loss.group_task_loss = True
        _ , err = search_alpha(cfg, purified_trainloader, testloader)
        if cfg.deploy:
            wandb.log({"{}_err".format("case4") : round(err, 4)})
        sub_errs.append(err)

        errs.append(sub_errs)

    info = {
        "errs": errs,
        "avg_err": np.mean(errs, axis=0),
        "std_err": np.std(errs, axis=0)                                    
    }
    
    if cfg.deploy:
        wandb.log(info)

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()