from random import betavariate
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import logging

def train(net, alpha, hp, train_loader, optimizer, lr_scheduler, gpu, is_multihead=False, verbose=False, patience=100):
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # initialize early stopping
    last_loss = 1000
    triggertimes = 0

    for epoch in range(hp['epochs']):
        target_train_loss = 0.0
        ood_train_loss = 0.0
        train_loss = 0.0
        batches = 0.0
        criterion = nn.CrossEntropyLoss(reduction='none') # when tuning, there are non-zero amount of OOD samples in the combined dataset

        net.train()

        for dat, target in train_loader:
            optimizer.zero_grad()
            tasks, labels = target
            tasks = tasks.long()
            labels = labels.long()
            tasks = tasks.to(device)
            labels = labels.to(device)
            
            batch_size = int(labels.size()[0])

            dat = dat.to(device)

            # forward pass
            if is_multihead:
                # if multi-head setup, the network takes data and task_id
                out = net(dat, tasks)
            else:
                # if single-head setup, the network only takes data
                out = net(dat)

            # compute the target and OOD risks separaely from the batch (and weight if specified)
            loss = criterion(out, labels)
            wt = alpha
            wo = (1-alpha)
            loss_target = torch.nan_to_num(loss[tasks==0].mean())
            loss_ood = torch.nan_to_num(loss[tasks==1].mean())
            loss = wt*loss_target + wo*loss_ood

            loss.backward()

            optimizer.step()

            lr_scheduler.step()

            # Compute Train metrics
            batches += batch_size
            train_loss += loss.item() * batch_size
            target_train_loss += loss_target.item() * batch_size
            ood_train_loss += loss_ood.item() * batch_size

        # Early stopping
        current_loss = train_loss/batches

        if current_loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                break
        else:
            trigger_times = 0
            last_loss = current_loss

        # logging
        if epoch % 20 == 0:
            info = {
                "tuning_alpha": round(alpha, 4),
                "epoch": epoch,
                "target_train_loss": round(target_train_loss/batches, 4),
                "ood_train_loss": round(ood_train_loss/batches, 4),
                "train_loss": round(train_loss/batches, 4),
            }
            logging.info(str(info))

    return net

def evaluate(net, val_loader, gpu, is_multihead=False):
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    
    net.eval()

    acc = 0
    count = 0
    with torch.no_grad():
        for dat, target in val_loader:

            tasks, labels = target
            tasks = tasks.long()
            labels = labels.long()
            tasks = tasks.to(device)
            labels = labels.to(device)
            
            batch_size = int(labels.size()[0])

            dat = dat.to(device)

            if is_multihead:
                # if multi-head setup, the network takes data and task_id
                out = net(dat, tasks)
            else:
                # if single-head setup, the network only takes data
                out = net(dat)
              
            out = out.cpu().detach().numpy()
            
            labels = labels.cpu().numpy()
            acc += np.sum(labels == (np.argmax(out, axis=1)))
            count += batch_size

    error = 1-acc/count
    return error

def search_alpha(net, dataset, n, hp, gpu, sensitivity=0.05, val_split=0.1, SEED=1996):

    train_loader = dataset.get_data_loader(hp['batch_size'], train=True, isTaskAware=True) # pass the task-aware flag to be true, since tuning of alpha occur only at the task-aware setting
    test_loader = dataset.get_task_data_loader(0, 100, train=False)

    # set alpha search space
    # alpha_range = np.arange(0.5, 1+1e-5, sensitivity)
    # alpha_range = np.concatenate((np.arange(0.5, 0.9, 0.05), np.arange(0.9, 1+1e-5, 0.001)))
    alpha_range = np.arange(0.9, 1+1e-5, 0.001)

    scores = []
 
    for alpha in alpha_range:
        risk_rep = []
        for rep in range(1):
            tune_net = deepcopy(net) # deepcopy the network architecture
            optimizer = torch.optim.SGD(
                tune_net.parameters(), 
                lr=hp['lr'],
                momentum=0.9, 
                nesterov=True,
                weight_decay=hp['l2_reg']
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                hp['epochs'] * len(train_loader)
            )
            tune_net = train(
                net=tune_net,
                alpha=alpha,
                hp=hp,
                train_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                gpu=gpu,
                verbose=False
            )
            risk_rep.append(evaluate(tune_net, test_loader, gpu))
        
        risk = np.mean(risk_rep)
        print("Risk at alpha = {:.4f} : {:.4f} +/- {:.4f}".format(alpha, risk, np.std(risk_rep)))
        info = {
                "alpha": round(alpha, 4),
                "risk_at_alpha": round(risk, 4),
            }
        logging.info(str(info))
        scores.append(risk)

    return alpha_range[np.argmin(scores)] # should we enforce that optimal alpha should be monotonically increasing?


    



