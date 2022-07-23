import logging
from multiprocessing import reduction
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def train(net, alpha, hp, train_loader, optimizer, lr_scheduler, gpu, is_multihead=False, verbose=False, isTaskAware=True, patience=100):
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # initialize early stopping
    last_loss = 1000
    triggertimes = 0

    for epoch in range(hp['epochs']):
        train_loss = 0.0
        train_acc = 0.0
        batches = 0.0
    
        if isTaskAware:
            # if task-aware return losses of the batch separately (no aggregation)
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            # if task-agnostic return the mean loss of the batch
            criterion = nn.CrossEntropyLoss()
            
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

            if isTaskAware:
                # if task-aware, compute the target and OOD risks separaely from the batch (and weight if specified)
                # print("target instance fraction: {:.3f}".format(1-tasks.sum()/len(tasks)))
                loss = criterion(out, labels)
                wt = alpha
                wo = (1-alpha)
                loss_target = torch.nan_to_num(loss[tasks==0].mean())
                loss_ood = torch.nan_to_num(loss[tasks==1].mean())
                loss = wt*loss_target + wo*loss_ood
            else:
                # if task-agnostic, compute the mean of the batch losses
                loss = criterion(out, labels)

            loss.backward()

            optimizer.step()

            lr_scheduler.step()

            # Compute Train metrics
            batches += batch_size
            train_loss += loss.item() * batch_size
            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            train_acc += np.sum(labels == (np.argmax(out, axis=1)))

        # Early stopping
        current_loss = train_loss/batches

        if current_loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                break
        else:
            trigger_times = 0
            last_loss = current_loss

        if epoch % 10 == 0:
            info = {
                "epoch": epoch,
                "train_loss": round(train_loss/batches, 4),
                "train_acc": round(train_acc/batches, 4)
            }
            logging.info(str(info))

        if verbose:
            print("Epoch = {}".format(epoch))
            print("Train loss = {:3f}".format(train_loss/batches))
            print("Train acc = {:3f}".format(train_acc/batches))
            print("\n")
    
    return net

def evaluate(net, dataset, gpu, is_multihead=False):
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    # get the data loader that only returns the target samples
    test_loader = dataset.get_task_data_loader(0, 100, train=False)

    net.eval()

    acc = 0
    count = 0
    with torch.no_grad():
        for dat, target in test_loader:

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
