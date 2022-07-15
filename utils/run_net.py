from multiprocessing import reduction
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def train(net, hp, train_loader, optimizer, lr_scheduler, gpu, task_id_flag=False, verbose=False, alpha=None, beta=None):
  device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
  net.to(device)

  for epoch in range(hp['epochs']):
    train_loss = 0.0
    train_acc = 0.0
    batches = 0.0
    
    if alpha is None:
      criterion = nn.CrossEntropyLoss()
    else:
      criterion = nn.CrossEntropyLoss(reduction='none')

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

        # Forward/Back-prop
        if task_id_flag:
          out = net(dat, tasks)
        else:
          out = net(dat)

        if hp["batch_size"] == 1:
          loss = criterion(out, labels)
          if tasks == 0:
            loss = alpha*loss
          else:
            loss = (1-alpha)*loss
        else: 
          if alpha is None:
            loss = criterion(out, labels)
          else:
            # print("target instance fraction: {:.3f}".format(1-tasks.sum()/len(tasks)))
            loss = criterion(out, labels)
            if tasks.sum() == len(tasks):
              loss = 1e-4*loss_ood
            else:
              wt = alpha
              wo = (1-alpha)
              loss_target = loss[tasks==0].mean()
              loss_ood = loss[tasks==1].mean()
              loss = wt*loss_target + wo*loss_ood

        loss.backward()

        optimizer.step()

        lr_scheduler.step()

        # Compute Train metrics
        batches += batch_size
        train_loss += loss.item() * batch_size
        labels = labels.cpu().numpy()
        out = out.cpu().detach().numpy()
        train_acc += np.sum(labels == (np.argmax(out, axis=1)))

    if verbose:
      print("Epoch = {}".format(epoch))
      print("Train loss = {:3f}".format(train_loss/batches))
      print("Train acc = {:3f}".format(train_acc/batches))
      print("\n")
    
  return net

def evaluate(net, dataset, gpu, task_id_flag=False):
  device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

  # if task_id_flag:
  #   test_loader = dataset.get_task_data_loader(0, 100, train=False)
  # else:
  #   test_loader = dataset.get_data_loader(100, train=False)
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

          if task_id_flag:
            out = net(dat, tasks)
          else:
            out = net(dat)
            
          out = out.cpu().detach().numpy()
          
          labels = labels.cpu().numpy()
          acc += np.sum(labels == (np.argmax(out, axis=1)))
          count += batch_size

  error = 1-acc/count
  return error
