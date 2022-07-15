from random import betavariate
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def train(net, hp, train_loader, optimizer, lr_scheduler, gpu, task_id_flag=False, verbose=False, alpha=None, beta=None, patience=100):
  device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
  net.to(device)

  last_loss = 1000
  triggertimes = 0
  for epoch in range(hp['epochs']):
    train_loss = 0.0
    batches = 0.0
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
          # print("Target Fraction : {:.3f}".format(1-tasks.sum()/len(tasks)))
          loss = criterion(out, labels)
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

    # Early stopping
    current_loss = train_loss/batches

    if current_loss > last_loss:
        trigger_times += 1
        if trigger_times >= patience:
            break
    else:
        trigger_times = 0
        last_loss = current_loss

  return net

def evaluate(net, val_loader, gpu, task_id_flag=False):
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

def search_alpha(net, dataset, n, beta, hp, gpu, sensitivity=0.05, val_split=0.1, SEED=1996):
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
    
    # X = dataset.comb_trainset.data
    # y = dataset.comb_trainset.targets

    # tune_trainset = deepcopy(dataset.comb_trainset)
    # tune_valset = deepcopy(dataset.comb_trainset) 

    # target_X_train, target_X_val, target_y_train, target_y_val = train_test_split(X[:n], y[:n], test_size=val_split)
    # tune_trainset.data = np.concatenate((target_X_train, X[n:]))
    # tune_trainset.targets = np.concatenate((target_y_train, y[n:])).tolist()
    # tune_valset.data = target_X_val
    # tune_valset.target = target_y_val

    # tune_train_loader = DataLoader(tune_trainset, batch_size=hp['batch_size'], shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4)
    # tune_val_loader = DataLoader(tune_valset, batch_size=len(target_y_val), shuffle=True, worker_init_fn=wif, pin_memory=True, num_workers=4)   

    train_loader = dataset.get_data_loader(hp['batch_size'], train=True)
    test_loader = dataset.get_task_data_loader(0, 100, train=False)

    alpha_range = np.arange(0.5, 1+1e-5, sensitivity)
    scores = []
 
    for alpha in alpha_range:
        risk_rep = []
        for rep in range(1):
            tune_net = deepcopy(net)
            optimizer = torch.optim.SGD(tune_net.parameters(), lr=hp['lr'],
                                                momentum=0.9, nesterov=True,
                                                weight_decay=hp['l2_reg'])
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, hp['epochs'] * len(train_loader))
            tune_net = train(tune_net, hp, train_loader, optimizer, lr_scheduler, gpu, verbose=False, task_id_flag=False, alpha=alpha, beta=beta)
            risk_rep.append(evaluate(tune_net, test_loader, gpu))
        risk = np.mean(risk_rep)
        print("Risk at alpha = {:.4f} : {:.4f} +/- {:.4f}".format(alpha, risk, np.std(risk_rep)))
        scores.append(risk)

    return alpha_range[np.argmin(scores)]


    



