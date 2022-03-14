import torch.nn as nn
import torch
import numpy as np

def train(net, hp, train_loader, optimizer, lr_scheduler, verbose=False):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net.to(device)

  for epoch in range(hp['epochs']):
    train_loss = 0.0
    train_acc = 0.0
    batches = 0.0
    criterion = nn.CrossEntropyLoss()

    net.train()

    for dat, target in train_loader:
        optimizer.zero_grad()

        labels = target
        labels = labels.long()
        batch_size = int(labels.size()[0])

        dat = dat.to(device)
        labels = labels.to(device)

        # Forward/Back-prop
        out = net(dat)
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

    if verbose:
      print("Epoch = {}".format(epoch))
      print("Train loss = {:3f}".format(train_loss/batches))
      print("Train acc = {:3f}".format(train_acc/batches))
      print("\n")
    
  return net

def evaluate(net, dataset):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  test_loader = dataset.get_data_loader(100, train=False)
  net.eval()

  acc = 0
  count = 0
  with torch.no_grad():
      for dat, target in test_loader:
          labels = target
          labels = labels.long()
          batch_size = int(labels.size()[0])

          dat = dat.to(device)
          labels = labels.to(device)

          out = net(dat)
          out = out.cpu().detach().numpy()
          
          labels = labels.cpu().numpy()
          acc += np.sum(labels == (np.argmax(out, axis=1)))
          count += batch_size

  error = 1-acc/count
  return error