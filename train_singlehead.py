import hydra
import torch
import torchvision
import numpy as np
import wandb

from utils.init import set_seed, open_log, init_wandb, cleanup

from datahandlers.cifar import SplitCIFARHandler
from net.smallconv import SmallConvSingleHeadNet
from net.wideresnet import WideResNetSingleHeadNet

from utils.run_net import train, evaluate


def get_data(cfg, seed):
    if cfg.task.dataset == "split_cifar10":
        dataHandler = SplitCIFARHandler(cfg)
    else:
        raise NotImplementedError

    # Use different seeds across different runs
    # But use the same seed
    dataHandler.sample_data(seed)
    trainloader = dataHandler.get_data_loader(train=True)
    testloader = dataHandler.get_data_loader(train=False)
    return trainloader, testloader


def get_net(cfg):
    if cfg.net == 'wrn10_2':
        net = WideResNetSingleHeadNet(
            depth=10,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=4,
            widen_factor=2,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'wrn16_4':
        net = WideResNetSingleHeadNet(
            depth=16,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=16,
            widen_factor=4,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'conv':
        net = SmallConvSingleHeadNet(
            num_cls=len(cfg.task.task_map[0]),
            channels=3, 
            avg_pool=2,
            lin_size=320
        )
    else: 
        raise NotImplementedError

    return net


@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    init_wandb(cfg, project_name="ood_tl")
    fp = open_log(cfg)

    if cfg.loss.tune_alpha:
        raise NotImplementedError

    errs = []
    for rnum in range(cfg.reps):
        seed =  cfg.seed + rnum * 10
        set_seed(seed)
        net = get_net(cfg)
        dataloaders = get_data(cfg, seed)
        train(cfg, net, dataloaders[0])
        errs.append(evaluate(cfg, net, dataloaders[1], rnum))

    info = {
        "avg_err": round(np.mean(errs), 4),
        "std_err": round(np.std(errs), 4)
    }
    print(info)
    if cfg.deploy:
        wandb.log(info)

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()
