import hydra
import torch
import torchvision
import numpy as np
import wandb
import pandas as pd

from utils.init import set_seed, open_log, init_wandb, cleanup

from datahandlers.cifar import SplitCIFARHandler, RotatedCIFAR10Handler, BlurredCIFAR10Handler
from datahandlers.cinic import SplitCINIC10Handler, SplitCIFAR10NegHandler
from datahandlers.mnist import RotatedMNISTHandler
from datahandlers.officehomes import OfficeHomeHandler
from datahandlers.pacs import PACSHandler
from net.smallconv import SmallConvSingleHeadNet, SmallConvMultiHeadNet
from net.wideresnet import WideResNetSingleHeadNet, WideResNetMultiHeadNet

from utils.run_net import train, evaluate


def get_data(cfg, seed):
    if cfg.task.dataset == "split_cifar10":
        dataHandler = SplitCIFARHandler(cfg)
    elif cfg.task.dataset == "split_cinic10":
        dataHandler = SplitCINIC10Handler(cfg)
    elif cfg.task.dataset == "rotated_cifar10":
        dataHandler = RotatedCIFAR10Handler(cfg)
    elif cfg.task.dataset == "blurred_cifar10":
        dataHandler = BlurredCIFAR10Handler(cfg)
    elif cfg.task.dataset == "split_cifar10neg":
        dataHandler = SplitCIFAR10NegHandler(cfg)
    elif cfg.task.dataset == "rotated_mnist":
        dataHandler = RotatedMNISTHandler(cfg)
    elif cfg.task.dataset == "officehomes":
        dataHandler = OfficeHomeHandler(cfg)
    elif cfg.task.dataset == "pacs":
        dataHandler = PACSHandler(cfg)
    else:
        raise NotImplementedError

    # Use different seeds across different runs
    # But use the same seed
    dataHandler.sample_data(seed)
    task_labels = np.array(dataHandler.comb_trainset.targets)[:, 0]
    num_target_samples = len(task_labels[task_labels==0])
    num_ood_samples = len(task_labels[task_labels==1])
    info = {
        "n": num_target_samples,
        "m": num_ood_samples
    }
    if cfg.deploy:
        wandb.log(info)
    trainloader = dataHandler.get_data_loader(train=True)
    testloader = dataHandler.get_data_loader(train=False)
    unshuffled_trainloader = dataHandler.get_data_loader(train=True, shuffle=False)
    return trainloader, testloader, unshuffled_trainloader


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
            channels=1, # for cifar:3, mnist:80
            avg_pool=2,
            lin_size=80 # for cifar:320, mnist:80
        )
    elif cfg.net == 'multi_conv':
        net = SmallConvMultiHeadNet(
            num_task=2,
            num_cls=len(cfg.task.task_map[0]),
            channels=3, 
            avg_pool=2,
            lin_size=320
        )
    elif cfg.net == 'multi_wrn10_2':
        net = WideResNetMultiHeadNet(
            depth=10,
            num_task=2,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=4,
            widen_factor=2,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'multi_wrn16_4':
        net = WideResNetMultiHeadNet(
            depth=16,
            num_task=2,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=16,
            widen_factor=4,
            drop_rate=0.2,
            inp_channels=3
        )
    else: 
        raise NotImplementedError

    return net

def get_opt_alpha(cfg):
    api = wandb.Api()
    runs = api.runs("ashwin1996/ood_tl")
    tag = cfg.loss.tune_alpha_tag

    for run in runs: 
        try:
            run_tag = run.config['tag']
        except KeyError:
            run_tag = "none"
        if (run_tag == tag) and (run.config['task']['target'] == cfg.task.target) and (run.config['task']['ood'][0] == cfg.task.ood[0]):
            opt_alpha_list = run.summary['opt_alpha_list']
            break

    # opt_alpha_list = [0.5,0.9,0.9444444444444444,0.9814814814814816,0.9835390946502058,0.9890260631001372,0.995122694711172]
    
    m_n_list = np.array(cfg.loss.m_n_list)
    idx = np.where(m_n_list == cfg.task.m_n)[0][0]
    return opt_alpha_list[idx]

@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    init_wandb(cfg, project_name="ood_tl")
    fp = open_log(cfg)

    if cfg.loss.use_opt_alpha:
        alpha = get_opt_alpha(cfg)
        cfg.loss.alpha = np.float64(alpha).item()
        info = {
            'alpha': alpha
        }
        if cfg.deploy:
            wandb.log(info)

    errs = []
    for rnum in range(cfg.reps):
        if cfg.random_reps:
            seed =  cfg.seed + rnum * 10
        else:
            seed = cfg.seed
        set_seed(seed)
        net = get_net(cfg)
        dataloaders = get_data(cfg, seed)
        train(cfg, net, dataloaders)
        err = evaluate(cfg, net, dataloaders[1], rnum)
        errs.append(err)

    info = {
        "avg_err": round(np.mean(errs), 4),
        "std_err": round(np.std(errs), 4)
    }
    print(info)
    if cfg.deploy:
        wandb.log(info)

    cleanup(cfg, fp)

    # save_results(cfg)


if __name__ == "__main__":
    main()
