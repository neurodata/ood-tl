import hydra
import torch
import torchvision

from utils.init import set_seed, open_log, init_wandb, cleanup

from datahandlers.cifar import SplitCIFARHandler

def get_data(cfg):
    if cfg.task.dataset == "split_cifar10":
        return SplitCIFARHandler(cfg)
    else:
        raise NotImplementedError



@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    init_wandb(cfg, project_name="ood_tl")
    set_seed(cfg.seed)
    fp = open_log(cfg)

    dataHandler = get_data(cfg)
    dataHandler.sample_data()

    # test
    trainloader = dataHandler.get_data_loader(train=True)
    

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()
