import hydra
import torch
import torchvision

from utils.init import set_seed, open_log, init_wandb, cleanup
from utils.data import get_cifar10_dataset, get_dataloader



@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    init_wandb(cfg, project_name="ood_tl")
    set_seed(cfg.seed)
    fp = open_log(cfg)

    # Code here

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()
