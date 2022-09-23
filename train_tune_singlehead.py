import hydra
import torch
import torchvision
import numpy as np
import wandb

from utils.init import set_seed, open_log, init_wandb, cleanup

from datahandlers.cifar import SplitCIFARHandler
from datahandlers.domain_net import DomainNetHandler
from net.smallconv import SmallConvSingleHeadNet
from net.wideresnet import WideResNetSingleHeadNet
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter

from utils.run_net import train, evaluate, train_tune
from train_singlehead import get_net

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



def get_data_handler(cfg, seed):
    if cfg.task.dataset == "split_cifar10":
        dataHandler = SplitCIFARHandler(cfg)
    elif cfg.task.dataset == "cinic10":
        raise NotImplementedError
    elif cfg.task.dataset == "cinic_img":
        raise NotImplementedError
    elif cfg.task.dataset == "domain_net":
        dataHandler = DomainNetHandler(cfg)
    else:
        raise NotImplementedError

    # Use different seeds across different runs
    # But use the same seed
    dataHandler.sample_data(seed)
    return dataHandler



def get_tune_config(cfg):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "wd": tune.loguniform(1e-6, 1e-1),
        "bs": tune.uniform(1.5, 6.5)
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=cfg.hp.epochs,
        grace_period=2,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["accuracy", "training_iteration"])

    algo = BayesOptSearch(
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
        metric="accuracy",
        mode="max")
    algo = ConcurrencyLimiter(algo, max_concurrent=8)

    return config, scheduler, reporter, algo


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
        datahandler = get_data_handler(cfg, seed)

        """
        https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
        """
        tune_config, scheduler, reporter, algo = get_tune_config(cfg)

        result = tune.run(
            partial(train_tune, cfg=cfg, net=net, dataHandler=datahandler),
            resources_per_trial={"cpu": 2, "gpu": 0.25},
            config=tune_config,
            num_samples=128,
            search_alg=algo,
            verbose=0,
            scheduler=scheduler,
            progress_reporter=reporter)

        best_config = result.get_best_config('accuracy', 'max')
        err = 1 - np.max(result.results_df['accuracy'])
        print(err, best_config)
        errs.append(err)

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
