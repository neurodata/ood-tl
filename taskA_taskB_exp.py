import argparse
from math import exp
import random
import numpy as np
import torch
import pandas as pd
import seaborn as sns
sns.set_theme()

from utils.config import fetch_configs
from datasets.cifar import SplitCIFARHandler
from net.smallconv import SmallConv
from utils.run_net import train, evaluate

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def run_experiment(exp_conf):
    n = exp_conf['n']
    hp = exp_conf['hp']
    df = pd.DataFrame()

    for task_id, task in enumerate(exp_conf['out_task']):
        print("Doing angle = {}".format(str(task)))
        tasks = [exp_conf['in_task'], task]
        dataset = SplitCIFARHandler(tasks)
        i = 0
        for m in exp_conf['m']:
            print("m = {}".format(m))
            for r, rep in enumerate(range(exp_conf['reps'])):
                print("Doing rep...{}".format(rep))
                df.at[i, "m"] = m
                df.at[i, "r"] = r

                dataset.sample_data(n=n, m=m)
                train_loader = dataset.get_data_loader(hp['batch_size'], train=True)
                net = SmallConv(
                    num_task=1, 
                    num_cls=2,
                    channels=3, 
                    avg_pool=2,
                    lin_size=320
                )
                optimizer = torch.optim.SGD(net.parameters(), lr=hp['lr'],
                                        momentum=0.9, nesterov=True,
                                        weight_decay=hp['l2_reg'])
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, hp['epochs'] * len(train_loader))
                net = train(net, hp, train_loader, optimizer, lr_scheduler, verbose=False)
                risk = evaluate(net, dataset)
                df.at[i, str(task_id)] = risk
                i+=1

    df.to_csv('./experiments/results/taskA_taskB_exp_{}.csv'.format(str(exp_conf['in_task'])))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_config', type=str,
                        default=".experiments/config/taskA_taskB_exp.yaml",
                        help="Experiment configuration")

    args = parser.parse_args()
    exp_conf = fetch_configs(args.exp_config)

    run_experiment(exp_conf)


if __name__ == '__main__':
    main()
