import argparse
from math import exp
import random
import numpy as np
import torch
import pandas as pd

from utils.config import fetch_configs
from datasets.cifar import SplitCIFARHandler
from net.smallconv import MultiHeadNet
from net.wideresnet import WideResNetMultihead
from utils.run_net import train, evaluate

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def run_experiment(exp_conf, gpu):
    n = exp_conf['n']
    hp = exp_conf['hp']
    task_dict = exp_conf['task_dict']
    df = pd.DataFrame()

    for task in exp_conf['out_task']:
        print("Doing task...T{}".format(task))
        tasks = [task_dict[exp_conf['in_task']], task_dict[task]]
        print("Tasks: {}".format(tasks))
        dataset = SplitCIFARHandler(tasks)
        i = 0
        for m in exp_conf['m']:
            print("m = {}".format(m))
            for r, rep in enumerate(range(exp_conf['reps'])):
                print("T{} vs. T{} : Doing rep...{}".format(exp_conf['in_task'], task, rep))
                df.at[i, "m"] = m
                df.at[i, "r"] = r

                dataset.sample_data(n=n, m=m)
                train_loader = dataset.get_data_loader(hp['batch_size'], train=True)
                if 'net' in exp_conf and exp_conf['net'] == 'wrn':
                    net = WideResNetMultihead(
                        depth=16,
                        num_task=len(tasks), 
                        num_cls=len(tasks[0]),
                        widen_factor=4,
                        drop_rate=0.2
                    )
                else:
                    net = MultiHeadNet(
                        num_task=len(tasks), 
                        num_cls=len(tasks[0]),
                        channels=3, 
                        avg_pool=2,
                        lin_size=320
                    )
                optimizer = torch.optim.SGD(net.parameters(), lr=hp['lr'],
                                            momentum=0.9, nesterov=True,
                                            weight_decay=hp['l2_reg'])
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, hp['epochs'] * len(train_loader))
                net = train(net, hp, train_loader, optimizer, lr_scheduler, gpu, verbose=False, task_id_flag=True)
                risk = evaluate(net, dataset, gpu, task_id_flag=True)
                print("Risk = %0.4f" % risk)
                df.at[i, str(task)] = risk
                i+=1
        print("Saving individual results...")
        df.to_csv('{}/{}_{}_{}_T{}_T{}.csv'.format(exp_conf['save_folder'], exp_conf['dataset'], exp_conf['net'], exp_conf['exp_name'], exp_conf['in_task'], task))
    print("Saving bulk results...")
    df.to_csv('{}/{}_{}_{}_T{}.csv'.format(exp_conf['save_folder'], exp_conf['dataset'], exp_conf['net'], exp_conf['exp_name'], exp_conf['in_task']))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_config', type=str,
                        default="./experiments/config/multihead_dual_tasks.yaml",
                        help="Experiment configuration")

    parser.add_argument('--in_task', type=int,
                            help="Source task")

    parser.add_argument('--out_task', nargs='+', type=int,
                            help="Target task(s)")

    parser.add_argument('--gpu', type=str,
                            default='cuda:0',
                            help="GPU")                      

    args = parser.parse_args()
    exp_conf = fetch_configs(args.exp_config)
    if args.in_task is not None:
        exp_conf['in_task'] = args.in_task
    if args.out_tasks is not None:
        exp_conf['out_task'] = args.out_task
    gpu = args.gpu

    print("Source Task : {}".format(exp_conf['in_task']))
    print("Target Task(s) : {}".format(exp_conf['out_task']))
    print("GPU : {}".format(gpu))

    run_experiment(exp_conf, gpu)


if __name__ == '__main__':
    main()
