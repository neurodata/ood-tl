import argparse
from math import exp
import random
import numpy as np
import torch
import pandas as pd
import seaborn as sns
sns.set_theme()

from utils.config import fetch_configs
from datahandlers.cifar import SplitCIFARHandler
from net.smallconv import SingleHeadNet
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
        for mn in exp_conf['m_n_ratio']:
            m = mn * n
            print("m = {}".format(m))
            for r, rep in enumerate(range(exp_conf['reps'])):
                print("T{} vs. T{} : Doing rep...{}".format(exp_conf['in_task'], task, rep))
                df.at[i, "m"] = mn
                df.at[i, "r"] = r

                dataset.sample_data(n=n, m=m, randomly=exp_conf['sample_scheme'])
                train_loader = dataset.get_data_loader(hp['batch_size'], train=True)
                net = SingleHeadNet(
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
                net = train(net, hp, train_loader, optimizer, lr_scheduler, gpu, verbose=False, task_id_flag=False)
                risk = evaluate(net, dataset, gpu, task_id_flag=False)
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
                        default="./experiments/config/singlehead_dual_tasks.yaml",
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
    if args.out_task is not None:
        exp_conf['out_task'] = args.out_task
    gpu = args.gpu

    print("Source Task(s) : {}".format(exp_conf['out_task']))
    print("Target Task : {}".format(exp_conf['in_task']))
    print("GPU : {}".format(gpu))

    run_experiment(exp_conf, gpu)


if __name__ == '__main__':
    main()


