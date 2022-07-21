import os
import argparse
import datetime
import yaml
from math import exp
import random
import numpy as np
import torch
import pandas as pd
import seaborn as sns
sns.set_theme()

from utils.config import fetch_configs
from datahandlers.cifar import RotatedCIFAR10Handler
from net.smallconv import SmallConvSingleHeadNet
from utils.run_net import train, evaluate
from utils.tune import search_alpha

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def run_experiment(exp_conf, gpu):
    n = exp_conf['n']
    hp = exp_conf['hp']
    df = pd.DataFrame()

    for angle in exp_conf['angles']:
        print("Doing angle = {}".format(angle))
        dataset = RotatedCIFAR10Handler(exp_conf['task'], angle)
        
        i = 0
        for mn in exp_conf['m_n_ratio']:
            m = mn * n
            print("m = {}".format(m))
            alpha = 0.5

            for r, rep in enumerate(range(exp_conf['reps'])):
                print("Angle = {} : Doing rep...{}".format(angle, rep))
                df.at[i, "m"] = mn
                df.at[i, "r"] = r

                dataset.sample_data(n=n, m=m)
                
                if exp_conf['net'] == 'smallconv':
                    net = SmallConvSingleHeadNet(
                        num_task=2, 
                        num_cls=2,
                        channels=3, 
                        avg_pool=2,
                        lin_size=320
                    )

                if exp_conf['task_aware']:
                    if r==0:
                        if m == 0:
                            alpha = 0.5
                        else:
                            if exp_conf['tune_alpha']:                            
                                alpha = search_alpha(net, dataset, n, hp, gpu, sensitivity=0.05, val_split=exp_conf['val_split'])
                                print("Optimal alpha = {:.4f}".format(alpha))
                            else:
                                alpha = 0.5
                else:
                    alpha = None
                
                train_loader = dataset.get_data_loader(hp['batch_size'], train=True)
                optimizer = torch.optim.SGD(net.parameters(), 
                                            lr=hp['lr'],
                                            momentum=0.9, 
                                            nesterov=True,
                                            weight_decay=hp['l2_reg'])
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                            optimizer, 
                                            hp['epochs'] * len(train_loader))
                net = train(net, hp, train_loader, optimizer, lr_scheduler, gpu, verbose=False, task_id_flag=False, alpha=alpha)
                risk = evaluate(net, dataset, gpu, task_id_flag=False)
                print("Risk = %0.4f" % risk)
                df.at[i, str(angle)] = risk
                df.at[i, "alpha"] = alpha
                i+=1
        
        print("Saving individual results...")
        dfi = df.filter(['m', 'r', str(angle)])
        dfi.to_csv('{}/{}_{}_{}_{}.csv'.format(exp_conf['save_folder'], exp_conf['dataset'], exp_conf['net'], exp_conf['exp_name'], angle))
    
    print("Saving bulk results...")
    df.to_csv('{}/{}_{}_{}.csv'.format(exp_conf['save_folder'], exp_conf['dataset'], exp_conf['net'], exp_conf['exp_name']))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_config', type=str,
                        default="./experiments/config/singlehead_rotated_tasks.yaml",
                        help="Experiment configuration")

    parser.add_argument('--tune_alpha', type=bool,
                            help="Whether to tune alpha or not")

    parser.add_argument('--gpu', type=str,
                            default='cuda:0',
                            help="GPU")    

    args = parser.parse_args()
    exp_conf = fetch_configs(args.exp_config)
    if args.tune_alpha is not None:
        exp_conf['tune_alpha'] = args.tune_alpha
    gpu = args.gpu

    print("angles : {}".format(exp_conf['angles']))
    print("GPU : {}".format(gpu))

    if not os.path.exists(exp_conf['save_folder']):
        os.makedirs(exp_conf['save_folder'])
    exp_folder_path = os.path.join(exp_conf['save_folder'], str(datetime.datetime.now()))
    os.makedirs(exp_folder_path)
    exp_conf['save_folder'] = exp_folder_path

    if exp_conf['task_aware']:
        if exp_conf['tune_alpha']:
            setting = "Optimal_Task_Aware"
        else:
            setting = "Naive_Task_Aware"
    else:
        setting = "Task_Agnostic"

    print("Experimental Setting : ", setting)
    exp_conf['setting'] = setting

    with open('exp_config.yml', 'w') as outfile:
        yaml.dump(exp_conf, outfile, default_flow_style=False)

    run_experiment(exp_conf, gpu)

if __name__ == '__main__':
    main()
