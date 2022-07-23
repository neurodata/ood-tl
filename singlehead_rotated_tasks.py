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
import logging
sns.set_theme()

from utils.config import fetch_configs
from datahandlers.cifar import RotatedCIFAR10Handler
from net.smallconv import SmallConvSingleHeadNet
from utils.run_net import train, evaluate
from utils.tune import search_alpha

# set random seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def run_experiment(exp_conf, gpu):
    log_filename = exp_conf['save_folder'] + "/exp_log.log"
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    logging.info(str(exp_conf))

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

                dataset.sample_data(n=n, m=m, randomly=False)
                
                if exp_conf['net'] == 'smallconv':
                    # define the network
                    net = SmallConvSingleHeadNet(
                        num_task=2, 
                        num_cls=2,
                        channels=3, 
                        avg_pool=2,
                        lin_size=320
                    )

                if exp_conf['task_aware']:
                    # tuning occurs only in the task-aware setting
                    if r == 0:
                        # tunning occures before the first replicate
                        if m == 0:
                            # if no OOD samples are present, alpha is set to 0.5
                            alpha = 0.5
                        else:
                            if exp_conf['tune_alpha']:     
                                # if OOD samples are present, we search for the optimal alpha  
                                alpha = search_alpha(
                                    net=net,
                                    dataset=dataset,
                                    n=n,
                                    hp=hp,
                                    gpu=gpu,
                                    sensitivity=0.05,
                                    val_split=exp_conf['val_split']
                                )                     
                                print("Optimal alpha = {:.4f}".format(alpha))
                            else:
                                # if naive-task-aware setting, no alpha tuning is done
                                alpha = 0.5
                else:
                    # if task-agnostic alpha is set to None
                    alpha = None
                
                train_loader = dataset.get_data_loader(
                    batch_size=hp['batch_size'],
                    train=True,
                    isTaskAware=exp_conf['task_aware']
                )

                optimizer = torch.optim.SGD(
                    net.parameters(), 
                    lr=hp['lr'],
                    momentum=0.9, 
                    nesterov=True,
                    weight_decay=hp['l2_reg']
                )

                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    hp['epochs'] * len(train_loader)
                )

                net = train(
                    net=net,
                    alpha=alpha,
                    hp=hp,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    gpu=gpu,
                    is_multihead=False,
                    verbose=False,
                    isTaskAware=exp_conf['task_aware']
                )

                risk = evaluate(net, dataset, gpu)
                print("Risk = %0.4f" % risk)
                df.at[i, str(angle)] = risk
                df.at[i, "{}_alpha".format(angle)] = alpha

                info = {
                    "angle": angle,
                    "m_n_ratio": mn,
                    "replicate_id": r,
                    "alpha": alpha,
                    "target_risk": risk
                }
                logging.info(str(info))
                
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

    parser.add_argument('--makefolder', type=bool,
                            default=True,
                            help="Whether to make a new exp folder or not") 

    args = parser.parse_args()
    exp_conf = fetch_configs(args.exp_config)
    if args.tune_alpha is not None:
        exp_conf['tune_alpha'] = args.tune_alpha
    gpu = args.gpu

    print("angles : {}".format(exp_conf['angles']))
    print("GPU : {}".format(gpu))

    # obtain the setting of the experiment
    if exp_conf['task_aware']:
        if exp_conf['tune_alpha']:
            setting = "Optimal_Task_Aware"
        else:
            setting = "Naive_Task_Aware"
    else:
        setting = "Task_Agnostic"

    if args.makefolder:
        # if the specified experiment folder doesn't exist, make a new folder with the specified folder name
        if not os.path.exists(exp_conf['save_folder']):
            os.makedirs(exp_conf['save_folder'])
        # create a results folder to store the results from the current experiment
        exp_folder_path = os.path.join(exp_conf['save_folder'], "{}_{}".format(setting, datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')))
        os.makedirs(exp_folder_path)
        exp_conf['save_folder'] = exp_folder_path

    print("Experimental Setting : ", setting)
    exp_conf['setting'] = setting

    # save the experiment config file in the current results folder
    with open(os.path.join(exp_folder_path, 'exp_config.yml'), 'w') as outfile:
        yaml.dump(exp_conf, outfile, default_flow_style=False)

    run_experiment(exp_conf, gpu)

if __name__ == '__main__':
    main()
