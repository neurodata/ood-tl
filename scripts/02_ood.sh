# This script had bug if m_n > 0 (includes ood dataset in testset)

# Task agnostic
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=02_beta/task_agnostic  loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15 task.augment=True,False

# Task-aware group losses
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=02_beta/task_loss_group loss.group_task_loss=True task.custom_sampler=False task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15 task.augment=True,False

# Task-aware - custom mini-batch
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=02_beta/task_aware_sampler loss.group_task_loss=False task.custom_sampler=True task.beta=unbiased  task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15 task.augment=True,False
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=02_beta/task_aware_sampler loss.group_task_loss=False task.custom_sampler=True task.beta=0.5       task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15 task.augment=True,False

# Task-aware - group losses + Custom mini-batch
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=02_beta/ta_sampler_loss task.beta=unbiased     loss.group_task_loss=True task.custom_sampler=True task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15 task.augment=True,False
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=02_beta/ta_sampler_loss task.beta=0.2,0.3,0.4,0.5,0.6,0.7,0.8 loss.group_task_loss=True task.custom_sampler=True task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15 task.augment=True,False


notify "target + ood"
