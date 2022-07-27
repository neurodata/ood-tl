# This script had bug if m_n > 0 (includes ood dataset in testset)

# Task agnostic
python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=03_ood/task_agnostic   loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15,17.5,20 task.augment=True,False
notify "target + ood - script 1"

# Task-aware group losses
python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=03_ood/task_loss_group loss.group_task_loss=True  task.custom_sampler=False task.m_n=0,0.5,1,2,3,4,5,7.5,10,12.5,15,17.5,20 task.augment=True,False

notify "target + ood - script 2"
