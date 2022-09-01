# This script had bug if m_n > 0 (includes ood dataset in testset)

# Task agnostic
python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=04_ood/task_agnostic loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=cinic10 task.n=10

notify "CINIC - target + ood"

# Task-aware group losses
