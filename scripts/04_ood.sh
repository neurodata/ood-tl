# This script had bug if m_n > 0 (includes ood dataset in testset)

# Task agnostic
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=04_ood/task_agnostic loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=cinic10 task.n=10

# Task-aware group losses
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=04_ood/singlehead_agnostic_mixed_cinic1 loss.group_task_loss=False task.custom_sampler=False task.m_n=1 task.augment=False task.dataset=cinic_img task.n=10 task.num_cinic=200,1500,500,750,1000

python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=04_ood/singlehead_agnostic_mixed_cinic1 loss.group_task_loss=False task.custom_sampler=False task.m_n=1 task.augment=False task.dataset=cinic_img task.n=10 task.num_cinic=2500,5000

# python3 train_singlehead.py -m seed=10 reps=5 deploy=True tag=04_ood/singlehead_agnostci_mixed_cinic loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.5,1,2,4,6,8,10,12,15,20,25,30 task.augment=False task.dataset=cinic_img task.n=10


notify "CINIC-negative + imagenet - target + ood"
