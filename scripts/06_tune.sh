#Debug

# Task agnostic
# CUDA_VISIBLE_DEVICES=2,3 python3 train_tune_singlehead.py -m seed=10 reps=1 deploy=False tag=06_ood/test loss.group_task_loss=False task.custom_sampler=False task.m_n=2 task.augment=False device=cuda:0 task.target=1 task.ood=[4] net=wrn10_2

CUDA_VISIBLE_DEVICES=2,3 python3 train_tune_singlehead.py -m seed=10 reps=10 deploy=True tag=06_ood/02_cifar10_exponent loss.group_task_loss=False task.custom_sampler=False task.m_n=0,1,2,3,4,5,8,10,15,20 task.augment=False device=cuda:0 task.target=1 task.ood=[4] net=wrn10_2


notify "hyper-param tuning"
