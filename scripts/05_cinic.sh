# Evaluat the effect of distribution shift within CINIC
python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_cinic/task_agnostic net=conv task.dataset=split_cinic10 task.custom_sampler=False task.target=1,2,3,4 task.ood=[0] task.m_n=0,1,2,3,4,5,10,20 task.augment=False