## 10-way classfication
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=08_cinic10_10way/task_agnostic task.dataset=split_cinic10 net=wrn10_2 task.custom_sampler=False task.task_map=[[0,1,2,3,4,5,6,7,8,9]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hydra.launcher.n_jobs=10

## 2-way classfication

# samples per class = 10
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=08_cinic10_2way/task_agnostic task.dataset=split_cinic10 net=wrn10_2 task.custom_sampler=False task.task_map=[[1,9]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hydra.launcher.n_jobs=10

# samples per class = 50
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 tag=08_cinic10_2way/n50 task.dataset=split_cinic10 net=wrn10_2 task.custom_sampler=False task.task_map=[[1,9]] task.target=0 task.ood=[0] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hydra.launcher.n_jobs=10