# cifar10 smallconv singlehead
python3 train_singlehead.py -m seed=50 reps=10 deploy=True device=cuda:3 is_multihead=False tag=13_singlehead/cifar10_smallconv_seed50 task.dataset=split_cifar10 net=conv task.custom_sampler=False task.target=0,1 task.ood=[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hydra.launcher.n_jobs=10

# cifar10 wrn_10_2 singlehead
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=False tag=13_singlehead/cifar10_smallconv task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hydra.launcher.n_jobs=10

# cifar10 wrn_16_4 singlehead
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 is_multihead=False tag=13_singlehead/cifar10_wrn-16-4 task.dataset=split_cifar10 net=wrn16_4 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hydra.launcher.n_jobs=10