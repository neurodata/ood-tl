# cifar10 smallconv multihead
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=True tag=10_multihead/cifar10_smallconv task.dataset=split_cifar10 net=multi_conv task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False

# cifar10 wrn_10_2 multihead
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 is_multihead=True tag=10_multihead/cifar10_wrn-10-2 task.dataset=split_cifar10 net=multi_wrn10_2 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False

# cifar10 wrn_10_2 multihead
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=True tag=10_multihead/cifar10_wrn-16-4 task.dataset=split_cifar10 net=multi_wrn16_4 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False