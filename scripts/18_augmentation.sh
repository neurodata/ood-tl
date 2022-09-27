# Naive Task Aware (With Augmentation)
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 is_multihead=False tag=18_aug/naive task.dataset=split_cifar10 net=wrn10_2 loss.group_task_loss=True loss.alpha=0.5 task.custom_sampler=True task.beta=0.75 task.target=1 task.ood=[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=True hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10

# Tune Alpha (With Augmentation)
# python3 tune_alpha.py -m seed=10 deploy=True device=cuda:1 is_multihead=False tag=18_aug/tune2 task.dataset=split_cifar10 net=wrn10_2 loss.tune_alpha=True loss.group_task_loss=True loss.alpha=0.5 loss.m_n_list=[0,1,2,3,4,5,10,20] task.custom_sampler=True task.beta=0.75 task.target=1 task.ood=[4] task.n=50 task.m_n=0 task.augment=True hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10

# Optimal Task Aware (With Augmentation)
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 is_multihead=False tag=18_aug/opt2 task.dataset=split_cifar10 net=wrn10_2 loss.use_opt_alpha=True loss.group_task_loss=True loss.tune_alpha_tag=18_aug/tune2 loss.m_n_list=[0,1,2,3,4,5,10,20] task.custom_sampler=True task.beta=0.75 task.target=1 task.ood=[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=True hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10