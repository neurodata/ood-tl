## Naive Task Aware

# PACS

# singlhead PACS wrn10-2 (3 classes)
# python3 train_singlehead.py -m seed=10 random_reps=True reps=10 deploy=True device=cuda:2 is_multihead=False tag=16_task_aware/pacs_naive1 task.dataset=pacs net=wrn16_4 loss.group_task_loss=True loss.alpha=0.5 task.custom_sampler=True task.beta=0.75 task.target_env=P task.ood_env=S task.task_map=[[0,1,4]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,6,7,8,9,10 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10

# Split-CIFAR-10
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 is_multihead=False tag=16_task_aware/cifar10_naive_lol task.dataset=split_cifar10 net=wrn10_2 loss.group_task_loss=True loss.alpha=0.5 task.custom_sampler=True task.beta=0.75 task.target=1 task.ood=[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10

## Optimal Task Aware

# Split-CIFAR-10
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=False tag=16_task_aware/cifar10/opt task.dataset=split_cifar10 net=wrn10_2 loss.use_opt_alpha=True loss.group_task_loss=True loss.tune_alpha_tag=17_tune/cifar10/wrn10-2 loss.m_n_list=[0,1,2,3,4,5,10,20] task.custom_sampler=True task.beta=0.75 task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10

python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 is_multihead=False tag=16_task_aware/cifar10/opt task.dataset=split_cifar10 net=wrn10_2 loss.use_opt_alpha=True loss.group_task_loss=True loss.tune_alpha_tag=17_tune/cifar10/wrn10-2/selected loss.m_n_list=[0,1,2,3,4,5,10,20] task.custom_sampler=True task.beta=0.75 task.target=1 task.ood=[0],[2],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 is_multihead=False tag=16_task_aware/cifar10/opt task.dataset=split_cifar10 net=wrn10_2 loss.use_opt_alpha=True loss.group_task_loss=True loss.tune_alpha_tag=17_tune/cifar10/wrn10-2 loss.m_n_list=[0,1,2,3,4,5,10,20] task.custom_sampler=True task.beta=0.75 task.target=0 task.ood=[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 is_multihead=False tag=16_task_aware/cifar10/opt task.dataset=split_cifar10 net=wrn10_2 loss.use_opt_alpha=True loss.group_task_loss=True loss.tune_alpha_tag=17_tune/cifar10/wrn10-2 loss.m_n_list=[0,1,2,3,4,5,10,20] task.custom_sampler=True task.beta=0.75 task.target=2 task.ood=[1] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10

# PACS
# python3 train_singlehead.py -m seed=10 random_reps=True reps=10 deploy=True device=cuda:2 is_multihead=False tag=16_task_aware/pacs_opt2 task.dataset=pacs net=wrn16_4 loss.use_opt_alpha=True loss.group_task_loss=True loss.tune_alpha_tag=17_tune/pacs2 loss.m_n_list=[0,1,2,3,4,5,10] task.custom_sampler=True task.beta=0.75 task.target_env=P task.ood_env=S task.task_map=[[0,1,4]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,10 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10
