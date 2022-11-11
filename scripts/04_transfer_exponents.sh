# Monitor the variation of transfer exponents when OOD sample size increases

## Task Agnostic 
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 tag=04_transfer_exps_cifar10/wrn task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.m_n=0,1,2,3,4,5,10,20 task.augment=False

## Naive Task Aware
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 tag=04_transfer_exps_cifar10/wrn/naive_aware task.dataset=split_cifar10 net=wrn10_2 loss.group_task_loss=True loss.alpha=0.5 task.custom_sampler=True task.beta=0.75 task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10

## Sanity Check
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 tag=04_cifar10/wrn-10-2/n50 task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.n=25 task.m_n=0,0.5,1,1.5,2,4,6,8,10,20,40 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=04_cifar10/wrn-10-2/n100 task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.n=50 task.m_n=0,0.25,0.5,0.75,1,2,3,4,5,10,20 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 tag=04_cifar10/wrn-10-2/n200 task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.n=100 task.m_n=0,0.125,0.25,0.375,0.5,1,1.5,2,2.5,5,10 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10