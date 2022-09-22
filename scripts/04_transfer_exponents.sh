# Monitor the variation of transfer exponents when OOD sample size increases

## Task Agnostic 
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 tag=04_transfer_exps_cifar10/wrn task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.m_n=0,1,2,3,4,5,10,20 task.augment=False

## Naive Task Aware
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:1 tag=04_transfer_exps_cifar10/wrn/naive_aware task.dataset=split_cifar10 net=wrn10_2 loss.group_task_loss=True loss.alpha=0.5 task.custom_sampler=True task.beta=0.75 task.target=0,1,2,3,4 task.ood=[0],[1],[2],[3],[4] task.n=50 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hydra.launcher.n_jobs=10