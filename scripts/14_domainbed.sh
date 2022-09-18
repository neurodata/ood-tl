# singlhead rotated mnist smallconv
# python3 train_singlehead.py -m seed=10 reps=10 deploy=False device=cuda:3 is_multihead=False tag=14_domainbed/rotated_mnist_smallconv task.dataset=rotated_mnist net=conv task.custom_sampler=False task.env=32,35,38,40,42 task.task_map=[[0,1,2,3,4,5,6,7,8,9]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10

# singlhead office homes wrn10-2 (7 classes)
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=False tag=14_domainbed/7_officehomes_wrn10-2 task.dataset=officehomes net=wrn10_2 task.custom_sampler=False task.env=C,P task.task_map=[[10,22,29,34,38,50,56]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,6,7 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10

# singlhead office homes wrn10-2 (5 classes)
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 is_multihead=False tag=14_domainbed/5_officehomes_wrn10-2 task.dataset=officehomes net=wrn10_2 task.custom_sampler=False task.env=C,P task.task_map=[[10,22,29,50,56]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,6,7,8 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10

# singlhead office homes wrn16-4 (7 classes)
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=False tag=14_domainbed/7_officehomes_wrn16-4 task.dataset=officehomes net=wrn16_4 task.custom_sampler=False task.env=C,P task.task_map=[[10,22,29,34,38,50,56]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,6,7 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10

# singlhead office homes wrn16-4 (5 classes)
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 is_multihead=False tag=14_domainbed/5_officehomes_wrn16-4 task.dataset=officehomes net=wrn16_4 task.custom_sampler=False task.env=C,P task.task_map=[[10,22,29,50,56]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,6,7,8 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10

# singlhead office homes wrn16-4 (5 classes) Target: Clip Art vs OOD: Real
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 is_multihead=False tag=14_domainbed/5_officehomes_CvR_wrn16-4 task.dataset=officehomes net=wrn16_4 task.custom_sampler=False task.target_env=C task.ood_env=R task.task_map=[[10,22,29,50,56]] task.target=0 task.ood=[0] task.n=10 task.m_n=0,1,2,3,4,5,6,7,8 task.augment=False hp.bs=16 hp.epochs=100 hydra.launcher.n_jobs=10