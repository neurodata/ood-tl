# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/00_domain_net loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=domain_net task.n=25 task.ood=[0]


# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/01_domain_net loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=domain_net task.n=25 task.ood=[1]


# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/02_domain_net loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=domain_net task.n=25 task.ood=[2]


# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/03_domain_net loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=domain_net task.n=25 task.ood=[3]


# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/04_domain_net loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=domain_net task.n=25 task.ood=[4]


# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/05_domain_net loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5,6,7,8,9,10 task.augment=False task.dataset=domain_net task.n=25 task.ood=[5]

# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/00_domain_net2 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[0]
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/01_domain_net2 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[1]
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/02_domain_net2 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[2]
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/03_domain_net2 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[3]
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/04_domain_net2 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[4]
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/05_domain_net2 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[5]

# CUDA_VISIBLE_DEVICES=3 python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/00_domain_alpha1 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[0] loss.group_task_loss=True
# CUDA_VISIBLE_DEVICES=3 python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/01_domain_alpha1 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[1] loss.group_task_loss=True
# CUDA_VISIBLE_DEVICES=3 python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/02_domain_alpha1 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[2] loss.group_task_loss=True 
# CUDA_VISIBLE_DEVICES=3 python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/03_domain_alpha1 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[3] loss.group_task_loss=True
# CUDA_VISIBLE_DEVICES=3 python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/04_domain_alpha1 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[4] loss.group_task_loss=True
# CUDA_VISIBLE_DEVICES=3 python3 train_singlehead.py -m seed=10 reps=10 deploy=True tag=05_ood/05_domain_alpha1 loss.group_task_loss=False task.custom_sampler=False task.m_n=0,0.25,0.5,1,2,3,4,5 task.augment=False task.dataset=domain_net task.n=100 task.ood=[5] loss.group_task_loss=True

# notify "Domain net - naive task-aware"
