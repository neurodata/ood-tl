# Evaluat the effect of distribution shift with Blurred CIFAR-10

# 3 reps
# python3 train_singlehead.py -m seed=10 reps=3 deploy=True tag=07_blurred_cifar10/task_agnostic net=conv task.dataset=blurred_cifar10 task.sigma=15 task.custom_sampler=False task.target=1 task.ood=[] task.m_n=0,1,2,3,4,5,10,20 task.augment=False

# final
python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=07_blurred_cifar10_rep10_2/task_agnostic net=conv task.dataset=blurred_cifar10 task.sigma=1,2,5,6,7,8,9,10,15 task.custom_sampler=False task.target=2 task.ood=[] task.m_n=0,1,2,3,4,5,10,20 task.augment=False