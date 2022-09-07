# Evaluat the effect of distribution shift with Blurred CIFAR-10

# 3 reps
# python3 train_singlehead.py -m seed=10 reps=3 deploy=True tag=07_blurred_cifar10/task_agnostic net=conv task.dataset=blurred_cifar10 task.sigma=15 task.custom_sampler=False task.target=1 task.ood=[] task.m_n=0,1,2,3,4,5,10,20 task.augment=False

# final
# python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:3 tag=07_blurred_cifar10_conv/task_agnostic net=conv task.dataset=blurred_cifar10 task.sigma=0 task.custom_sampler=False task.target=0,1,2,3,4 task.ood=[] task.m_n=0,1,2,3,4,5,10,20 task.augment=False

python3 train_singlehead.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=07_blurred_cifar10_wrn/task_agnostic net=wrn10_2 task.dataset=blurred_cifar10 task.sigma=0 task.custom_sampler=False task.target=1 task.ood=[] task.m_n=0,1,2,3,4,5,10,20 task.augment=False

## run the sigma = 0 case for wrn!!!!