This folder contains the outcomes of the cifar10_SmallConv_singlehead_naive_beta experiment.

target instances are weighted by 0.5/beta and OOD instances by 0.5/(1-beta). beta = n/(n+m)

exp_name: "singlehead_dual_tasks"
dataset: "cifar10"
task_dict: {1:[0,1], 2:[2,3], 3:[4,5], 4:[6,7], 5:[8,9]}
in_task: 2
out_task: [1, 3, 4, 5]
net: "smallconv"
n: 100
m_n_ratio: [0, 1, 2, 3, 4, 5, 10, 20]
sample_scheme: False
hp: {'epochs': 100,'batch_size': 16, 'lr': 0.01, 'l2_reg': 0.00001}
reps: 10
tune_alpha: False
val_split: 0.15
save_folder: "./experiments/results/cifar10_SmallConv_singlehead_optimal"
