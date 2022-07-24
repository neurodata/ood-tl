# CIFAR-10 T2 vs theta-T2 + SmallConv + Aug + Conventional Batches (Task Agnostic)
python singlehead_rotated_tasks.py --angles 0 10 45 90 135 --no-task_aware --no-tune_alpha --net smallconv --augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:0 --makefolder

# CIFAR-10 T2 vs theta-T2 + SmallConv + Aug + Target-Fraction Preserving Batches (Naive Task Aware)
python singlehead_rotated_tasks.py --angles 0 10 45 90 135 --task_aware --no-tune_alpha --net smallconv --augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:0 --makefolder

# CIFAR-10 T2 vs theta-T2 + SmallConv + No Aug + Conventional Batches (Task Agnostic)
python singlehead_rotated_tasks.py --angles 0 10 45 90 135 --no-task_aware --no-tune_alpha --net smallconv --no-augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:0 --makefolder

# CIFAR-10 T2 vs theta-T2 + SmallConv + No Aug + Target-Fraction Preserving Batches (Naive Task Aware)
python singlehead_rotated_tasks.py --angles 0 10 45 90 135 --task_aware --no-tune_alpha --net smallconv --no-augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:0 --makefolder

# CIFAR-10 T2 vs theta-T2 + SmallConv + Aug + Target-Fraction Preserving Batches (Optimal Task Agnostic)
python singlehead_rotated_tasks.py --angles 135 --task_aware --tune_alpha --net smallconv --augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:0 --makefolder

# CIFAR-10 T2 vs theta-T2 + SmallConv + No Aug + Target-Fraction Preserving Batches (Optimal Task Aware)
python singlehead_rotated_tasks.py --angles 135 --task_aware --tune_alpha --net smallconv --no-augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:0 --makefolder