# (Naive Task Aware) T1 vs T5 + SmallConv + Aug + Traget-Fraction-Preserving Batches (bs = 128, e = 100)
# python singlehead_dual_tasks.py --in_task 1 --out_task 5 --task_aware --no-tune_alpha --net smallconv --augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:3 --makefolder

# (Task Agnostic) T1 vs T5 + SmallConv + Aug + Conventional Batches (bs = 128, e = 100) ()
python singlehead_dual_tasks.py --in_task 1 --out_task 5 --no-task_aware --no-tune_alpha --net smallconv --augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:3 --makefolder