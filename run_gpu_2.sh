git pull

# T1 vs T5 + SmallConv + Aug + Traget-Fraction-Preserving Batches (bs = 128, e = 200)
python singlehead_dual_tasks.py --in_task 1 --out_task 5 --task_aware --tune_alpha --net smallconv --augment --epochs 100 --batch_size 128 --reps 10 --gpu cuda:2