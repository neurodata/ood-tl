# python singlehead_dual_tasks.py --in_task 1 --out_task 5 --tune_alpha False
# python singlehead_dual_tasks.py --in_task 2 --out_task 1 3 4 5
# python singlehead_dual_tasks.py --in_task 1 --out_task 2 3 4 5
# python singlehead_dual_tasks.py --in_task 3 --out_task 1 2 4 5
# python singlehead_dual_tasks.py --in_task 4 --out_task 1 2 3 5
# python singlehead_dual_tasks.py --in_task 5 --out_task 1 2 3 4

python singlehead_dual_tasks.py --in_task 1 --out_task 5 --no-task_aware --no-tune_alpha --net wrn --augment --epochs 200 --batch_size 64 --reps 5 --gpu cuda:3

python singlehead_dual_tasks.py --in_task 1 --out_task 5 --no-task_aware --no-tune_alpha --net wrn --no-augment --epochs 200 --batch_size 64 --reps 5 --gpu cuda:2

python singlehead_dual_tasks.py --in_task 1 --out_task 5 --no-task_aware --no-tune_alpha --net smallconv --augment --epochs 200 --batch_size 64 --reps 5 --gpu cuda:1