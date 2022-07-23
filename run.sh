# python singlehead_dual_tasks.py --in_task 1 --out_task 5 --tune_alpha False
# python singlehead_dual_tasks.py --in_task 2 --out_task 1 3 4 5
# python singlehead_dual_tasks.py --in_task 1 --out_task 2 3 4 5
# python singlehead_dual_tasks.py --in_task 3 --out_task 1 2 4 5
# python singlehead_dual_tasks.py --in_task 4 --out_task 1 2 3 5
# python singlehead_dual_tasks.py --in_task 5 --out_task 1 2 3 4

python singlehead_dual_tasks.py --in_task 1 --out_task 5 --tast_aware False --tune_alpha False --net wrn --augment True --gpu cuda:3