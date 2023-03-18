# DomainNet Fine-grained categories
python3 train_singlehead.py -m seed=10 deploy=True device=cuda:1 tag=final-DN-RvQ-fine-grained-3 reps=1 task.target_env=real task.ood_env=quick task.dataset=domainnet task.n=5 task.m_n=0,0.5,1,2,3 hp.bs=32 hp.epochs=100 task.task_map=[[23,26,34,52,57,67,84,87,94,95,102,106,120,128,132,144,150,162,177,189,194,207,211,215,221,225,238,239,245,256,258,260,261,271,272,284,299,312,336,343]]

# [[19,29,37,39,41,50,54,55,57,61,76,79,81,83,89,91,145,148,157,164,177,188,191,210,257,263,266,278,282,286,287,289,292,306,309,312,319,320,323,341]]