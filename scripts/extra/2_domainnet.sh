# DomainNet Coarse-grained categories
python3 train_singlehead.py -m seed=10 deploy=True device=cuda:3 tag=final-DN-RvQ-coarse-grained-2 reps=1 task.target_env=real task.ood_env=quick task.dataset=domainnet task.n=10 task.m_n=0,0.5,1,2,3 hp.bs=32 hp.epochs=100 task.task_map=[[21,40,41,42,94,103,106,121,142,143,150,172,199,200,261,262,263,264,266,277,281,284,285,289,294,302,305,306,308,309,312,313,317,328,331,333,336,340,341,343]]

# [[12,15,32,34,36,58,61,71,81,83,96,105,110,111,112,113,115,131,139,148,150,151,156,157,165,166,169,172,188,206,213,216,227,246,274,291,292,299,314,319]]