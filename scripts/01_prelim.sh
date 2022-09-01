## THIS HAD EVAL BUG
## Neural net
# python3 train_singlehead.py seed=10 net=conv reps=1 deploy=True tag=01_prelim/nets
# python3 train_singlehead.py seed=10 net=wrn16_4 reps=1 deploy=True tag=01_prelim/nets
# python3 train_singlehead.py seed=10 net=wrn10_2 reps=1 deploy=True tag=01_prelim/nets

## hps
# python3 train_singlehead.py -m seed=10 net=conv,wrn16_4 reps=1 deploy=True tag=01_prelim/bs hp.bs=16,32,128
# python3 train_singlehead.py -m seed=10 net=wrn16_4 reps=1 deploy=True tag=01_prelim/bs hp.bs=8

# Epochs
# python3 train_singlehead.py -m seed=10 net=wrn16_4 reps=1 deploy=True tag=01_prelim/ep hp.epochs=250
# python3 train_singlehead.py -m seed=10 net=conv reps=1 deploy=True tag=01_prelim/ep hp.epochs=300 hp.lr=0.01 hp.bs=16

# Lr
# python3 train_singlehead.py -m seed=10 net=conv reps=1 deploy=True tag=01_prelim/lr hp.lr=0.1,0.05,0.02,0.01,0.005 hp.bs=16,32,64

# Samples
# python3 train_singlehead.py -m seed=10 reps=1 deploy=True tag=01_prelim/nsamples task.n=50,250,500,750,1000,1500,2000,5000


# WITHOUT EVAL BUG
# python3 train_singlehead.py seed=10 net=conv reps=10 deploy=True tag=01_prelim_v2/nets
# python3 train_singlehead.py seed=10 net=wrn16_4 reps=10 deploy=True tag=01_prelim_v2/nets
# python3 train_singlehead.py seed=10 net=wrn10_2 reps=10 deploy=True tag=01_prelim_v2/nets

# Lr
python3 train_singlehead.py -m seed=10 net=conv reps=1 deploy=True tag=01_prelim_v2/lr hp.lr=0.01 hp.bs=16,32,64,128

# Epochs
# python3 train_singlehead.py -m seed=10 net=conv reps=1 deploy=True tag=01_prelim/ep hp.epochs=300 hp.lr=0.01 hp.bs=16

# Samples
# python3 train_singlehead.py -m seed=10 reps=1 deploy=True tag=01_prelim/nsamples task.n=50,250,500,750,1000,1500,2000,5000
