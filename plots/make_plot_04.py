import matplotlib.pyplot as plt
import numpy as np
import glob




def plot_04_01():
    mval = []
    acc = []
    std = []

    for fname in glob.glob("./logs/04_ood/task_agnostic/*"):
        with open(fname, "r") as fp:
            cont = fp.readlines()

        cfg = eval(cont[0])
        err = eval(cont[-1])

        mval.append(cfg['task']['m_n'])
        acc.append(err['avg_err'])
        std.append(err['std_err'])

    print(mval)
    print(acc)
    acc = np.array(acc)
    mval = np.array(mval)
    std = np.array(std)

    ind = np.argsort(mval)
    acc = acc[ind] * 100
    mval = mval[ind]
    std = std[ind] * 100

    plt.errorbar(mval, acc, yerr=std)
    plt.xlabel('m/n with n=100')
    plt.ylabel('Accuracy')
    plt.savefig('plots/imgs/04_01.png',
                bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_04_01()
