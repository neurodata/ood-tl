# Related Work


**What are different types of distributions shifts?**

1. **Covariate-shift**: p(y|x) is same but p(x) changes. Re-weighting the the most popular strategy
    1. Doubly robust estimator: https://ojs.aaai.org/index.php/AAAI/article/download/9576/9435 balances the bias and variance bu picking an estimator that is close to the original hypothesis (low variance) but also re-weighted to adjust for the covariate shift.
    2. https://link.springer.com/chapter/10.1007/978-3-642-34106-9_14: Compute the reweighting terms using unlabeled data to compute the importance sampling ratios
2. **Domain generalization:** We want to train on distribution-A but we want to generalize to an “unseen” distribution “B”. This is done by hopefully capturing some invariance across the domains
    1. You can’t really generalize without additional assumptions (like a causal model, or some method to identify spurious correlations)
    2. IRM
    3. CORAL
3. **Domain Adaptation:** Usually refers to training on a source task with labeled data and data from another unlabeled target task, which we would like to generalize to. The two main variants are unsupervised and semi-supervised domain adaptation. Semi-supervised domain adaptation is really just transfer learning. 
    1. Ben-david’s works: https://cs.uwaterloo.ca/~shai/domain_adapt.pdf and https://link.springer.com/content/pdf/10.1007/s10994-009-5152-4.pdf https://proceedings.mlr.press/v9/david10a/david10a.pdf - 
    2. If we have multiple sources, then we can consider combining hypotheses train on each source - https://cs.nyu.edu/~mohri/pub/adap.pdf. Authors consider distribution weighted combinations instead of naive convex combinations which do not always work. 
    3. Transfer component analysis https://www.ijcai.org/Proceedings/09/Papers/200.pdf projects data into sub-space such that MMD is minimized and information to predict labels is retained. 
    4. https://arxiv.org/pdf/1505.07818.pdf - DANN (learn features which are same for both domains). This is inspired from minimizing the HΔH divergence in Ben-David. We learn features that are invariant to both domains.
    5. Adaptation can be tackled by re-weigting the samples (covariate shifts) or by learning a good representation. For more recent/better weighting schemes, see https://jmlr.csail.mit.edu/papers/volume20/15-192/15-192.pdf
    6. https://papers.nips.cc/paper/2021/file/ecf9902e0f61677c8de25ae60b654669-Paper.pdf:  Studied if HΔH is predictive of OOD generalization. In turns out that this isn’t the case, since there no longer exists a single hypothesis that generalizes on both the tasks. 
4. **Concept drift:** Over multiple time-steps, the distribution shifts/changes 
    1. https://arxiv.org/pdf/1205.4343.pdf  - Re-derives some of Ben-davids proofs but in a setting where the concept drifts over multiple time steps
    2. https://dl.acm.org/doi/abs/10.1145/130385.130412 - Bartlett’s paper where consecutive time-steps have a divergence with small L1 norm
    3. https://link.springer.com/content/pdf/10.1007/s10994-007-5003-0.pdf  - Online perceptron/hyper-plane tracking algo
5. **Sub-population shift:** We aim to perform well on a range of groups/domains/sub-populations that all occur in the train-set. We want to maximize the performance of the lowest performing group. The test set is only one of those sub-populations, i.e. Dtest​⊂Dtrain​. (and is usually the worst performing one)
    1. Group-DRO https://arxiv.org/pdf/1611.02041.pdf,  https://arxiv.org/abs/1911.08731 claims that heavy regularization usually works
6. **Data poisoning:** This is an adversarial setup where the dataset is corrupted with a few samples. The goal is to remove these samples since the poisoned samples will almost certainly hurt generalization.
    1. https://people.eecs.berkeley.edu/~tygar/papers/Machine_Learning_Security/asiaccs06.pdf
    2. https://arxiv.org/pdf/1206.6389.pdf - Poisoning on SVMs or linear hypothesis spaces: https://dl.acm.org/doi/pdf/10.1145/1143844.1143889  - mitigating poisoning by not relying on small set of features
    3.  https://arxiv.org/pdf/1703.01340.pdf, https://arxiv.org/pdf/1706.03691.pdf for poisoning on NNs.


**Other Theory around distribution shifts**


* If we want to identify if the distribution is different for two sets of samples, we can use https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf : A Kernel two-sample test
* Generalization of Gibbs algorithm: https://arxiv.org/abs/2107.13656
* Task-competition
* http://www.acad.bg/ebook/ml/The.MIT.Press.Dataset.Shift.in.Machine.Learning.Feb.2009.eBook-DDU.pdf


**Benchmarks:**

* WILDS: Has domain generalization and sub-populations shifts
* Domain Bed: For domain generalization https://arxiv.org/pdf/2007.01434.pdf
* BREEDS: https://arxiv.org/pdf/2008.04859.pdf - Subpopulation shift, try to evaluate on sub-populations that haven’t been seen before.


