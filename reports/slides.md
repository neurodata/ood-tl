---
marp: true
theme: default
paginate: true
style: |
    section {
        justify-content: flex-start;
        --orange: #ed7d31;
        --left: #66c2a5;
        --right: #fc8d62;
        --source: #8da0cb;
        --target: #e78ac3;
    }
    img[alt~="center"] {
        display: block;
        margin: 0 auto;
    }
    img[alt~="icon"] {
        display: inline;
        margin: 0 0.125em;
        padding: 0;
        vertical-align: middle;
        height: 30px;
    }
    header {
        top: 0px;
        margin-top: auto;
    }
    .columns {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
    .social_icons {
        padding: 0em;
        display: inline-block;
        vertical-align: top;
    }

---


<style scoped> 
h1 {
    font-size:40px;
}
p {
    font-size: 24px;
}
</style>

# Out-of-Distribution Learning

___

## Gaussian Tasks Experiment

- Consider an in-distribution task that consists of two class conditional gaussians. 
- Now, consider an out-of-distribution task similar to the above task, but whose center is displaced by an amount $\Delta$.
- The amount $\Delta$ reflects the "similarity" between the two tasks.

![center w:500](figures/gausstask_fig.png)

___

## Gaussian Tasks Experiment

- We have access to $n$ samples from the in-distribution task, and $m$ samples from the out-of-distribution task.
- Using both the in-distribution and out-of-distribution samples, we train a classifier $h$ aimed at the in-distribution classification task.
- Let's denote the classification error of $h$ by $\mathbb{E}[L_{n, m, \Delta}]$.

___

## Gaussian Tasks Experiment

- Let $n$ be a small fixed constant. We hypothesize that, 
    - For very small $\Delta$, as we add more out-of-distribution data (as $m$ increases) the $\mathbb{E}[L_{n, m, \Delta}]$ would decrease. 
    - For moderately large $\Delta$, as we add more out-of-distribution data (as $m$ increases) the $\mathbb{E}[L_{n, m, \Delta}]$ would initially decrease and start increasing later. The initial decrease is due to the reduction in the variance of $h$. The later increase is due to the increase in bias of $h$ caused by the out-of-distribution samples. 
    - For very large $\Delta$, as we add more out-of-distribution data (as $m$ increases) the $\mathbb{E}[L_{n, m, \Delta}]$ would keep increasing.

___

## Gaussian Tasks Experiment

![center w:1000](figures/gaussian_task_analytical_plot.svg)

___

## Gaussian Tasks Experiment

- Number of replicates: 1000

![center w:525](figures/gaussian_tasks_sim_plot.svg)

___

## Gaussian Tasks Experiment

- Number of replicates: 1000

![center w:2000](figures/gaussian_tasks_rep_plot.svg)

___

## Bird vs. Cat & $\alpha$-Rotated Bird vs. Cat (Single-Head Network)

- Number of replicates: 10, Network: SmallConv

![center w:500](figures/rotated_BvC.svg)

---

## Bird vs. Cat & $\alpha$-Rotated Bird vs. Cat (Single-Head Network)

- Number of replicates: 10, Network: SmallConv

![center w:2000](figures/rotated_BvC_reps.svg)

___

## Task 2: Bird vs. Cat & Task 3: Deer vs. Dog (Single-Head Network)

- Number of replicates: 20, Network: SmallConv

![center w:950](figures/bridcat_deerdog.svg)

___

## Task 2: Bird vs. Cat & Task 3: Deer vs. Dog (Single-Head Network)

- Number of replicates: 20, Network: SmallConv, each model was trained for 100 epochs

![center w:900](figures/bridcat_deerdog_100epochs.svg)

___

## Task 2: Bird vs. Cat & Task 3: Deer vs. Dog (Multi-Head Network)

- Number of replicates: 20, Network: SmallConv

![center w:950](figures/cifar10_multihead_dual_tasks_T2_T3.svg)

___

## Task 2: Bird vs. Cat & Task 4: Frog vs. Horse (Multi-Head Network)

- Number of replicates: 20, Network: SmallConv

![center w:950](figures/cifar10_multihead_dual_tasks_T2_T4.svg)

___

## Task 2: Bird vs. Cat & Task 3: Deer vs. Dog (Multi-Head Network)

- Number of replicates: 10, Network: Wide Res-Net

![center w:890](figures/cifar10_wrn_multihead_dual_tasks_T2_T3.svg)
___

## Task 2: Bird vs. Cat & Task 4: Frog vs. Horse (Multi-Head Network)

- Number of replicates: 10, Network: Wide Res-Net

![center w:890](figures/cifar10_wrn_multihead_dual_tasks_T2_T4.svg)

___

## Task 4: Frog vs. Horse & Task 2: Bird vs. Cat (Multi-Head Network)

- Number of replicates: 10, Network: Wide Res-Net

![center w:890](figures/cifar10_wrn_multihead_dual_tasks_T4_T2.svg)

___

## CIFAR-10 Tasks (Multi-Head Network)

![center w:550](figures/cifar_wrn_multihead.svg)

___

## CIFAR-10 Tasks (Multi-Head Network)

![center w:550](figures/split_cifar10_task_matrix.png)

___

## CIFAR-10 Tasks (Multi-Head Network)

- Random sampling (large range of $m/n$ values)

![center w:510](figures/cifar_wrn_multihead_larger_m.svg)

<!-- ___

## Bivariate LDA Problem

![center w:800](figures/mulFLD_fig.png)

___

## Bivariate LDA Problem

- $X | Y = -1 \sim \mathcal{N}(-\mu_0, \Sigma)$ and $X | Y = +1 \sim \mathcal{N}(\mu_0, \Sigma)$ consititute the in-distribution where $\mu_0 = [\mu, 0]^\top$
- $X | Y = -1 \sim \mathcal{N}(-\mu_{\theta}, \Sigma)$ and $X | Y = +1 \sim \mathcal{N}(\mu_{\theta}, \Sigma)$ consititute the out-of-distribution where $\mu_{\theta} = [\mu \cos \theta, - \mu \sin \theta]^\top$
- Then, the estimated class means $\hat{\mu}_{-1}$ and $\hat{\mu}_{+1}$ are given by, 
    $$ \hat{\mu}_{-1} \sim \mathcal{N}\bigg( \bigg[ \frac{-\mu( n + m \cos \theta)}{n+m}, \frac{\mu m \sin \theta }{n+m} \bigg]^\top, \frac{1}{n+m} \Sigma \bigg) $$
    $$ \hat{\mu}_{+1}= - \hat{\mu}_{-1} \sim \mathcal{N}\bigg( \bigg[ \frac{\mu( n + m \cos \theta)}{n+m}, -\frac{\mu m \sin \theta }{n+m} \bigg]^\top, \frac{1}{n+m} \Sigma \bigg) $$

___

## Bivariate LDA Problem

- The LDA's classification rule is given by, 
    $$ g(x) = \text{sign} ( w \cdot x > c) $$ 
    where, 
$$ w = \Sigma^{-1} (\hat{\mu}_{+1} - \hat{\mu}_{-1}) = 2 \Sigma^{-1} \hat{\mu}_{+1} $$
$$ c = \frac{1}{2}(\hat{\mu}_{+1} + \hat{\mu}_{-1}) = 0 $$
- Therefore,  
    $$ g(x) = \text{sign} ( \hat{\mu}_{+1} \cdot x > 0) $$ 
___

## Bivariate LDA Problem

- If $\mu = 1$ and $\Sigma = I$,
    $$ \hat{\mu}_{+1} \sim \mathcal{N}\bigg( \bigg[ \frac{( n + m \cos \theta)}{n+m}, -\frac{m \sin \theta }{n+m} \bigg]^\top, \frac{1}{n+m} I \bigg) $$
    $$ x | y = -1 \sim f_{-1} = \mathcal{N}\big( [-1, 0]^\top, \Sigma) \big) $$
    $$ x | y = +1 \sim f_{+1} = \mathcal{N}\big( [1, 0]^\top, \Sigma) \big) $$
- Hence, the error $L(\hat{\mu}_{+1})$ is given by, 
    $$ L(\hat{\mu}_{+1}) = \mathbb{P}_{x \sim f_{-1}}[ \hat{\mu}_{+1} \cdot x > 0 ] + \mathbb{P}_{x \sim f_{+1}}[ \hat{\mu}_{+1} \cdot x < 0 ] $$ 
- Therefore, 
    $$ \mathbb{E}[L_{m, n, \theta}] = \mathbb{E}_{\hat{\mu}_{+1}}[L(\hat{\mu}_{+1})] $$ -->

<!-- ___

## Bivariate LDA

![center w:800](figures/bivariate_lda.png)

___

## Bivariate LDA

- In-distribution samples (class 0): $X_1, \dots, X_{n/2} \sim \mathcal{N}(\mu, I)$
- In-distribution samples (class 1): $X_{n/2+1}, \dots, X_n \sim \mathcal{N}(-\mu, I)$
- Out-of-distribution samples (class 0): $X_{n+1}, \dots, X_{n+m/2} \sim \mathcal{N}(\mu + \Delta , I)$
- In-distribution samples (class 1): $X_{n+m/2+1}, \dots, X_{n+m} \sim \mathcal{N}(-\mu + \Delta, I)$

Consider the class 0 which is comprised of $n/2$ in-distribution and $m/2$ OOD samples. Let $\bar{\mu}_0$ and $\bar{\Sigma}_0$ be sample mean and sample covariance matrix of class 0.

$$ \bar{\mu}_0 \sim \mathcal{N}\big( \mu + \frac{m}{n+m}\Delta, \frac{1}{n+m}I \big) $$

Similarly, 

$$ \bar{\mu}_1 \sim \mathcal{N}\big( -\mu + \frac{m}{n+m}\Delta, \frac{1}{n+m}I \big) $$

___

## Bivariate Single-Head LDA

![center w:800](figures/bivariate_singlehead_lda.png)

___

## Bivariate Single-Head LDA

The projection vector of the LDA is given by, (Assuming $\bar{\Sigma}_0 = \bar{\Sigma}_1 = \bar{\Sigma}$),

$$ w = \bar{\Sigma}^{-1} (\bar{\mu}_1 - \bar{\mu}_0) $$

The threshold is given by, 

$$ c = w \cdot \frac{(\bar{\mu}_0 + \bar{\mu}_1)}{2} $$

Then the LDA classification rule is given by, 

$$ g(x) = \mathbb{I}(w \cdot x > c) $$
___

## Bivariate Single-Head LDA

The risk $L_{n, m, \Delta}$ on the in-distribution data is given by, 

$$L_{n, m, \Delta} = \mathbb{E}_{f_{w,c}}[L(w, c)] $$

$$ L(w, c) = \mathbb{P}_{x \sim f_0}[ w \cdot x > c ] + \mathbb{P}_{x \sim f_1}[ w \cdot x < c] $$ 

___

## Bivariate Multi-Head LDA

![center w:800](figures/bivariate_multihead_lda.png)

___

## Bivariate Multi-Head LDA

The shared projection vector of the LDA is given by, (Assuming $\bar{\Sigma}_0 = \bar{\Sigma}_1 = \bar{\Sigma}$),

$$ w = \bar{\Sigma}^{-1} (\bar{\mu}_1 - \bar{\mu}_0) $$

The task-specific thresholds are given by, 

$$ c_{in} = w \cdot \frac{(\bar{\mu}_{0,in} + \bar{\mu}_{1,in})}{2} ;\quad \bar{\mu}_{0,in} \sim \mathcal{N}(\mu, I), \bar{\mu}_{1,in} \sim \mathcal{N}(-\mu, I) $$
$$ c_{out} = w \cdot \frac{(\bar{\mu}_{0,out} + \bar{\mu}_{1,out})}{2} ; \quad \bar{\mu}_{0,out} \sim \mathcal{N}(\mu+\Delta, I), \bar{\mu}_{1,out} \sim \mathcal{N}(-\mu+\Delta, I) $$

Then the LDA classification rule for in-distribution data is given by, 
$$ g(x) = \mathbb{I}(w \cdot x > c_{in}) $$

___

## Bivariate Single-Head LDA

The risk $L_{n, m, \Delta}$ on the in-distribution data is given by, 

$$L_{n, m, \Delta} = \mathbb{E}_{f_{w,c_{in}}}[L(w, c_{in})] $$

$$ L(w, c_{in}) = \mathbb{P}_{x \sim f_0}[ w \cdot x > c_{in} ] + \mathbb{P}_{x \sim f_1}[ w \cdot x < c_{in}] $$  -->