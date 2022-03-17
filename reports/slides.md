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

---

## Bird vs. Cat & $\alpha$-Rotated Bird vs. Cat

- Number of replicates: 10

![center w:500](figures/rotated_BvC.svg)

---

## Bird vs. Cat & $\alpha$-Rotated Bird vs. Cat (Replicates)

- Number of replicates: 10

![center w:2000](figures/rotated_BvC_reps.svg)

___

## Task 3 (Bird vs. Cat) & Task 3 (Deer vs. Dog)

- Number of replicates: 20

![center w:950](figures/bridcat_deerdog.svg)

___

## Task 3 (Bird vs. Cat) & Task 3 (Deer vs. Dog)

- Number of replicates: 20, each model was trained for 100 epochs

![center w:950](figures/bridcat_deerdog_100epochs.svg)

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