# Optimization Behavior of Riemannian Optimizers for Hyperbolic WordNet Subtree Classification

## 1. Project Overview

This project studies the optimization behavior of different optimizers when training a hyperbolic classifier on WordNet subtree classification.

The key research question is:

> When training a hyperbolic multinomial logistic regression classifier on WordNet subtree classification, do Riemannian optimizers provide better convergence, stability, and generalization than projected Euclidean optimizers?

This project should not focus on inventing a new hyperbolic neural network architecture. Instead, it should isolate the optimization problem:

- Fix the dataset.
- Fix the model family.
- Change only the optimizer.
- Compare convergence, stability, final F1 score, parameter norm behavior, projection/clipping behavior, and learned decision boundaries.

The final implementation should produce all quantitative tables and qualitative figures needed for a 3-page Optimization for Machine Learning mini-project report.

---

## 2. Motivation and Background

WordNet noun hierarchy is a tree-like semantic structure. Hyperbolic space, especially the Poincaré ball model, is suitable for representing hierarchical data because distances expand exponentially toward the boundary.

Previous work such as **Hyperbolic Neural Networks** by Ganea, Bécigneul, and Hofmann used pre-trained Poincaré embeddings of WordNet nodes and trained classifiers for WordNet subtree classification. They compared:

1. Hyperbolic multinomial logistic regression;
2. Euclidean logistic regression directly on Poincaré coordinates;
3. Euclidean logistic regression after mapping embeddings to the tangent space at the origin using the logarithmic map.

Their work mainly compared hyperbolic and Euclidean classifiers. This project extends that setting by focusing on the optimizer:

> Given the same hyperbolic classifier, how do different optimizers affect the training process?

This is an optimization-focused project because parameters in a hyperbolic classifier may live on the Poincaré ball, which is a Riemannian manifold rather than an unconstrained Euclidean vector space. Therefore, the optimizer should ideally respect the geometry of the manifold.

---

## 3. Data and Repositories

### 3.1 Main Data Source

Use WordNet noun hierarchy and Poincaré embeddings.

Recommended repository:

```text
https://github.com/facebookresearch/poincare-embeddings
```

This repository can generate WordNet noun hierarchy transitive closure data and train Poincaré embeddings.

Relevant usage:

```bash
cd wordnet
python transitive_closure.py
```

This generates the transitive closure of the full noun hierarchy and the mammals subtree.

Then train embeddings using scripts such as:

```bash
./train-mammals.sh
./train-nouns.sh
```

For this project, the implementation can either:

1. use pre-trained Poincaré embeddings if available;
2. train Poincaré embeddings using the Facebook Research repo;
3. provide a lightweight fallback that trains a small WordNet / mammals subset embedding.

The preferred option is to avoid spending too much time re-training embeddings and focus on subtree classifier optimization.

---

### 3.2 Hyperbolic MLR Reference Repository

Recommended reference repository:

```text
https://github.com/dalab/hyperbolic_nn
```

This is the source code for Ganea et al., **Hyperbolic Neural Networks**.

Use this repository mainly as a reference for:

- hyperbolic MLR formula;
- Poincaré hyperplane decision boundary;
- MLR visualization;
- numerical stability tricks.

Important note:

Do not build the main project directly on this repository because it uses old dependencies:

```text
Python 3.5
TensorFlow 1.8
```

Instead, reimplement the required parts in modern PyTorch.

---

### 3.3 Visualization Reference Repository

Optional reference repository:

```text
https://github.com/dalab/hyperbolic_cones
```

This repository is useful for:

- WordNet / mammals subtree visualization;
- 2D Poincaré disk plotting;
- qualitative visualization of hierarchical embeddings.

Use it only as a reference. The main project should be implemented in PyTorch.

---

### 3.4 Riemannian Optimization Library

Use Geoopt:

```text
https://github.com/geoopt/geoopt
```

Install:

```bash
pip install geoopt
```

Geoopt provides:

- `geoopt.PoincareBall`;
- `geoopt.ManifoldParameter`;
- `geoopt.optim.RiemannianSGD`;
- `geoopt.optim.RiemannianAdam`;
- projection, retraction, exponential map, logarithmic map, and Riemannian gradient utilities.

This should be the main library for Riemannian optimization.

---

## 4. Task Definition

The task is binary WordNet subtree classification.

Given a selected subtree root, classify every WordNet node as:

- **positive**: the node belongs to the subtree rooted at the selected concept;
- **negative**: the node does not belong to that subtree.

Candidate subtree roots:

```text
ANIMAL.N.01
MAMMAL.N.01
GROUP.N.01
WORKER.N.01
```

Example:

If the subtree root is `MAMMAL.N.01`, nodes such as `dog`, `cat`, and `horse` are positive examples, while unrelated nodes such as `chair`, `tree`, and `vehicle` are negative examples.

Use the following split:

```text
positive nodes: 80% train, 20% test
negative nodes: 80% train, 20% test
```

Use stratified splitting and fixed random seeds.

---

## 5. Models to Implement

Implement three classifier families.

---

### 5.1 Direct Euclidean Logistic Regression

Input:

```text
Poincaré embedding coordinates x
```

Treat `x` as an ordinary Euclidean vector.

Model:

```text
logit = w^T x + b
```

Loss:

```text
binary cross entropy with logits
```

Optimizers:

```text
Adam
SGD
```

This is a baseline that ignores hyperbolic geometry.

---

### 5.2 Log-map Euclidean Logistic Regression

Map Poincaré embeddings to tangent space at the origin:

```text
z = log_0(x)
```

Then train Euclidean logistic regression:

```text
logit = w^T z + b
```

This baseline partially respects the geometry because it uses the tangent space representation.

Optimizers:

```text
Adam
SGD
```

---

### 5.3 Hyperbolic Logistic Regression / Hyperbolic MLR

Implement binary hyperbolic MLR on the Poincaré ball.

The model should learn a hyperbolic decision boundary represented by a Poincaré hyperplane.

A hyperbolic hyperplane is parameterized by:

```text
p ∈ Poincaré ball
a ∈ T_p B
```

For implementation simplicity, parameterize `a` through a Euclidean tangent vector at the origin and use parallel transport when needed, or follow Geoopt-compatible parameterization.

The binary logit should be proportional to the signed distance from a point `x` to the learned hyperbolic hyperplane.

Important implementation goals:

- support dimensions 2, 5, and 10;
- support batch computation;
- avoid numerical instability near the ball boundary;
- include clipping / projection where necessary;
- log projection and clipping events.

---

## 6. Optimizers to Compare

The main experimental variable is the optimizer for the hyperbolic classifier.

---

### 6.1 Projected SGD

Implementation:

1. Perform ordinary Euclidean SGD update.
2. Project parameters back inside the Poincaré ball.

Track:

```text
number of projection events
parameter norm
training loss
test F1
```

---

### 6.2 Projected Adam

Implementation:

1. Perform ordinary Adam update in ambient Euclidean coordinates.
2. Project parameters back inside the Poincaré ball.

Expected behavior:

- fast early convergence;
- possible instability near the boundary;
- frequent projection or clipping.

---

### 6.3 Riemannian SGD

Use Geoopt:

```python
geoopt.optim.RiemannianSGD
```

Expected behavior:

- slower but more geometrically consistent updates;
- smoother parameter trajectories;
- fewer unstable boundary interactions.

---

### 6.4 Riemannian Adam

Use Geoopt:

```python
geoopt.optim.RiemannianAdam
```

Expected behavior:

- faster than Riemannian SGD;
- more stable than projected Adam;
- potentially best tradeoff between convergence speed and final F1.

---

## 7. Metrics

The implementation should report both classification metrics and optimization metrics.

---

### 7.1 Classification Metrics

For each run, report:

```text
test accuracy
test precision
test recall
test F1
```

Main metric:

```text
test F1
```

F1 is important because subtree classification can be imbalanced.

---

### 7.2 Optimization Metrics

For each epoch, log:

```text
train loss
test loss
test F1
gradient norm
average parameter norm
maximum parameter norm
projection count
clipping count
runtime per epoch
```

Save logs as:

```text
results/logs/{subtree}_{dim}_{model}_{optimizer}_seed{seed}.csv
```

---

## 8. Qualitative Visualizations

The project must include intuitive qualitative comparisons, not only tables.

---

### 8.1 Decision Boundary Visualization

For 2D embeddings, plot the Poincaré disk.

For each selected subtree, show:

```text
positive nodes
negative nodes
misclassified nodes
learned decision boundary
```

Compare:

```text
Direct Euclidean LR + Adam
Log-map Euclidean LR + Adam
Hyperbolic MLR + Projected Adam
Hyperbolic MLR + Riemannian SGD
Hyperbolic MLR + Riemannian Adam
```

Required output:

```text
results/figures/boundary_{subtree}_dim2.png
```

Purpose:

Show whether the hyperbolic decision boundary better matches the WordNet subtree structure and whether different optimizers learn more stable or more reasonable boundaries.

---

### 8.2 Optimizer Trajectory Visualization

For 2D hyperbolic MLR, record the trajectory of the learned hyperbolic parameter `p` across epochs.

Plot in Poincaré disk:

```text
Projected Adam trajectory
Riemannian SGD trajectory
Riemannian Adam trajectory
```

Required output:

```text
results/figures/trajectory_{subtree}_dim2.png
```

Purpose:

Show whether some optimizers move too aggressively toward the boundary, oscillate, or follow smoother trajectories.

---

### 8.3 Norm-F1 Relationship

For each optimizer, plot:

```text
epoch vs average parameter norm
epoch vs test F1
```

Use either two y-axes or two aligned plots.

Required output:

```text
results/figures/norm_f1_{subtree}_dim{dim}.png
```

Purpose:

Study whether moving toward the Poincaré ball boundary correlates with better classification performance.

---

### 8.4 Loss and F1 Curves

Plot:

```text
epoch vs train loss
epoch vs test F1
```

Required output:

```text
results/figures/loss_curve_{subtree}_dim{dim}.png
results/figures/f1_curve_{subtree}_dim{dim}.png
```

Purpose:

Compare convergence speed and stability.

---

## 9. Experimental Matrix

Recommended full experiment:

```text
subtrees = [ANIMAL.N.01, MAMMAL.N.01, GROUP.N.01, WORKER.N.01]
dims = [2, 5, 10]
seeds = [0, 1, 2]
```

Models and optimizers:

```text
Direct Euclidean LR + Adam
Log-map Euclidean LR + Adam
Hyperbolic MLR + Projected SGD
Hyperbolic MLR + Projected Adam
Hyperbolic MLR + Riemannian SGD
Hyperbolic MLR + Riemannian Adam
```

Total number of runs:

```text
4 subtrees × 3 dims × 3 seeds × 6 settings = 216 runs
```

If time is limited, implement the smaller experiment first:

```text
subtrees = [MAMMAL.N.01, GROUP.N.01]
dims = [2, 5]
seeds = [0, 1, 2]
settings = [
    Direct Euclidean LR + Adam,
    Log-map Euclidean LR + Adam,
    Hyperbolic MLR + Projected Adam,
    Hyperbolic MLR + Riemannian SGD,
    Hyperbolic MLR + Riemannian Adam
]
```

Small version:

```text
2 subtrees × 2 dims × 3 seeds × 5 settings = 60 runs
```

The small version is enough for debugging and for a first report draft.

---

## 10. Expected Results

Expected observations:

1. **Projected Adam may converge quickly but become unstable near the Poincaré boundary.**

2. **Riemannian SGD may converge more slowly but produce smoother and more stable parameter trajectories.**

3. **Riemannian Adam may offer a good tradeoff between speed and stability.**

4. **The advantage of hyperbolic optimization should be more visible in low dimensions, especially dimension 2 and 5.**

5. **The parameter norm trajectory may correlate with test F1.**

6. **Even when final F1 scores are similar, decision boundary visualizations and optimizer trajectories may reveal different optimization behavior.**

---

## 11. Code Structure

Implement the project with the following structure:

```text
project/
├── README.md
├── requirements.txt
├── run.py
├── configs/
│   ├── default.yaml
│   ├── small_experiment.yaml
│   └── full_experiment.yaml
│
├── data/
│   ├── raw/
│   ├── embeddings/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── __init__.py
│   ├── data_wordnet.py
│   ├── poincare_ops.py
│   ├── models.py
│   ├── optimizers.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│
├── scripts/
│   ├── prepare_wordnet.sh
│   ├── train_embeddings.sh
│   ├── run_small.sh
│   ├── run_full.sh
│   └── plot_all.sh
│
└── results/
    ├── logs/
    ├── tables/
    ├── figures/
    └── checkpoints/
```

---

## 12. File Responsibilities

### `src/data_wordnet.py`

Responsibilities:

- load WordNet node list and edges;
- load Poincaré embeddings;
- identify subtree membership;
- create positive and negative labels;
- create train/test splits;
- support subtree roots:

```text
ANIMAL.N.01
MAMMAL.N.01
GROUP.N.01
WORKER.N.01
```

Expected functions:

```python
load_embeddings(path)
load_wordnet_edges(path)
get_subtree_nodes(root, edges)
make_subtree_dataset(root, embeddings, edges)
train_test_split_subtree(labels, seed, test_size=0.2)
```

---

### `src/poincare_ops.py`

Responsibilities:

- implement or wrap Poincaré operations;
- use Geoopt where possible;
- provide numerically stable functions.

Expected functions:

```python
mobius_add(x, y, c=1.0)
poincare_distance(x, y, c=1.0)
expmap0(v, c=1.0)
logmap0(x, c=1.0)
project_to_ball(x, c=1.0, eps=1e-5)
egrad_to_rgrad(x, grad, c=1.0)
```

If Geoopt is used directly, keep this file as a thin wrapper for clarity and logging.

---

### `src/models.py`

Implement:

```python
EuclideanLogisticRegression
LogMapEuclideanLogisticRegression
HyperbolicMLR
```

The `HyperbolicMLR` class should:

- support binary classification;
- learn hyperbolic hyperplane parameters;
- compute signed hyperbolic logits;
- expose parameters for trajectory and norm logging.

---

### `src/optimizers.py`

Implement optimizer factory:

```python
make_optimizer(model, optimizer_name, lr, weight_decay=0.0)
```

Supported optimizers:

```text
adam
sgd
projected_sgd
projected_adam
rsgd
radam
```

For projected optimizers:

- use PyTorch optimizer;
- after each update, project manifold parameters back inside the Poincaré ball;
- count projection events.

For Riemannian optimizers:

- use Geoopt optimizers.

---

### `src/train.py`

Responsibilities:

- run one training configuration;
- log all metrics;
- save checkpoint;
- save epoch-wise CSV.

Expected CLI options:

```bash
python -m src.train \
  --subtree MAMMAL.N.01 \
  --dim 2 \
  --model hyperbolic_mlr \
  --optimizer radam \
  --seed 0 \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.001
```

---

### `src/evaluate.py`

Responsibilities:

- compute test accuracy, precision, recall, F1;
- aggregate results across seeds;
- generate summary CSV / LaTeX table.

Expected outputs:

```text
results/tables/main_results.csv
results/tables/main_results_latex.txt
```

---

### `src/visualize.py`

Responsibilities:

Generate all figures:

```text
decision boundary plots
optimizer trajectory plots
loss curves
F1 curves
norm-F1 plots
bar plots with error bars
```

Expected CLI:

```bash
python -m src.visualize --config configs/small_experiment.yaml
```

---

### `run.py`

Top-level reproducibility script.

Expected usage:

```bash
python run.py --config configs/small_experiment.yaml
python run.py --config configs/full_experiment.yaml
```

It should:

1. prepare data if needed;
2. run all experiments;
3. aggregate results;
4. generate figures.

---

## 13. Requirements

Create `requirements.txt`:

```text
torch
numpy
scipy
scikit-learn
pandas
matplotlib
networkx
nltk
geoopt
pyyaml
tqdm
```

Optional:

```text
seaborn
```

But if possible, use only Matplotlib for plots.

---

## 14. README Requirements

The README should include:

1. project overview;
2. installation instructions;
3. data preparation instructions;
4. how to run small experiment;
5. how to run full experiment;
6. explanation of output files;
7. how to reproduce paper figures;
8. references.

Example:

```bash
conda create -n hypopt python=3.10
conda activate hypopt
pip install -r requirements.txt
python run.py --config configs/small_experiment.yaml
```

---

## 15. Logging Format

Each run should save a CSV with columns:

```text
epoch
train_loss
test_loss
test_accuracy
test_precision
test_recall
test_f1
grad_norm
avg_param_norm
max_param_norm
projection_count
clipping_count
runtime_sec
```

Example path:

```text
results/logs/MAMMAL.N.01_dim2_hyperbolic_mlr_radam_seed0.csv
```

---

## 16. Final Tables to Generate

### Table 1: Main quantitative comparison

Columns:

```text
Subtree
Dim
Model
Optimizer
Test F1 Mean
Test F1 Std
Test Accuracy Mean
Final Loss Mean
Avg Param Norm
Projection Count
```

### Table 2: Optimizer stability

Columns:

```text
Subtree
Dim
Optimizer
Seed Variance
Mean Projection Count
Mean Grad Norm
Runtime per Epoch
```

---

## 17. Final Figures to Generate

Required:

```text
results/figures/main_f1_barplot.png
results/figures/loss_curve_MAMMAL_dim2.png
results/figures/f1_curve_MAMMAL_dim2.png
results/figures/norm_f1_MAMMAL_dim2.png
results/figures/boundary_MAMMAL_dim2.png
results/figures/trajectory_MAMMAL_dim2.png
```

Optional:

```text
results/figures/boundary_GROUP_dim2.png
results/figures/trajectory_GROUP_dim2.png
results/figures/projection_count_barplot.png
```

---

## 18. Numerical Stability Requirements

Hyperbolic operations can be unstable near the Poincaré ball boundary.

Use:

```text
eps = 1e-5
max_norm = (1 - eps) / sqrt(c)
```

Whenever a parameter or embedding exceeds the allowed norm, project it back.

For inverse hyperbolic tangent:

```text
atanh_input = clamp(atanh_input, max=1 - 1e-5)
```

For very small norms:

```text
norm = clamp(norm, min=1e-15)
```

Track clipping events.

---

## 19. Minimum Viable Implementation

If time is limited, implement this first:

```text
Subtrees:
MAMMAL.N.01
GROUP.N.01

Dimensions:
2
5

Models:
Direct Euclidean LR + Adam
Log-map Euclidean LR + Adam
Hyperbolic MLR + Projected Adam
Hyperbolic MLR + Riemannian SGD
Hyperbolic MLR + Riemannian Adam

Seeds:
0, 1, 2

Epochs:
30

Batch size:
16
```

This is sufficient for:

- main F1 table;
- loss curves;
- norm-F1 curves;
- 2D decision boundary visualization;
- optimizer trajectory visualization.

---

## 20. Success Criteria

The implementation is complete if:

1. `python run.py --config configs/small_experiment.yaml` runs end-to-end.
2. All logs are saved under `results/logs/`.
3. Aggregated results are saved under `results/tables/`.
4. All required figures are saved under `results/figures/`.
5. At least three optimizers are compared on hyperbolic MLR.
6. At least one 2D decision boundary plot is generated.
7. At least one optimizer trajectory plot is generated.
8. The README explains how to reproduce the results.
9. The code is modular enough to add new optimizers or subtrees.

---

## 21. Suggested Report Story

The final report can follow this structure.

### Introduction

Hyperbolic geometry is suitable for hierarchical data, but training hyperbolic models requires optimization on a Riemannian manifold. We study how optimizer choice affects hyperbolic subtree classification.

### Methods

We compare Euclidean baselines, log-map Euclidean baselines, and hyperbolic MLR. For hyperbolic MLR, we compare projected Euclidean optimizers and Riemannian optimizers.

### Experiments

Dataset: WordNet subtree classification.

Metrics:

```text
F1
loss curves
parameter norm
projection count
trajectory stability
```

Subtrees:

```text
MAMMAL.N.01
GROUP.N.01
optionally ANIMAL.N.01 and WORKER.N.01
```

### Results

Report:

- main F1 table;
- convergence curves;
- norm-F1 curves;
- decision boundary visualizations;
- optimizer trajectory visualizations.

### Discussion

Analyze:

- whether Riemannian optimizers are more stable;
- whether projected Adam converges fast but has boundary instability;
- whether parameter norm behavior explains performance;
- whether low-dimensional hyperbolic classifiers benefit most.

### Conclusion

The project should conclude whether respecting the Poincaré ball geometry during optimization improves training behavior and classification performance.

---

## 22. References

Use these references in the final report:

1. O.-E. Ganea, G. Bécigneul, and T. Hofmann, “Hyperbolic Neural Networks,” NeurIPS, 2018.
2. O.-E. Ganea, “Non-Euclidean Neural Representation Learning of Words, Entities and Hierarchies,” Doctoral Thesis, ETH Zurich, 2019.
3. M. Nickel and D. Kiela, “Poincaré Embeddings for Learning Hierarchical Representations,” NeurIPS, 2017.
4. S. Bonnabel, “Stochastic Gradient Descent on Riemannian Manifolds,” IEEE Transactions on Automatic Control, 2013.
5. D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015.
6. G. Bécigneul and O.-E. Ganea, “Riemannian Adaptive Optimization Methods,” ICLR, 2019.
7. M. Kochurov, R. Karimov, and S. Kozlukov, “Geoopt: Riemannian Optimization in PyTorch,” 2020.
