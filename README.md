# Hyperbolic WordNet Optimizer Experiments

This project studies how optimizer choice affects hyperbolic WordNet subtree classification. The final experiment follows a published WordNet subtree setup while focusing on optimization behavior: the dataset, pretrained Poincare embeddings, and classifier family are fixed, and the optimizer is varied.

The central research question is:

```text
When training a Poincare-hyperplane hyperbolic multinomial logistic regression classifier on WordNet subtree classification, do Riemannian optimizers provide better convergence, stability, and generalization than projected Euclidean optimizers?
```

The project is not intended to introduce a new architecture. It isolates the optimization problem on the Poincare ball and compares convergence, final F1, precision/recall, parameter norms, projection/clipping events, runtime, learned 2D decision boundaries, and optimizer trajectories.

## Motivation

WordNet's noun hierarchy is tree-like: semantic categories branch into increasingly specific descendants. Hyperbolic geometry is well suited to such data because volume grows exponentially with radius, matching the expansion of hierarchical trees. The Poincare ball model represents high-level concepts near the origin and more specific concepts closer to the boundary.

Prior work showed that hyperbolic multinomial logistic regression can classify WordNet subtrees from pretrained Poincare embeddings. This project uses that setting to ask a more optimization-centered question: once the hyperbolic classifier is fixed, how much do projected Euclidean and Riemannian optimizers differ in accuracy, stability, and geometry-aware behavior?

## Environments

Use two separate conda environments:

```bash
# Project classifier experiments
conda create -n hypopt python=3.10 -y
conda activate hypopt
pip install -r requirements.txt
```

```bash
# facebookresearch/poincare-embeddings training/conversion
conda activate poincare-modern
```

`poincare-modern` is for the legacy embedding code under `poincare-embeddings/`. `hypopt` is for this project: subtree extraction, classifier training, tables, and figures.

## Repository Layout

```text
configs/                 # Experiment configs; full_experiment.yaml is the strict final config
scripts/                 # Reproducible RunAI, conversion, and plotting entry points
src/                     # Dataset loading, Poincare-hyperplane models, optimizers, training, evaluation, visualization
data/processed/          # Shared WordNet edge list
data/embeddings/         # Converted dim-specific Poincare embeddings
poincare-embeddings/     # Nickel and Kiela embedding trainer, with RunAI wrapper
results_wordnet_dims/      # Final strict experiment outputs used for the report
logs/                    # RunAI stdout/stderr logs for embedding and classifier jobs
```

The final report should use `results_wordnet_dims/`. The older single-checkpoint prototype run is documented below only as historical context.

### External Embedding Repository

The `facebookresearch/poincare-embeddings` code is treated as an optional external sub-repository/workspace dependency. It is useful when retraining WordNet noun embeddings or regenerating `.bin.best` checkpoints, but it is intentionally excluded from this Git repository through `.gitignore` because it contains vendored code, build artifacts, and large checkpoints.

For ordinary reproduction of the classifier experiments, retraining embeddings is not required. This repository already includes the converted WordNet data used by the final experiment:

```text
data/processed/wordnet_edges.csv
data/embeddings/wordnet_embeddings_d2.csv
data/embeddings/wordnet_embeddings_d3.csv
data/embeddings/wordnet_embeddings_d5.csv
data/embeddings/wordnet_embeddings_d10.csv
```

Use `poincare-embeddings/` only if you want to reproduce the embedding-training stage itself. After training, convert the best checkpoints with `scripts/convert_poincare_dims.sh` and commit only the resulting CSV files under `data/`, not the external training repository or `.bin` checkpoints.


## Methodology

### Task

The task is binary WordNet subtree classification. For each selected root synset, every WordNet noun synset is labeled as:

```text
positive: the node is the root or a descendant of the root subtree
negative: the node is any remaining WordNet noun synset
```

The evaluated roots are:

```text
ANIMAL.N.01, MAMMAL.N.01, GROUP.N.01, WORKER.N.01
```

The `subtree_balanced` split protocol splits positive nodes 80/20 into train/test, and independently splits negative nodes 80/20 into train/test. This preserves the rare-positive structure of the subtree task while ensuring both classes are represented in both splits.

### Embeddings

WordNet noun embeddings are trained with the Nickel and Kiela `facebookresearch/poincare-embeddings` implementation. The strict experiment trains separate Poincare embeddings for each dimension rather than truncating a higher-dimensional embedding:

```text
D = 2, 3, 5, 10
```

Each dimension is converted from the best evaluated checkpoint (`nouns_gpu_d{D}.bin.best`) into a CSV consumed by the classifier pipeline.

### Classifiers

The experiment compares three classifier families.

1. Direct Euclidean logistic regression treats Poincare coordinates as ordinary Euclidean vectors:

```text
logit = w^T x + b
```

2. Log-map Euclidean logistic regression maps points to the tangent space at the origin, then applies logistic regression:

```text
z = log_0(x)
logit = w^T z + b
```

3. Hyperbolic MLR implements a two-class Poincare hyperplane classifier. Each class learns a point `p_k` in the Poincare ball and a tangent normal vector. The binary logit is the positive-class hyperbolic score minus the negative-class score.

### Optimizers

The main experimental variable is the optimizer. Baselines use standard Euclidean optimizers; hyperbolic MLR uses both projected Euclidean and Riemannian optimizers:

```text
euclidean_lr + adam
euclidean_lr + sgd
logmap_lr + adam
logmap_lr + sgd
hyperbolic_mlr + projected_sgd
hyperbolic_mlr + projected_adam
hyperbolic_mlr + rsgd
hyperbolic_mlr + radam
```

Projected optimizers perform ordinary Euclidean updates and then project Poincare-ball class points back inside the ball. Riemannian optimizers use Geoopt's `RiemannianSGD` and `RiemannianAdam`, which update manifold parameters through geometry-aware operations.

### Metrics and Figures

Each run logs classification and optimization metrics at every epoch:

```text
train loss, test loss, accuracy, precision, recall, F1, gradient norm,
average/max parameter norm, projection count, clipping count, runtime per epoch
```

The primary classification metric is F1 because the negative class is much larger than the positive subtree. Optimization analysis uses projection/clipping counts, gradient norms, parameter norms, and runtime. Qualitative figures include loss/F1 curves, norm-F1 curves, 2D Poincare decision boundaries, and hyperbolic parameter trajectories.

## Data

The strict experiment uses real WordNet noun data with one independently trained Poincare embedding file per dimension:

```text
data/processed/wordnet_edges.csv          # parent,child; shared WordNet noun closure edges
data/embeddings/wordnet_embeddings_d2.csv # D=2 noun synset embeddings
data/embeddings/wordnet_embeddings_d3.csv # D=3 noun synset embeddings
data/embeddings/wordnet_embeddings_d5.csv # D=5 noun synset embeddings
data/embeddings/wordnet_embeddings_d10.csv # D=10 noun synset embeddings
```

The CSVs contain WordNet noun synsets such as `mammal.n.01`, not raw English word forms. In `embeddings_by_dim` mode the runner requires these files to exist and will not silently create fallback synthetic data.

The converter handles the Facebook closure format, where `id1` is the child/hyponym and `id2` is the parent/hypernym.

## Embedding Training

The embedding code lives under the vendored `facebookresearch/poincare-embeddings` repository:

```text
poincare-embeddings/runai_train_nouns.sh       # one dimension
scripts/runai_train_poincare_dims.sh           # D=2,3,5,10 wrapper
scripts/convert_poincare_dims.sh               # checkpoint-to-CSV conversion
```

The strict experiment uses one checkpoint family per dimension:

```text
poincare-embeddings/nouns_gpu_d2.bin
poincare-embeddings/nouns_gpu_d3.bin
poincare-embeddings/nouns_gpu_d5.bin
poincare-embeddings/nouns_gpu_d10.bin
```

RunAI restart behavior is controlled by:

```text
FRESH=0              # resume if checkpoint exists
CHECKPOINT_EACH=10   # update the main .bin resume checkpoint every 10 epochs
EVAL_EACH=100        # periodically evaluate reconstruction quality
EPOCHS=1500          # total target epoch, not additional epochs
```

Checkpoint semantics:

```text
nouns_gpu_d{D}.bin      # latest resume checkpoint, updated by checkpoint_each
nouns_gpu_d{D}.bin.best # best evaluated embedding, selected by reconstruction MAP rank
```

Evaluation checkpoints are temporary and do not overwrite the main `.bin` resume checkpoint. Final CSV conversion should use `.bin.best`:

```bash
DIMS="2 3 5 10" CHECKPOINT_SUFFIX=".bin.best" scripts/convert_poincare_dims.sh
```

## Experiments

The strict WordNet subtree experiment uses independently trained embeddings for each dimension:

```text
configs/full_experiment.yaml
outputs.root: results_wordnet_dims
batch_size: 256
split_protocol: subtree_balanced
subtrees: ANIMAL.N.01, MAMMAL.N.01, GROUP.N.01, WORKER.N.01
dims: 2, 3, 5, 10
seeds: 0, 1, 2
settings: 8 model/optimizer combinations
```

Dimension-specific embeddings are configured as:

```text
D=2  -> data/embeddings/wordnet_embeddings_d2.csv
D=3  -> data/embeddings/wordnet_embeddings_d3.csv
D=5  -> data/embeddings/wordnet_embeddings_d5.csv
D=10 -> data/embeddings/wordnet_embeddings_d10.csv
```

The runner refuses to start this formal experiment if any dim-specific embedding CSV is missing, so it cannot silently fall back to synthetic data.

`split_protocol: subtree_balanced` follows a subtree-aware WordNet split: positive subtree nodes are split 80/20, and the remaining WordNet noun nodes are independently split 80/20 as negatives. Published WordNet subtree tables often list only the positive train/test subtree counts, not the number of negative examples.

The current `hyperbolic_mlr` implements a two-class Poincare hyperplane MLR. Each class learns a point in the Poincare ball and a tangent normal vector; the binary logit is the positive-class hyperbolic MLR score minus the negative-class score.

The model/optimizer settings are:

```text
euclidean_lr + adam
euclidean_lr + sgd
logmap_lr + adam
logmap_lr + sgd
hyperbolic_mlr + projected_sgd
hyperbolic_mlr + projected_adam
hyperbolic_mlr + rsgd
hyperbolic_mlr + radam
```

Recommended command order for the strict WordNet subtree experiment:

```bash
# 1. Train independent WordNet noun embeddings for D=2,3,5,10.
cd /home/sjiang/hyperbolic_workspace
DIMS="2 3 5 10" FRESH=0 CONVERT_AFTER=0 scripts/runai_train_poincare_dims.sh

# 2. After all four embedding checkpoints finish, convert them to CSV.
DIMS="2 3 5 10" CHECKPOINT_SUFFIX=".bin.best" scripts/convert_poincare_dims.sh

# 3. Prepare/verify the classifier environment.
conda activate hypopt

# 4. Optional smoke test.
scripts/run_local.sh configs/small_experiment.yaml

# 5. Submit the four-subtree classifier experiment in parallel.
GPU=0 CPU=8 MEMORY=48G INSTALL_DEPS=0 scripts/run_full_split.sh

# 6. After all four classifier jobs finish, aggregate tables and figures.
CONFIG=configs/full_experiment.yaml INSTALL_DEPS=0 scripts/plot_all.sh
```

Run a local smoke test:

```bash
cd /home/sjiang/hyperbolic_workspace
conda activate hypopt
scripts/run_local.sh configs/small_experiment.yaml
```

Submit the four-subtree full experiment in parallel on RunAI:

```bash
cd /home/sjiang/hyperbolic_workspace
GPU=0 CPU=8 MEMORY=48G INSTALL_DEPS=0 scripts/run_full_split.sh
```

This submits:

```text
hypopt-full-animal
hypopt-full-mammal
hypopt-full-group
hypopt-full-worker
```

Each job runs one subtree with 96 runs:

```text
1 subtree * 4 dims * 3 seeds * 8 settings = 96 runs
```

After all four jobs finish, aggregate tables and regenerate figures:

```bash
CONFIG=configs/full_experiment.yaml INSTALL_DEPS=0 scripts/plot_all.sh
```

For local finalization instead:

```bash
conda activate hypopt
python run.py --config configs/full_experiment.yaml --finalize-only
```

## Outputs

Strict WordNet outputs are under:

```text
results_wordnet_dims/logs/
results_wordnet_dims/checkpoints/
results_wordnet_dims/tables/main_results.csv
results_wordnet_dims/tables/main_results_latex.txt
results_wordnet_dims/tables/optimizer_stability.csv
results_wordnet_dims/figures/
```

The expected complete run count is:

```text
4 subtrees * 4 dims * 3 seeds * 8 settings = 384 per-run logs
```

## Current Strict WordNet Results

The completed strict run has:

```text
main table rows: 128
per-run logs: 384 / 384
embeddings: independently trained D=2,3,5,10 Poincare noun embeddings
split protocol: subtree_balanced
classifier head: Poincare-hyperplane two-class hyperbolic MLR
output root: results_wordnet_dims
```

Best mean F1 by subtree and dimension:

| Subtree | Dim | Best setting | F1 mean ± std | Acc | Precision | Recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ANIMAL | 2 | hyperbolic_mlr + radam | 0.901 ± 0.017 | 0.990 | 0.855 | 0.955 |
| ANIMAL | 3 | hyperbolic_mlr + rsgd | 0.992 ± 0.005 | 0.999 | 0.990 | 0.994 |
| ANIMAL | 5 | hyperbolic_mlr + projected_adam | 0.998 ± 0.002 | 1.000 | 0.997 | 0.998 |
| ANIMAL | 10 | hyperbolic_mlr + projected_sgd | 0.999 ± 0.001 | 1.000 | 1.000 | 0.998 |
| GROUP | 2 | hyperbolic_mlr + projected_adam | 0.732 ± 0.013 | 0.931 | 0.604 | 0.929 |
| GROUP | 3 | hyperbolic_mlr + projected_sgd | 0.982 ± 0.002 | 0.996 | 0.997 | 0.968 |
| GROUP | 5 | euclidean_lr + adam | 0.952 ± 0.002 | 0.991 | 0.986 | 0.920 |
| GROUP | 10 | hyperbolic_mlr + projected_sgd | 0.998 ± 0.002 | 1.000 | 1.000 | 0.996 |
| MAMMAL | 2 | hyperbolic_mlr + radam | 0.893 ± 0.099 | 0.997 | 0.886 | 0.911 |
| MAMMAL | 3 | hyperbolic_mlr + radam | 0.957 ± 0.036 | 0.999 | 0.928 | 0.989 |
| MAMMAL | 5 | hyperbolic_mlr + projected_adam | 0.969 ± 0.041 | 0.999 | 0.956 | 0.983 |
| MAMMAL | 10 | hyperbolic_mlr + radam | 0.965 ± 0.014 | 0.999 | 0.947 | 0.983 |
| WORKER | 2 | hyperbolic_mlr + radam | 0.749 ± 0.074 | 0.992 | 0.678 | 0.848 |
| WORKER | 3 | hyperbolic_mlr + radam | 0.617 ± 0.534 | 0.994 | 0.634 | 0.602 |
| WORKER | 5 | hyperbolic_mlr + projected_adam | 0.968 ± 0.007 | 0.999 | 0.973 | 0.963 |
| WORKER | 10 | hyperbolic_mlr + radam | 0.958 ± 0.008 | 0.999 | 0.951 | 0.966 |

Overall mean F1 across all 16 subtree/dim tasks:

| Setting | Mean F1 | Mean Acc | Wins |
| --- | ---: | ---: | ---: |
| hyperbolic_mlr + radam | 0.895 | 0.993 | 7 |
| hyperbolic_mlr + projected_adam | 0.706 | 0.987 | 4 |
| hyperbolic_mlr + rsgd | 0.680 | 0.988 | 1 |
| hyperbolic_mlr + projected_sgd | 0.564 | 0.985 | 3 |
| logmap_lr + adam | 0.552 | 0.984 | 0 |
| euclidean_lr + adam | 0.482 | 0.983 | 1 |
| logmap_lr + sgd | 0.259 | 0.975 | 0 |
| euclidean_lr + sgd | 0.161 | 0.970 | 0 |

Mean F1 by embedding dimension, averaged over subtrees and settings:

| Dim | Mean F1 |
| ---: | ---: |
| 2 | 0.254 |
| 3 | 0.510 |
| 5 | 0.613 |
| 10 | 0.772 |

Interpretation:

- After switching to independently trained embeddings and Poincare-hyperplane hyperbolic MLR, hyperbolic classifiers dominate the best-per-task results: 15 of 16 subtree/dim cells are won by a hyperbolic setting.
- `hyperbolic_mlr + radam` is the strongest overall setting, with the highest mean F1 and 7 wins.
- Higher embedding dimension improves average F1 substantially, with D=10 strongest overall.
- `GROUP` at D=5 is the only best cell won by a Euclidean baseline.
- `WORKER` at D=3 has high variance, so conclusions for that cell should be treated cautiously.

Generated figures include F1/loss/norm curves for every subtree and dimension, a global F1 bar plot, and 2D boundary/trajectory visualizations for MAMMAL and GROUP.

## Complete Subtree Result Tables
Each table reports seed-averaged accuracy and optimization metrics over 3 seeds. `F1` is shown as mean ± std; `Loss` is final test loss; `Grad` is final gradient norm; `Param` is average parameter norm; `Proj`/`Clip` are final projection and clipping counts; `Sec/epoch` is final epoch runtime.
### ANIMAL
| Dim | Setting | F1 mean ± std | Acc | Precision | Recall | Loss | Grad | Param | Proj | Clip | Sec/epoch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | euclidean_lr + adam | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.097 | 0.013 | 11.464 | 0.0 | 0.0 | 1.15 |
| 2 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.146 | 0.022 | 1.600 | 0.0 | 0.0 | 1.15 |
| 2 | logmap_lr + adam | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.096 | 0.057 | 2.137 | 0.0 | 0.0 | 1.29 |
| 2 | logmap_lr + sgd | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.137 | 0.064 | 0.396 | 0.0 | 0.0 | 1.11 |
| 2 | hyperbolic_mlr + projected_sgd | 0.687 ± 0.382 | 0.980 | 0.906 | 0.689 | 0.043 | 0.582 | 0.777 | 112.0 | 1.0 | 2.22 |
| 2 | hyperbolic_mlr + projected_adam | 0.661 ± 0.394 | 0.965 | 0.625 | 0.703 | 0.218 | 0.676 | 0.686 | 1078.0 | 1.0 | 2.12 |
| 2 | hyperbolic_mlr + rsgd | 0.894 ± 0.008 | 0.989 | 0.858 | 0.935 | 0.029 | 0.371 | 0.834 | 0.0 | 0.0 | 2.19 |
| 2 | hyperbolic_mlr + radam | 0.901 ± 0.017 | 0.990 | 0.855 | 0.955 | 0.027 | 0.519 | 0.631 | 0.0 | 0.0 | 2.29 |
| 3 | euclidean_lr + adam | 0.956 ± 0.008 | 0.996 | 0.935 | 0.979 | 0.046 | 0.008 | 16.470 | 0.0 | 0.0 | 1.28 |
| 3 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.136 | 0.024 | 1.873 | 0.0 | 0.0 | 1.13 |
| 3 | logmap_lr + adam | 0.927 ± 0.008 | 0.993 | 0.941 | 0.913 | 0.042 | 0.035 | 3.099 | 0.0 | 0.0 | 1.23 |
| 3 | logmap_lr + sgd | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.111 | 0.062 | 0.590 | 0.0 | 0.0 | 1.17 |
| 3 | hyperbolic_mlr + projected_sgd | 0.662 ± 0.574 | 0.983 | 0.663 | 0.661 | 0.068 | 0.260 | 0.608 | 390.7 | 5.3 | 1.93 |
| 3 | hyperbolic_mlr + projected_adam | 0.988 ± 0.008 | 0.999 | 0.983 | 0.992 | 0.008 | 0.207 | 0.722 | 384.7 | 0.3 | 2.13 |
| 3 | hyperbolic_mlr + rsgd | 0.992 ± 0.005 | 0.999 | 0.990 | 0.994 | 0.008 | 0.148 | 0.744 | 0.0 | 0.0 | 2.16 |
| 3 | hyperbolic_mlr + radam | 0.991 ± 0.004 | 0.999 | 0.994 | 0.988 | 0.006 | 0.079 | 0.599 | 0.0 | 0.0 | 2.57 |
| 5 | euclidean_lr + adam | 0.944 ± 0.017 | 0.995 | 0.995 | 0.897 | 0.026 | 0.006 | 19.184 | 0.0 | 0.0 | 1.36 |
| 5 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.132 | 0.025 | 1.967 | 0.0 | 0.0 | 1.43 |
| 5 | logmap_lr + adam | 0.947 ± 0.006 | 0.995 | 0.975 | 0.919 | 0.023 | 0.022 | 3.501 | 0.0 | 0.0 | 1.22 |
| 5 | logmap_lr + sgd | 0.344 ± 0.035 | 0.960 | 0.898 | 0.213 | 0.094 | 0.054 | 0.835 | 0.0 | 0.0 | 1.28 |
| 5 | hyperbolic_mlr + projected_sgd | 0.968 ± 0.050 | 0.997 | 0.957 | 0.979 | 0.010 | 0.185 | 0.662 | 3.3 | 0.0 | 1.95 |
| 5 | hyperbolic_mlr + projected_adam | 0.998 ± 0.002 | 1.000 | 0.997 | 0.998 | 0.002 | 0.085 | 0.735 | 152.3 | 0.0 | 2.00 |
| 5 | hyperbolic_mlr + rsgd | 0.996 ± 0.002 | 1.000 | 0.995 | 0.996 | 0.005 | 0.127 | 0.716 | 0.0 | 0.0 | 2.03 |
| 5 | hyperbolic_mlr + radam | 0.995 ± 0.004 | 0.999 | 0.992 | 0.998 | 0.005 | 0.104 | 0.723 | 0.0 | 0.0 | 2.30 |
| 10 | euclidean_lr + adam | 0.996 ± 0.001 | 1.000 | 0.995 | 0.998 | 0.004 | 0.002 | 24.140 | 0.0 | 0.0 | 1.19 |
| 10 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.951 | 0.000 | 0.000 | 0.113 | 0.028 | 2.381 | 0.0 | 0.0 | 1.11 |
| 10 | logmap_lr + adam | 0.992 ± 0.001 | 0.999 | 0.997 | 0.987 | 0.005 | 0.008 | 4.570 | 0.0 | 0.0 | 1.42 |
| 10 | logmap_lr + sgd | 0.986 ± 0.001 | 0.999 | 0.998 | 0.974 | 0.047 | 0.034 | 1.315 | 0.0 | 0.0 | 1.28 |
| 10 | hyperbolic_mlr + projected_sgd | 0.999 ± 0.001 | 1.000 | 1.000 | 0.998 | 0.002 | 0.025 | 0.441 | 0.0 | 0.0 | 2.12 |
| 10 | hyperbolic_mlr + projected_adam | 0.956 ± 0.073 | 0.996 | 0.999 | 0.923 | 0.025 | 0.059 | 0.838 | 0.0 | 0.0 | 2.08 |
| 10 | hyperbolic_mlr + rsgd | 0.998 ± 0.000 | 1.000 | 0.999 | 0.998 | 0.001 | 0.044 | 0.388 | 0.0 | 0.0 | 2.04 |
| 10 | hyperbolic_mlr + radam | 0.998 ± 0.002 | 1.000 | 0.998 | 0.998 | 0.004 | 0.048 | 0.749 | 0.0 | 0.0 | 2.29 |

### GROUP
| Dim | Setting | F1 mean ± std | Acc | Precision | Recall | Loss | Grad | Param | Proj | Clip | Sec/epoch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | euclidean_lr + adam | 0.392 ± 0.040 | 0.896 | 0.488 | 0.330 | 0.163 | 0.017 | 11.106 | 0.0 | 0.0 | 0.58 |
| 2 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.898 | 0.000 | 0.000 | 0.220 | 0.027 | 2.217 | 0.0 | 0.0 | 0.54 |
| 2 | logmap_lr + adam | 0.575 ± 0.019 | 0.913 | 0.576 | 0.574 | 0.165 | 0.078 | 1.981 | 0.0 | 0.0 | 0.59 |
| 2 | logmap_lr + sgd | 0.000 ± 0.000 | 0.898 | 0.000 | 0.000 | 0.205 | 0.086 | 0.533 | 0.0 | 0.0 | 0.67 |
| 2 | hyperbolic_mlr + projected_sgd | 0.222 ± 0.385 | 0.914 | 0.293 | 0.179 | 0.210 | 1.157 | 0.925 | 659.3 | 5.0 | 0.96 |
| 2 | hyperbolic_mlr + projected_adam | 0.732 ± 0.013 | 0.931 | 0.604 | 0.929 | 0.138 | 1.670 | 0.511 | 2187.3 | 0.7 | 1.03 |
| 2 | hyperbolic_mlr + rsgd | 0.437 ± 0.389 | 0.914 | 0.390 | 0.506 | 0.175 | 0.786 | 0.826 | 0.0 | 0.0 | 1.11 |
| 2 | hyperbolic_mlr + radam | 0.708 ± 0.076 | 0.937 | 0.707 | 0.792 | 0.130 | 1.532 | 0.480 | 0.0 | 0.0 | 1.13 |
| 3 | euclidean_lr + adam | 0.979 ± 0.004 | 0.996 | 0.984 | 0.975 | 0.039 | 0.006 | 16.818 | 0.0 | 0.0 | 0.54 |
| 3 | euclidean_lr + sgd | 0.829 ± 0.003 | 0.970 | 0.997 | 0.709 | 0.137 | 0.031 | 3.355 | 0.0 | 0.0 | 0.49 |
| 3 | logmap_lr + adam | 0.944 ± 0.005 | 0.989 | 0.937 | 0.952 | 0.044 | 0.030 | 2.842 | 0.0 | 0.0 | 0.59 |
| 3 | logmap_lr + sgd | 0.913 ± 0.001 | 0.982 | 0.881 | 0.949 | 0.090 | 0.052 | 0.997 | 0.0 | 0.0 | 0.60 |
| 3 | hyperbolic_mlr + projected_sgd | 0.982 ± 0.002 | 0.996 | 0.997 | 0.968 | 0.021 | 0.220 | 0.545 | 307.7 | 2.3 | 0.95 |
| 3 | hyperbolic_mlr + projected_adam | 0.979 ± 0.006 | 0.996 | 0.987 | 0.972 | 0.021 | 0.236 | 0.585 | 0.0 | 0.0 | 1.05 |
| 3 | hyperbolic_mlr + rsgd | 0.948 ± 0.047 | 0.989 | 0.956 | 0.941 | 0.042 | 0.416 | 0.812 | 0.0 | 0.0 | 1.08 |
| 3 | hyperbolic_mlr + radam | 0.974 ± 0.011 | 0.995 | 0.984 | 0.965 | 0.021 | 0.238 | 0.448 | 0.0 | 0.0 | 1.13 |
| 5 | euclidean_lr + adam | 0.952 ± 0.002 | 0.991 | 0.986 | 0.920 | 0.052 | 0.008 | 16.189 | 0.0 | 0.0 | 0.54 |
| 5 | euclidean_lr + sgd | 0.780 ± 0.054 | 0.963 | 1.000 | 0.641 | 0.147 | 0.033 | 3.315 | 0.0 | 0.0 | 0.54 |
| 5 | logmap_lr + adam | 0.924 ± 0.009 | 0.985 | 0.953 | 0.897 | 0.062 | 0.038 | 2.763 | 0.0 | 0.0 | 0.63 |
| 5 | logmap_lr + sgd | 0.907 ± 0.003 | 0.981 | 0.929 | 0.885 | 0.091 | 0.051 | 1.231 | 0.0 | 0.0 | 0.55 |
| 5 | hyperbolic_mlr + projected_sgd | 0.645 ± 0.559 | 0.962 | 0.664 | 0.628 | 0.113 | 0.382 | 0.717 | 482.3 | 2.7 | 0.99 |
| 5 | hyperbolic_mlr + projected_adam | 0.645 ± 0.559 | 0.962 | 0.665 | 0.627 | 0.112 | 0.674 | 0.669 | 11.3 | 0.0 | 1.03 |
| 5 | hyperbolic_mlr + rsgd | 0.948 ± 0.022 | 0.990 | 0.988 | 0.911 | 0.040 | 0.467 | 0.802 | 0.0 | 0.0 | 1.11 |
| 5 | hyperbolic_mlr + radam | 0.946 ± 0.007 | 0.989 | 0.990 | 0.905 | 0.038 | 0.556 | 0.751 | 0.0 | 0.0 | 1.14 |
| 10 | euclidean_lr + adam | 0.996 ± 0.001 | 0.999 | 1.000 | 0.991 | 0.006 | 0.002 | 20.894 | 0.0 | 0.0 | 0.58 |
| 10 | euclidean_lr + sgd | 0.967 ± 0.005 | 0.993 | 1.000 | 0.935 | 0.098 | 0.032 | 3.822 | 0.0 | 0.0 | 0.50 |
| 10 | logmap_lr + adam | 0.994 ± 0.000 | 0.999 | 1.000 | 0.987 | 0.007 | 0.009 | 4.664 | 0.0 | 0.0 | 0.59 |
| 10 | logmap_lr + sgd | 0.987 ± 0.001 | 0.997 | 1.000 | 0.975 | 0.032 | 0.027 | 1.671 | 0.0 | 0.0 | 0.60 |
| 10 | hyperbolic_mlr + projected_sgd | 0.998 ± 0.002 | 1.000 | 1.000 | 0.996 | 0.003 | 0.058 | 0.282 | 0.0 | 0.0 | 1.05 |
| 10 | hyperbolic_mlr + projected_adam | 0.994 ± 0.002 | 0.999 | 0.998 | 0.989 | 0.007 | 0.076 | 0.941 | 0.0 | 0.0 | 1.05 |
| 10 | hyperbolic_mlr + rsgd | 0.997 ± 0.002 | 0.999 | 1.000 | 0.995 | 0.004 | 0.079 | 0.368 | 0.0 | 0.0 | 1.14 |
| 10 | hyperbolic_mlr + radam | 0.996 ± 0.002 | 0.999 | 1.000 | 0.992 | 0.006 | 0.087 | 0.577 | 0.0 | 0.0 | 1.18 |

### MAMMAL
| Dim | Setting | F1 mean ± std | Acc | Precision | Recall | Loss | Grad | Param | Proj | Clip | Sec/epoch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | euclidean_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.050 | 0.008 | 6.969 | 0.0 | 0.0 | 1.22 |
| 2 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.069 | 0.015 | 0.685 | 0.0 | 0.0 | 1.11 |
| 2 | logmap_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.050 | 0.036 | 1.279 | 0.0 | 0.0 | 1.27 |
| 2 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.066 | 0.040 | 0.203 | 0.0 | 0.0 | 1.11 |
| 2 | hyperbolic_mlr + projected_sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.066 | 0.397 | 1.169 | 1581.3 | 9.7 | 1.92 |
| 2 | hyperbolic_mlr + projected_adam | 0.270 ± 0.460 | 0.988 | 0.564 | 0.318 | 0.026 | 0.775 | 0.672 | 6396.0 | 13.3 | 2.00 |
| 2 | hyperbolic_mlr + rsgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.037 | 0.208 | 0.973 | 0.0 | 0.0 | 1.77 |
| 2 | hyperbolic_mlr + radam | 0.893 ± 0.099 | 0.997 | 0.886 | 0.911 | 0.009 | 0.618 | 0.659 | 0.0 | 0.0 | 2.00 |
| 3 | euclidean_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.038 | 0.008 | 9.519 | 0.0 | 0.0 | 1.36 |
| 3 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.069 | 0.015 | 0.721 | 0.0 | 0.0 | 1.11 |
| 3 | logmap_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.036 | 0.033 | 1.851 | 0.0 | 0.0 | 1.28 |
| 3 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.062 | 0.040 | 0.299 | 0.0 | 0.0 | 1.21 |
| 3 | hyperbolic_mlr + projected_sgd | 0.327 ± 0.566 | 0.990 | 0.328 | 0.326 | 0.050 | 0.296 | 0.634 | 4.7 | 0.0 | 2.09 |
| 3 | hyperbolic_mlr + projected_adam | 0.154 ± 0.267 | 0.986 | 0.181 | 0.134 | 0.025 | 0.300 | 0.719 | 2562.0 | 2.3 | 2.18 |
| 3 | hyperbolic_mlr + rsgd | 0.695 ± 0.085 | 0.991 | 0.684 | 0.712 | 0.019 | 0.213 | 0.759 | 0.0 | 0.0 | 2.15 |
| 3 | hyperbolic_mlr + radam | 0.957 ± 0.036 | 0.999 | 0.928 | 0.989 | 0.004 | 0.531 | 0.737 | 0.0 | 0.0 | 2.33 |
| 5 | euclidean_lr + adam | 0.000 ± 0.000 | 0.985 | 0.000 | 0.000 | 0.028 | 0.007 | 12.649 | 0.0 | 0.0 | 1.20 |
| 5 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.068 | 0.016 | 0.819 | 0.0 | 0.0 | 1.18 |
| 5 | logmap_lr + adam | 0.380 ± 0.219 | 0.988 | 0.688 | 0.274 | 0.025 | 0.027 | 2.475 | 0.0 | 0.0 | 1.22 |
| 5 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.056 | 0.038 | 0.470 | 0.0 | 0.0 | 1.21 |
| 5 | hyperbolic_mlr + projected_sgd | 0.664 ± 0.575 | 0.995 | 0.662 | 0.665 | 0.026 | 0.303 | 0.640 | 36.7 | 0.7 | 2.00 |
| 5 | hyperbolic_mlr + projected_adam | 0.969 ± 0.041 | 0.999 | 0.956 | 0.983 | 0.005 | 0.248 | 0.842 | 1482.7 | 2.3 | 2.01 |
| 5 | hyperbolic_mlr + rsgd | 0.750 ± 0.298 | 0.995 | 0.844 | 0.747 | 0.013 | 0.162 | 0.673 | 0.0 | 0.0 | 2.03 |
| 5 | hyperbolic_mlr + radam | 0.899 ± 0.159 | 0.997 | 0.892 | 0.905 | 0.008 | 0.133 | 0.821 | 0.0 | 0.0 | 2.31 |
| 10 | euclidean_lr + adam | 0.640 ± 0.064 | 0.992 | 0.887 | 0.503 | 0.018 | 0.005 | 16.792 | 0.0 | 0.0 | 1.15 |
| 10 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.066 | 0.016 | 0.955 | 0.0 | 0.0 | 1.07 |
| 10 | logmap_lr + adam | 0.902 ± 0.014 | 0.997 | 0.946 | 0.863 | 0.014 | 0.018 | 4.041 | 0.0 | 0.0 | 1.23 |
| 10 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.047 | 0.036 | 0.637 | 0.0 | 0.0 | 1.11 |
| 10 | hyperbolic_mlr + projected_sgd | 0.588 ± 0.522 | 0.993 | 0.607 | 0.572 | 0.014 | 0.172 | 0.802 | 817.0 | 2.3 | 2.09 |
| 10 | hyperbolic_mlr + projected_adam | 0.518 ± 0.500 | 0.992 | 0.655 | 0.463 | 0.017 | 0.220 | 0.929 | 484.3 | 1.3 | 2.11 |
| 10 | hyperbolic_mlr + rsgd | 0.642 ± 0.556 | 0.994 | 0.625 | 0.660 | 0.013 | 0.197 | 0.621 | 0.0 | 0.0 | 2.19 |
| 10 | hyperbolic_mlr + radam | 0.965 ± 0.014 | 0.999 | 0.947 | 0.983 | 0.003 | 0.157 | 1.022 | 0.0 | 0.0 | 2.29 |

### WORKER
| Dim | Setting | F1 mean ± std | Acc | Precision | Recall | Loss | Grad | Param | Proj | Clip | Sec/epoch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | euclidean_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.046 | 0.008 | 6.702 | 0.0 | 0.0 | 0.64 |
| 2 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.066 | 0.015 | 0.665 | 0.0 | 0.0 | 0.55 |
| 2 | logmap_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.046 | 0.036 | 1.230 | 0.0 | 0.0 | 0.62 |
| 2 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.062 | 0.039 | 0.220 | 0.0 | 0.0 | 0.53 |
| 2 | hyperbolic_mlr + projected_sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.060 | 0.233 | 0.764 | 790.7 | 7.0 | 0.94 |
| 2 | hyperbolic_mlr + projected_adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.049 | 0.907 | 0.619 | 3755.3 | 8.3 | 0.97 |
| 2 | hyperbolic_mlr + rsgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.040 | 0.196 | 1.064 | 0.0 | 0.0 | 1.00 |
| 2 | hyperbolic_mlr + radam | 0.749 ± 0.074 | 0.992 | 0.678 | 0.848 | 0.019 | 0.596 | 0.566 | 0.0 | 0.0 | 1.13 |
| 3 | euclidean_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.046 | 0.008 | 7.330 | 0.0 | 0.0 | 0.53 |
| 3 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.067 | 0.015 | 0.647 | 0.0 | 0.0 | 0.49 |
| 3 | logmap_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.045 | 0.034 | 1.345 | 0.0 | 0.0 | 0.58 |
| 3 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.062 | 0.037 | 0.234 | 0.0 | 0.0 | 0.59 |
| 3 | hyperbolic_mlr + projected_sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.067 | 0.220 | 0.520 | 224.0 | 2.0 | 0.95 |
| 3 | hyperbolic_mlr + projected_adam | 0.506 ± 0.440 | 0.990 | 0.426 | 0.626 | 0.026 | 0.346 | 0.598 | 2551.7 | 4.0 | 0.99 |
| 3 | hyperbolic_mlr + rsgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.037 | 0.218 | 0.814 | 0.0 | 0.0 | 1.06 |
| 3 | hyperbolic_mlr + radam | 0.617 ± 0.534 | 0.994 | 0.634 | 0.602 | 0.018 | 0.448 | 0.633 | 0.0 | 0.0 | 1.12 |
| 5 | euclidean_lr + adam | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.026 | 0.007 | 12.172 | 0.0 | 0.0 | 0.59 |
| 5 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.065 | 0.015 | 0.791 | 0.0 | 0.0 | 0.54 |
| 5 | logmap_lr + adam | 0.328 ± 0.067 | 0.989 | 1.000 | 0.197 | 0.024 | 0.026 | 2.388 | 0.0 | 0.0 | 0.58 |
| 5 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.057 | 0.038 | 0.389 | 0.0 | 0.0 | 0.65 |
| 5 | hyperbolic_mlr + projected_sgd | 0.321 ± 0.557 | 0.990 | 0.320 | 0.323 | 0.032 | 0.225 | 0.706 | 98.7 | 2.0 | 1.08 |
| 5 | hyperbolic_mlr + projected_adam | 0.968 ± 0.007 | 0.999 | 0.973 | 0.963 | 0.007 | 0.193 | 0.659 | 1748.7 | 2.3 | 1.13 |
| 5 | hyperbolic_mlr + rsgd | 0.627 ± 0.543 | 0.994 | 0.616 | 0.638 | 0.030 | 1.039 | 0.859 | 0.0 | 0.0 | 1.04 |
| 5 | hyperbolic_mlr + radam | 0.777 ± 0.306 | 0.996 | 0.965 | 0.729 | 0.014 | 0.212 | 0.615 | 0.0 | 0.0 | 1.18 |
| 10 | euclidean_lr + adam | 0.849 ± 0.007 | 0.996 | 0.947 | 0.770 | 0.013 | 0.004 | 19.836 | 0.0 | 0.0 | 0.55 |
| 10 | euclidean_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.065 | 0.015 | 0.799 | 0.0 | 0.0 | 0.55 |
| 10 | logmap_lr + adam | 0.927 ± 0.012 | 0.998 | 0.971 | 0.888 | 0.012 | 0.015 | 3.876 | 0.0 | 0.0 | 0.60 |
| 10 | logmap_lr + sgd | 0.000 ± 0.000 | 0.986 | 0.000 | 0.000 | 0.049 | 0.036 | 0.708 | 0.0 | 0.0 | 0.56 |
| 10 | hyperbolic_mlr + projected_sgd | 0.956 ± 0.006 | 0.999 | 0.947 | 0.966 | 0.005 | 0.099 | 0.592 | 3.3 | 0.0 | 1.06 |
| 10 | hyperbolic_mlr + projected_adam | 0.953 ± 0.007 | 0.999 | 0.946 | 0.961 | 0.009 | 0.245 | 0.904 | 175.7 | 0.3 | 1.04 |
| 10 | hyperbolic_mlr + rsgd | 0.953 ± 0.006 | 0.999 | 0.930 | 0.978 | 0.008 | 0.153 | 0.609 | 0.0 | 0.0 | 1.11 |
| 10 | hyperbolic_mlr + radam | 0.958 ± 0.008 | 0.999 | 0.951 | 0.966 | 0.008 | 0.188 | 0.924 | 0.0 | 0.0 | 1.20 |

## Conclusions and Report Guidance

This section is intended as a compact guide for writing the final report. The key claim supported by the strict experiment is:

```text
When the WordNet embeddings are trained independently for each dimension and the classifier is the Poincare hyperplane MLR, hyperbolic optimization recovers the expected advantage on WordNet subtree classification.
```

Main conclusions:

- Hyperbolic classifiers win 15 of 16 subtree/dimension tasks. This is the central result to emphasize.
- `hyperbolic_mlr + radam` is the strongest overall setting, with mean F1 = 0.895 across all 16 subtree/dimension tasks.
- The strict setup resolves the earlier misleading result from the prototype classifier: the old two-prototype head underperformed, while the Poincare-hyperplane hyperbolic MLR strongly outperforms Euclidean/log-map baselines in most cells.
- Dimensionality matters. Average F1 rises from 0.254 at D=2 to 0.772 at D=10 when averaged over all models/settings, showing that the embedding dimension has a large effect on separability.
- `GROUP` at D=5 is the only cell where a Euclidean baseline is best. This should be described as an exception rather than the main pattern.
- Accuracy alone is not a reliable metric because the negative class is very large. F1, precision, and recall should be the primary metrics in the report.

Optimization behavior to discuss:

- Riemannian optimizers (`rsgd`, `radam`) have zero projection counts because they operate through Geoopt manifold-aware updates.
- Projected optimizers can achieve excellent F1, but their projection counts can be large, especially in low-dimensional or rare-positive tasks. This indicates that unconstrained Euclidean steps often leave the Poincare ball and require correction.
- Runtime per epoch is higher for hyperbolic MLR than for Euclidean/log-map baselines because it computes Mobius operations and manifold geometry. In these runs, average final-epoch runtime is approximately 0.87s for Euclidean LR, 0.91s for log-map LR, and 1.59s for hyperbolic MLR.
- `radam` provides the best accuracy/optimization tradeoff in this experiment: it is the overall F1 leader and avoids projection events.

Recommended report structure:

1. Motivation: WordNet is hierarchical, so hyperbolic geometry should represent and classify subtree structure efficiently.
2. Data: use WordNet noun closure with positives as subtree descendants and negatives as all remaining noun synsets.
3. Embeddings: train separate Poincare noun embeddings for D=2,3,5,10 using the Nickel and Kiela implementation; convert `.bin.best` checkpoints to CSV.
4. Models: compare Euclidean LR, log-map LR, and Poincare-hyperplane hyperbolic MLR.
5. Optimizers: compare SGD/Adam baselines, projected SGD/Adam, and Riemannian SGD/Adam.
6. Results: lead with the best-by-subtree/dim table, then the overall setting table, then optimizer stability.
7. Discussion: emphasize hyperbolic MLR's advantage, the cost of manifold computation, and the instability signaled by projection counts.
8. Limitations: single train/test protocol, three seeds, CPU classifier training, and dependence on the quality of pretrained embeddings.

Most useful artifacts for the report:

```text
results_wordnet_dims/tables/main_results.csv          # primary accuracy + optimization metrics
results_wordnet_dims/tables/optimizer_stability.csv   # optimizer stability and runtime summary
results_wordnet_dims/figures/main_f1_barplot.png      # global F1 comparison
results_wordnet_dims/figures/f1_curve_*               # convergence curves
results_wordnet_dims/figures/boundary_*               # 2D decision boundaries
results_wordnet_dims/figures/trajectory_*             # 2D hyperbolic parameter trajectories
```

Limitations and cautions:

- The old single-checkpoint prototype results preserved below should only be used as an ablation/history point, not as the final conclusion.
- The strict results depend on `.bin.best` embedding checkpoints selected by reconstruction MAP rank; using latest `.bin` checkpoints may produce slightly different classification results.
- The `WORKER, D=3` result has high seed variance, so avoid overinterpreting that one cell.
- Projection counts are useful diagnostics, but they are not directly comparable to Riemannian optimizers where projection is not part of the update rule.
- The full subtree tables are intentionally exhaustive; for a report, use summarized tables in the main text and move the complete tables to an appendix.

## Useful Checks

Check that real WordNet data is present:

```bash
wc -l data/processed/wordnet_edges.csv data/embeddings/wordnet_embeddings.csv
```

Check Poincare-hyperplane full result completeness:

```bash
find results_wordnet_dims/logs -maxdepth 1 -type f -name '*.csv' ! -name '*trajectory*' | wc -l
```

Check RunAI jobs:

```bash
runai list --project vita-sjiang
```

Follow split job logs:

```bash
tail -f logs/0604_hypopt-full-animal.txt
tail -f logs/0604_hypopt-full-mammal.txt
tail -f logs/0604_hypopt-full-group.txt
tail -f logs/0604_hypopt-full-worker.txt
```

## Implementation Notes

Subtree extraction treats all descendants of a root synset as positives and all other noun synsets as negatives. Roots in configs can be uppercase (`MAMMAL.N.01`) because the loader normalizes to the lowercase WordNet nodes (`mammal.n.01`).

The hyperbolic classifier is a two-class Poincare hyperplane MLR. Projected optimizers use ordinary PyTorch updates followed by projection of Poincare-ball class points. Riemannian optimizers use Geoopt.

## References

1. Ganea, Becigneul, and Hofmann, "Hyperbolic Neural Networks," NeurIPS 2018.
2. Nickel and Kiela, "Poincare Embeddings for Learning Hierarchical Representations," NeurIPS 2017.
3. Bonnabel, "Stochastic Gradient Descent on Riemannian Manifolds," IEEE TAC 2013.
4. Kingma and Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
5. Becigneul and Ganea, "Riemannian Adaptive Optimization Methods," ICLR 2019.
6. Kochurov, Karimov, and Kozlukov, "Geoopt: Riemannian Optimization in PyTorch," 2020.
