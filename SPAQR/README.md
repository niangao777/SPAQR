# SPAQR

Official main-experiment implementation of **SPAQR: Sampling-Aware Patch
Allocation and Query-Conditioned Retrieval for Irregular Multivariate Time
Series Forecasting**.

This release contains only the full SPAQR model and the four main benchmarks:
HumanActivity, USHCN, PhysioNet'12 (P12), and MIMIC-III. Ablation variants and
unrelated PyOmniTS baselines are intentionally excluded.

## Method overview

```text
irregular multivariate observations
    -> sampling-aware support allocation under a sample-level budget cap
    -> four support states: coarse, left, right, transition
    -> coarse global query retrieval
    -> query-conditioned within-support state selection
    -> forecasting head
```

Every valid historical observation is assigned to exactly one contiguous,
non-overlapping support. The configured budget is an upper bound: a sample can
use fewer supports when its observations cannot be split further.

## Environment

The experiments were developed in the PyOmniTS environment with PyTorch and
CUDA. Install the direct dependencies with:

```bash
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build matching the local CUDA version before
installing the remaining requirements.

## Data

The run scripts use the following logical roots:

```text
storage/datasets/HumanActivity
storage/datasets/USHCN
storage/datasets/P12
storage/datasets/MIMIC_III
```

HumanActivity is loaded directly from its configured root. The three embedded
tsdm tasks preserve the PyOmniTS protocol and cache their raw/processed files
under `~/.tsdm/rawdata/` and `~/.tsdm/datasets/`. In particular, a locally
prepared MIMIC-III `complete_tensor.csv` is expected by the bundled tsdm task;
the preprocessing scripts are under
`data/dependencies/MIMIC_III/preprocess/`.

## Main experiments

Run commands from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/SPAQR/HumanActivity.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/SPAQR/USHCN.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/SPAQR/P12.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/SPAQR/MIMIC_III.sh
```

Each script runs five seeds (`2024`--`2028`) by default. A short smoke run can
be launched without editing the script:

```bash
CUDA_VISIBLE_DEVICES=0 ITR=1 TRAIN_EPOCHS=1 \
  bash scripts/SPAQR/P12.sh
```

The principal environment overrides are:

```text
ITR                         number of seeds/runs
TRAIN_EPOCHS                maximum training epochs
PRED_LENS                   prediction horizon(s)
MISSING_RATE                optional observation masking rate
SPAQR_P_TARGET              Pmax per variable
SPAQR_MIN_PATCH_MASS        target mass in the allocation score
SPAQR_TOKEN_BUDGET          sample-level support budget cap
SPAQR_COUNT_PENALTY         allocation count term weight
SPAQR_SPAN_PENALTY          allocation span term weight
SPAQR_GAP_PENALTY           allocation gap term weight
SPAQR_DECODER_LAYERS        query-retrieval depth
SPAQR_DEBUG                 print tokenizer diagnostics when set to 1
```

The default main settings are:

| Dataset | Observation horizon | Prediction horizon | Variables | Batch | `d_model` | Heads | Budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| HumanActivity | 3000 ms | 300 ms | 12 | 32 | 128 | 4 | 48 |
| USHCN | 150 benchmark units | 3 | 5 | 16 | 32 | 1 | 20 |
| P12 | 36 h | 3 h | 36 | 32 | 64 | 1 | 144 |
| MIMIC-III | 72 half-hour bins (36 h) | 3 bins | 96 | 32 | 64 | 1 | 384 |

All four scripts use Adam, initial learning rate `1e-3`, MSE training loss,
early-stopping patience 10, and the same PyOmniTS split/evaluation protocol.
Reported MSE and MAE are computed on valid target observations in normalized
model space.

## Outputs and checkpoint evaluation

Training artifacts are saved under:

```text
storage/results/<dataset>/<dataset_id>/SPAQR/<model_id>/
  <seq_len>_<pred_len>/<timestamp>/iter<i>/
```

Each evaluation writes `metric.json` into an `eval_<timestamp>` directory.
To test a particular checkpoint directory without training, invoke `main.py`
with the same dataset/model arguments as its run script and add:

```text
--is_training 0
--checkpoints_test /path/to/iter0
```

The tensor names in existing checkpoints are unchanged by the public model
rename, so checkpoints produced by the same full architecture remain loadable.
Their old result-directory names do not need to be renamed when supplied
explicitly through `--checkpoints_test`.

## Configuration notes

The shell scripts are the executable source of the main-experiment settings.
YAML files under `configs/SPAQR/` are synchronized reference snapshots; the
configuration loader does not use YAML values to override command-line
arguments.

The main implementation is located in:

```text
models/SPAQR.py
layers/SPAQR/components.py
layers/SPAQR/support_unfolding.py
```

Third-party source information is recorded in `THIRD_PARTY_NOTICE.md`.
