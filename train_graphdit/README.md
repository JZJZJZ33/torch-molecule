# Graph-DiT polymer generation

This workflow has two stages:

1. pretrain an unconditional polymer generator on RPPD for a local baseline or
   on PI1M for the server model;
2. initialize a conditional model from that checkpoint and fine-tune it on RPPD
   property labels.

Run commands from the repository root in the `torch-molecule` environment.

## Fine-tune the downloaded Llamole Graph-DiT checkpoint

The Hugging Face checkpoint bundle is stored at:

```text
train_graphdit/pretrained/llamole_pretrained_graphdit/
├── model.pt
├── config.yaml
└── data.meta.json
```

The checkpoint bundle is not a standalone Python model. `model.pt` stores the
learned tensors, `config.yaml` stores architecture and diffusion settings, and
`data.meta.json` stores the atom vocabulary and graph distributions. The Python
classes that connect those tensors are supplied by:

```text
train_graphdit/llamole_graphdit/
```

These files must also be present on the H20 server. The original project assumes
its complete repository is installed and therefore already has these classes;
this repository keeps only the graph-decoder portion needed by the RPPD trainer.
The downloaded weights cannot be loaded by this repository's smaller Graph-DiT
implementation because its layer names and tensor shapes differ.

The minimum server layout for this fine-tuning workflow is:

```text
data/
└── 20260920_rppd.csv
data_process/
└── common.py
train_grin/
└── properties.py
train_graphdit/
├── common.py
├── finetune_llamole_graphdit.py
├── llamole_graphdit/
│   ├── __init__.py
│   ├── conditions.py
│   ├── diffusion_model.py
│   ├── diffusion_utils.py
│   ├── layers.py
│   ├── molecule_utils.py
│   └── transformer.py
└── pretrained/
    └── llamole_pretrained_graphdit/
        ├── model.pt
        ├── config.yaml
        └── data.meta.json
```

Copying the complete repository plus the ignored pretrained checkpoint directory
is the simplest way to reproduce this layout. Git does not include `model.pt`.

Validate the bundle and the requested RPPD columns without starting training:

```bash
python train_graphdit/finetune_llamole_graphdit.py \
  --property density,tg \
  --output-dir train_graphdit/output/rppd_density_tg_llamole \
  --check-only
```

Fine-tune one joint density and Tg model on a CUDA server:

```bash
python train_graphdit/finetune_llamole_graphdit.py \
  --pretrained-dir train_graphdit/pretrained/llamole_pretrained_graphdit \
  --input data/20260920_rppd.csv \
  --property density,tg \
  --output-dir train_graphdit/output/rppd_density_tg_llamole \
  --device cuda \
  --precision bf16 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 5e-5 \
  --epochs 200 \
  --patience 30
```

The 16 GB starting configuration trains the two property encoders, the final two
transformer blocks, and the output layer while keeping the rest of the downloaded
574M-parameter model frozen. Gradient checkpointing is enabled. If memory is still
insufficient, add `--last-layers 0` to train only the property encoders.

This is a separate workflow from `finetune.py`, which only accepts checkpoints
created by this repository's smaller Graph-DiT implementation.

The PI1M source file in this repository is:

```text
data/PI1M_v2.csv
```

The existing `train_graphdit/output/local_pretrain/unconditional` checkpoint is
an RPPD smoke-test checkpoint. It was not trained on PI1M. Use the commands below
to create a new PI1M checkpoint.

A simulated server output is available under the production-style name
`train_graphdit/output/graphdit_pi1m_full/`. It has the complete expected output
layout and a real, loadable checkpoint copied from the two-epoch, 1,000-structure
PI1M Mac smoke run. It is only for downstream pipeline and fine-tuning tests. Read
`DUMMY_OUTPUT.json` before use; this checkpoint is not deployment-ready and must
not be presented as a full PI1M-trained model.

## Task 1: Test the workflow locally on an M2 Mac

The `mac` preset uses CPU, FP32, 3 layers, hidden size 128, 8 heads, batch size 8,
and no data-loader subprocesses. A short end-to-end check is:

```bash
python train_graphdit/pretrain.py \
  --preset mac --dataset rppd --epochs 2 --timesteps 50 \
  --output-root train_graphdit/output/local_pretrain

python train_graphdit/generate.py \
  --model-dir train_graphdit/output/local_pretrain/unconditional \
  --number 8 --batch-size 4
```

This produces an unconditional checkpoint and generated p-SMILES. Two epochs are
only a software check and will not produce a useful generator.

To exercise the PI1M path locally without attempting full pretraining:

```bash
python train_graphdit/pretrain.py \
  --preset mac --dataset pi1m --input data/PI1M_v2.csv \
  --max-structures 1000 --epochs 2 --timesteps 50 \
  --output-root train_graphdit/output/pi1m_local_test

python train_graphdit/generate.py \
  --model-dir train_graphdit/output/pi1m_local_test/unconditional \
  --number 8 --batch-size 4
```

This subset run verifies preprocessing, training, checkpoint loading, and
unconditional generation. It is not the checkpoint intended for deployment.

Fine-tune a property-conditioned model from that checkpoint:

```bash
python train_graphdit/finetune.py \
  --preset mac --dataset rppd --property density \
  --pretrained-model train_graphdit/output/local_pretrain/unconditional/best_model.pt \
  --epochs 2 --output-root train_graphdit/output/local_density

python train_graphdit/generate.py \
  --model-dir train_graphdit/output/local_density/density \
  --condition density=1.2 --number 8 --batch-size 4
```

The fine-tuning model inherits the unconditional graph vocabulary, size
distribution, diffusion schedule, transformer architecture, and every compatible
denoiser tensor. New property-embedding tensors are initialized separately. The
exact transfer is recorded in `transfer_report.json`.

## Task 2: Pretrain an unconditional PI1M model on an H20

Copy `data/PI1M_v2.csv` to the server with the repository, or provide another
absolute path. The loader recognizes common column names (`SMILES`, `smiles`,
`smiles_list`, `p_smiles`, and `p-SMILES`), or use `--smiles-column NAME`.

First benchmark a subset:

```bash
python train_graphdit/pretrain.py \
  --preset h20 --dataset pi1m --input data/PI1M_v2.csv \
  --max-structures 100000 --output-root /checkpoints/graphdit_pi1m_100k
```

Then remove `--max-structures` for the cleaned full dataset:

```bash
python train_graphdit/pretrain.py \
  --preset h20 --dataset pi1m --input data/PI1M_v2.csv \
  --output-root /checkpoints/graphdit_pi1m_full
```

The `h20` preset selects CUDA BF16, 6 layers, hidden size 512, 8 heads, batch size
128, and 8 data-loader workers. These are starting values. Reduce `--batch-size`
if CUDA runs out of memory; increase it after observing headroom. The process must
have the full GPU rather than a small MIG slice.

## Task 3: Fine-tune one property on the H20

Use a single-property model when the user will request one property at a time.
The recommended initial RPPD fine-tuning settings are:

| Setting | Recommended value | Reason |
|---|---:|---|
| Batch size | 16 | Gives more optimizer updates from the small RPPD dataset |
| Learning rate | `5e-5` | Reduces damage to the PI1M-pretrained representation |
| Maximum epochs | 200 | Allows the new condition embedding to learn |
| Early-stopping patience | 30 | Stops when validation no longer improves |

Before the real H20 checkpoint exists, test this step locally against the dummy
server layout:

```bash
python train_graphdit/finetune.py \
  --preset mac \
  --dataset rppd \
  --property density \
  --pretrained-model train_graphdit/output/graphdit_pi1m_full/unconditional/best_model.pt \
  --epochs 2 \
  --output-root train_graphdit/output/dummy_server_density_test
```

Replace the dummy checkpoint with the full H20 checkpoint for meaningful training.

Train a density-conditioned model:

```bash
python train_graphdit/finetune.py \
  --preset h20 \
  --dataset rppd \
  --input /data/rppd_clean.csv \
  --property density \
  --pretrained-model /checkpoints/graphdit_pi1m_full/unconditional/best_model.pt \
  --batch-size 16 \
  --learning-rate 0.00005 \
  --epochs 200 \
  --patience 30 \
  --output-root /checkpoints/graphdit_rppd_density \
  --fail-fast
```

Generate polymers for one density target, expressed in the original RPPD units:

```bash
python train_graphdit/generate.py \
  --model-dir /checkpoints/graphdit_rppd_density/density \
  --device cuda \
  --condition density=1.2 \
  --number 1000 \
  --batch-size 64
```

The same pattern works for `tg`, `thermal_conductivity`, and the other curated
RPPD properties by changing `--property`, the output directory, and the condition.

## Task 4: Fine-tune one model for several simultaneous properties

Use a joint model when one generated polymer must satisfy every selected property
at the same time. Joint training keeps only RPPD structures with all requested
labels, so adding properties reduces the available training set.

Train one model jointly conditioned on density and Tg:

```bash
python train_graphdit/finetune.py \
  --preset h20 \
  --dataset rppd \
  --input /data/rppd_clean.csv \
  --property density,tg \
  --joint \
  --pretrained-model /checkpoints/graphdit_pi1m_full/unconditional/best_model.pt \
  --batch-size 16 \
  --learning-rate 0.00005 \
  --epochs 200 \
  --patience 30 \
  --output-root /checkpoints/graphdit_rppd_density_tg \
  --fail-fast
```

Generate with both targets:

```bash
python train_graphdit/generate.py \
  --model-dir /checkpoints/graphdit_rppd_density_tg/joint_density__tg \
  --device cuda \
  --condition density=1.2 \
  --condition tg=420 \
  --number 1000 \
  --batch-size 64
```

## Task 5: Train separate models for several properties

Omit `--joint` when each property should have its own checkpoint:

```bash
python train_graphdit/finetune.py \
  --preset h20 \
  --dataset rppd \
  --input /data/rppd_clean.csv \
  --property density,tg,thermal_conductivity \
  --pretrained-model /checkpoints/graphdit_pi1m_full/unconditional/best_model.pt \
  --batch-size 16 \
  --learning-rate 0.00005 \
  --epochs 200 \
  --patience 30 \
  --output-root /checkpoints/graphdit_rppd_single_properties
```

This produces independent `density/`, `tg/`, and `thermal_conductivity/` model
directories. `--all` trains every curated property as an independent model.

### Choosing separate or joint models

| User request | Model strategy |
|---|---|
| Generate for density only | Single density model |
| Generate for Tg only | Single Tg model |
| Let the user choose exactly one property | Separate model per property |
| One polymer must meet density and Tg together | Joint density + Tg model |
| User can select arbitrary property combinations | Train supported joint combinations, or add masked conditioning later |

Separate models cannot be combined at inference to guarantee that one polymer
satisfies several targets. A joint model learns those targets together, but can
only support the property set on which it was trained. The current implementation
does not yet provide arbitrary masked subsets within one checkpoint.

## Reference: Data and training behavior

Both sources are canonicalized and deduplicated. A repeat unit must be connected
and contain exactly two degree-one `*` attachment atoms. The default limit is 50
heavy atoms; attachment atoms are also included in graph padding.

PI1M is treated as an unlabeled structural dataset. RPPD replicate measurements
are aggregated by median. Unique structures are split 70%/15%/15%. Conditional
target scaling is fitted on training only, while original values remain in `_raw`
columns. Optional IQR filtering is applied only to training.

The PI1M loader canonicalizes the source CSV at the beginning of a fresh run. For
a million-row run, allow time and system memory for this CPU stage. Prepared splits
remain in the run directory, so resume does not repeat it.

Models use validation denoising loss for checkpoint selection and test loss once
after training. `best_model.pt` is used for generation. `last_model.pt` and
`last_checkpoint.pt` support recovery with optimizer, scheduler, RNG, and
early-stopping state. Resume by supplying the same output root, `--resume`, and a
larger total `--epochs` value. Only load checkpoints you trust.

## Reference: Generation and evaluation

Unconditional generation needs no property values:

```bash
python train_graphdit/generate.py \
  --model-dir /checkpoints/graphdit_pi1m_full/unconditional \
  --device cuda --number 100 --batch-size 32
```

Conditional inputs use original RPPD units:

```bash
python train_graphdit/generate.py \
  --model-dir /checkpoints/graphdit_rppd_density_tg/joint_density__tg \
  --device cuda --condition density=1.2 --condition tg=420 \
  --number 100 --batch-size 32
```

You may supply `--conditions-csv` with one target vector per row or use
`--held-out-conditions` for evaluation. Outputs report RDKit validity,
two-attachment polymer validity, uniqueness, novelty, graph size, and nearest
training fingerprint similarity.

`evaluate.py` can optionally apply existing GRIN predictors. GRIN results are
surrogate predictions rather than experimental or simulated property measurements.
