<p align="center">
  <img src="assets/logo.png" alt="torch-molecule logo" width="600"/>
</p>

<p align="center">
  <a href="https://github.com/liugangcode/torch-molecule">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub Repository">
  </a>
  <a href="https://liugangcode.github.io/torch-molecule/">
    <img src="https://img.shields.io/badge/Documentation-Online-brightgreen?logo=readthedocs" alt="Documentation">
  </a>
</p>

<p align="center">
  <b>Deep learning for molecular discovery with a simple sklearn-style interface</b>
</p>

---

`torch-molecule` is a package that facilitates molecular discovery through deep learning, featuring a user-friendly, `sklearn`-style interface. It includes model checkpoints for efficient deployment and benchmarking across a range of molecular tasks. The package focuses on three main components: **Predictive Models**, **Generative Models**, and **Representation Models**, which make molecular AI models easy to implement and deploy. 

<p align="center">
  <img src="assets/cover.png" alt="scikit-learn vs torch-molecule comparison" width="800"/>
</p>

See the [List of Supported Models](#list-of-supported-models) section for all available models.

## Installation

1. **Create a Conda environment**:
   ```bash
   conda create --name torch_molecule python=3.11.7
   conda activate torch_molecule
   ```

2. **Install using pip**:
   ```bash
   pip install torch-molecule
   ```

3. **Install from source for the latest version**:

   Clone the repository:
   ```bash
   git clone https://github.com/liugangcode/torch-molecule
   cd torch-molecule
   ```

   Install:
   ```bash
   pip install .
   ```

### Additional Packages

| Model | Required Packages |
|-------|-------------------|
| HFPretrainedMolecularEncoder | transformers |
| BFGNNMolecularPredictor | torch-scatter |
| GRINMolecularPredictor | torch-scatter |
| GRINMolecularPredictor (if enable `repetition_augmentation=True`) | CombineMols |

**For models that require `torch-scatter`**: Install using the following command: `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`, e.g.,

> `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html`

**For models that require `transformers`:** `pip install transformers`

## HKRI Polymer Studio workflow

This repository also contains a local RPPD-to-GRIN workflow and a single web
application for serving all trained polymer-property models. The source code is
organized as follows:

```text
data_process/   RPPD inspection, cleaning, and leakage-safe data preparation
train_grin/     standardized multi-property GRIN training and evaluation
grin_api/       FastAPI model service and RDKit structure rendering
grin_frontend/  Predictor interface served by the API
```

The repository includes the source RPPD export, cleaned dataset, and first
standardized model run required for server deployment. Temporary prepared splits
and future timestamped training runs are excluded from Git until deliberately
selected for a release.

### Local setup (Apple Silicon or CPU)

```bash
conda create --name torch-molecule python=3.11.7 -y
conda activate torch-molecule
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation torch-scatter
```

`torch-scatter` must be installed after PyTorch. On CUDA servers, use the wheel
matching the installed PyTorch/CUDA versions from the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### NVIDIA H20 server setup

The H20 should use a CUDA-enabled PyTorch build. Check the installed NVIDIA driver
first with `nvidia-smi`, then select a supported CUDA wheel from the
[PyTorch installation page](https://pytorch.org/get-started/locally/). For a
CUDA 12.8 deployment:

```bash
conda create --name torch-molecule python=3.11.7 -y
conda activate torch-molecule
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

TORCH_VERSION=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
python -m pip install torch-scatter \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html"
```

Confirm that PyTorch can see the H20:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Prepare data and train models

Place the RPPD CSV at the repository root, then run:

```bash
python data_process/clean_rppd.py 20260920_rppd.csv
python train_grin/train.py --list-properties
python train_grin/train.py --all --device cpu  # use --device cuda on the H20
```

Training standardizes each target using training-split statistics and saves the
inverse-transform parameters with every checkpoint. Each timestamped property
directory also contains original-unit MAE, MSE, RMSE and R² metrics, prediction
CSVs, a scatter plot, and a training-loss plot. Train selected targets with one
or more `--property` arguments instead of `--all`.

### Run the predictor application

```bash
GRIN_MODEL_RUN_DIR="$PWD/train_grin/output_standardized/run_20260921_003803" \
GRIN_API_DEVICE=cpu \
python -m uvicorn grin_api.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir grin_api --reload-dir grin_frontend
```

Set `GRIN_API_DEVICE=cuda` when serving on the H20.

Open `http://127.0.0.1:8000`. The same process hosts the API, frontend, all
available property models, and RDKit structure drawing. The page is delivered as
one self-contained HTML response, so a separate static-file server is unnecessary.
Model inference is local and does not require Hugging Face access.

See [`data_process/README.md`](data_process/README.md),
[`train_grin/README.md`](train_grin/README.md), and
[`grin_api/README.md`](grin_api/README.md) for detailed options and API examples.

## Usage

> More examples can be found in the `examples` and `tests` folders.

`torch-molecule` supports applications in broad domains from chemistry, biology, to materials science. To get started, you can load prepared datasets from `torch_molecule.datasets` (updated after v0.1.3):
| Dataset | Description | Function |
|---------|-------------|----------|
| qm9 | Quantum chemical properties (DFT level) | `load_qm9` |
| chembl2k | Bioactive molecules with drug-like properties | `load_chembl2k` |
| broad6k | Bioactive molecules with drug-like properties | `load_broad6k` |
| toxcast | Toxicity of chemical compounds | `load_toxcast` |
| admet | Chemical absorption, distribution, metabolism, excretion, and toxicity | `load_admet` |
| gasperm | Six gas permeability properties for polymeric materials | `load_gasperm` |
| zinc250k | A common subset of ZINC dataset, which does not have labels and could be used for unconditional generation or virtual screening | `load_zinc250k` |

```python
from torch_molecule.datasets import load_qm9

# local_dir is the local path where the dataset will be saved
molecular_data = load_qm9(local_dir='torchmol_data')
smiles_list, property_np_array = molecular_data.data, molecular_data.target

# len(smiles_list): 133885
# Property array shape: (133885, 1)

# load_qm9 returns the target "gap" by default, but you can adjust it by passing new target_cols
target_cols = ['homo', 'lumo', 'gap']
molecular_data = load_qm9(local_dir='torchmol_data', target_cols=target_cols)
smiles_list, property_np_array = molecular_data.data, molecular_data.target

# the target could be None if loading an unlabeled dataset
from torch_molecule.datasets import load_zinc250k
molecular_data = load_zinc250k(local_dir='torchmol_data')
smiles_list = molecular_data.data
assert molecular_data.target is None
```

(We are actively adding more datasets. We welcome your suggestions and contributions on your datasets!)

### Fit a Model

After preparing the dataset, we can easily fit a model similar to how we use sklearn (actually, the coding is even simpler than sklearn, as we still need to do feature engineering in sklearn to convert molecule SMILES into vectors):

```python
from torch_molecule import GREAMolecularPredictor

split = int(0.8 * len(smiles_list))

grea = GREAMolecularPredictor(
    num_task=num_task,
    task_type="regression",
    evaluate_higher_better=False,
    verbose="progress_bar" #or "print_statement" recommended for jupyter notebooks, or "none"
)

# Fit with automatic hyperparameter tuning with 10 attempts, or implement .fit() with the default/manual hyperparameters
grea.autofit(
    X_train=smiles_list[:split],
    y_train=property_np_array[:split],
    X_val=smiles_list[split:],
    y_val=property_np_array[split:],
    n_trials=10,
)
```

### Checkpoints

`torch-molecule` provides checkpoint functions that can be interacted with on Hugging Face:

```python
from torch_molecule import GREAMolecularPredictor

repo_id = "user/repo_id"  # replace with your own Hugging Face username and repo_id

# Save the trained model to Hugging Face
grea.save_to_hf(
    repo_id=repo_id,
    task_id="qm9_grea",
    commit_message="Upload qm9_grea",
    private=False
)

# Load a pretrained checkpoint from Hugging Face
model = GREAMolecularPredictor()
model.load_from_hf(repo_id=repo_id, local_cache=f"{model_dir}/GREA_{task_name}.pt")

# Adjust model parameters and make predictions
model.set_params(verbose='none')
predictions = model.predict(smiles_list)
```

Or you can save the model to a local path:

```python
grea.save_to_local("qm9_grea.pt")

new_model = GREAMolecularPredictor()
new_model.load_from_local("qm9_grea.pt")
```

## List of Supported Models

### Predictive Models

| Model                | Reference           |
|----------------------|---------------------|
| GRIN                 | [Learning Repetition-Invariant Representations for Polymer Informatics. NeurIPS 2025.](https://arxiv.org/abs/2505.10726) |
| BFGNN                 | [Graph neural networks extrapolate out-of-distribution for shortest paths. March 2025](https://arxiv.org/abs/2503.19173) |
| SGIR                 | [Semi-Supervised Graph Imbalanced Regression. KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599497) |
| GREA                | [Graph Rationalization with Environment-based Augmentations. KDD 2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539347) |
| DIR                  | [Discovering Invariant Rationales for Graph Neural Networks. ICLR 2022](https://arxiv.org/abs/2201.12872) |
| SSR                  | [SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks. NeurIPS 2022](https://arxiv.org/abs/2206.07096) |
| IRM                  | [Invariant Risk Minimization (2019)](https://arxiv.org/abs/1907.02893) |
| RPGNN                | [Relational Pooling for Graph Representations. ICML 2019](https://arxiv.org/abs/1903.02541) |
| GNNs                 | [Graph Convolutional Networks. ICLR 2017](https://arxiv.org/abs/1609.02907) and [Graph Isomorphism Network. ICLR 2019](https://arxiv.org/abs/1810.00826) |
| Transformer (SMILES) | [Transformer (Attention is All You Need. NeurIPS 2017)](https://arxiv.org/abs/1706.03762) based on SMILES strings |
| LSTM (SMILES)        | [Long short-term memory (Neural Computation 1997)](https://ieeexplore.ieee.org/abstract/document/6795963) based on SMILES strings |

### Generative Models

| Model      | Reference           |
|------------|---------------------|
| DeFoG      | [DeFoG: Discrete Flow Matching for Graph Generation. ICML 2025](https://openreview.net/pdf?id=KPRIwWhqAZ) |
| Graph DiT  | [Graph Diffusion Transformers for Multi-Conditional Molecular Generation. NeurIPS 2024](https://openreview.net/forum?id=cfrDLD1wfO) |
| DiGress    | [DiGress: Discrete Denoising Diffusion for Graph Generation. ICLR 2023](https://openreview.net/forum?id=UaAD-Nu86WX) |
| GDSS       | [Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations. ICML 2022](https://proceedings.mlr.press/v162/jo22a/jo22a.pdf) |
| MolGPT     | [MolGPT: Molecular Generation Using a Transformer-Decoder Model. Journal of Chemical Information and Modeling 2021](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600) |
| JTVAE      | [Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML 2018.](https://proceedings.mlr.press/v80/jin18a) |
| GraphGA    | [A Graph-Based Genetic Algorithm and Its Application to the Multiobjective Evolution of Median Molecules. Journal of Chemical Information and Computer Sciences 2004](https://pubs.acs.org/doi/10.1021/ci034290p) |
| LSTM (SMILES)        | [Long short-term memory (Neural Computation 1997)](https://ieeexplore.ieee.org/abstract/document/6795963) based on SMILES strings |

### Representation Models

| Model        | Reference           |
|--------------|---------------------|
| MoAMa        | [Motif-aware Attribute Masking for Molecular Graph Pre-training. LoG 2024](https://arxiv.org/abs/2309.04589) |
| GraphMAE        | [GraphMAE: Self-Supervised Masked Graph Autoencoders. KDD 2022](https://arxiv.org/abs/2205.10803) |
| AttrMasking  | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| ContextPred  | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| EdgePred     | [Strategies for Pre-training Graph Neural Networks. ICLR 2020](https://arxiv.org/abs/1905.12265) |
| InfoGraph    | [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. ICLR 2020](https://arxiv.org/abs/1908.01000) |
| Supervised   | Supervised pretraining |
| Pretrained   | [GPT2-ZINC-87M](https://huggingface.co/entropy/gpt2_zinc_87m): GPT-2 based model (87M parameters) pretrained on ZINC dataset with ~480M SMILES strings. <br> [RoBERTa-ZINC-480M](https://huggingface.co/entropy/roberta_zinc_480m): RoBERTa based model (102M parameters) pretrained on ZINC dataset with ~480M SMILES strings. <br> [UniKi/bert-base-smiles](https://huggingface.co/unikei/bert-base-smiles): BERT model pretrained on SMILES strings. <br> [ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1): RoBERTa model pretrained on ZINC dataset with ~100k SMILES strings. <br> ChemBERTa series: Available in multiple sizes and training objectives (MLM/MTR). [ChemBERTa-5M-MLM](https://huggingface.co/DeepChem/ChemBERTa-5M-MLM), [ChemBERTa-5M-MTR](https://huggingface.co/DeepChem/ChemBERTa-5M-MTR), [ChemBERTa-10M-MLM](https://huggingface.co/DeepChem/ChemBERTa-10M-MLM), [ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR), [ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR). <br> ChemGPT series: GPT-Neo based models pretrained on PubChem10M dataset with SELFIES strings. [ChemGPT-1.2B](https://huggingface.co/ncfrey/ChemGPT-1.2B), [ChemGPT-4.7B](https://huggingface.co/ncfrey/ChemGPT-4.7M), [ChemGPT-19B](https://huggingface.co/ncfrey/ChemGPT-19M). |

<!-- ## Project Structure

See the structure of `torch_molecule` with the command `tree -L 2 torch_molecule -I '__pycache__|*.pyc|*.pyo|.git|old*'` -->

<!-- ## Plan

1. **Predictive Models**: Done: GREA, SGIR, IRM, GIN/GCN w/ virtual, DIR. SMILES-based LSTM/Transformers. TODO more
2. **Generative Models**: Done: Graph DiT, GraphGA, DiGress, GDS, MolGPT TODO: more
3. **Representation Models**: Done: MoAMa, AttrMasking, ContextPred, EdgePred. Many pretrained models from HF. TODO: checkpoints, more  -->

<!-- > **Note**: This project is in active development, and features may change. -->

## Acknowledgements

The project template was adapted from [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their contribution to the open-source community.
