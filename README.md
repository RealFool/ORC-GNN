# ORC-GNN: Open-Set Risk Collaborative Consistency Graph Neural Network

ORC-GNN is an open-set recognition (OSR) framework for multi-class classification of psychiatric disorders from resting-state fMRI. It integrates whole-brain and hemispheric functional connectivity (FC) graphs, performs multi-level feature fusion, and learns adaptive decision boundaries for known classes while detecting unknown classes.

![ORC-GNN architecture](image/model.jpg)

## Paper

- Title: ORC-GNN: A novel open-set recognition based on graph neural network for multi-class classification of psychiatric disorders
- Link: [https://www.sciencedirect.com/science/article/abs/pii/S1566253524006651]([https://www.sciencedirect.com/science/article/abs/pii/S1566253524006651]())

### Abstract

Open-set recognition (OSR) refers to the challenge of introducing classes not seen during model training into the test set. This issue is particularly critical in the medical field due to incomplete data collection and the continuous emergence of new and rare diseases. Medical OSR techniques necessitate not only the accurate classification of known cases but also the ability to detect unknown cases and send the corresponding information to experts for further diagnosis. However, there is a significant research gap in the current medical OSR field, which not only lacks research methods for OSR in psychiatric disorders, but also lacks detailed procedures for OSR evaluation based on neuroimaging. To address the challenges associated with the OSR of psychiatric disorders, we propose a method named the open-set risk collaborative consistency graph neural network (ORC-GNN). First, functional connectivity (FC) is used to extract measurable representations in the deep feature space by coordinating hemispheric and whole-brain networks, thereby achieving multi-level brain network feature fusion and regional communication. Subsequently, these representations are used to guide the model to adaptively learn the decision boundaries for known classes using the instance-level density awareness and to identify samples outside these boundaries as unknown. We introduce a novel open-risk margin loss (ORML) to balance empirical risk and open-space risk; this approach makes open-space risk quantifiable through the introduction of open-risk term. We evaluate our method using an integrated multi-class dataset and a tailored experimental protocol suited for psychiatric disorder-related OSR challenges. Compared to state-of-the-art techniques, ORC-GNN demonstrates significant performance improvements and yields important clinically interpretative information regarding the shared and distinct characteristics of multiple psychiatric disorders.

## Repository Structure

```
ORC-GNN/
  01_BrainNet_Generate.py        # Build whole-brain + hemispheric graphs from ROI time series
  02_UpperFeat_Generate.py       # Save upper-triangular features (stored as .h5)
  03_Run_Openset.py              # Main training + open-set evaluation script
  get_weight.py                  # Weight fusion / ROI contribution utilities
  show_loss.py                   # Plot boundary learning loss / delta curves
  model/                         # ORC-GNN models and layers
  utils/                         # Dataset reader and evaluation utilities
  Glasso/                        # Group Lasso implementation (bundled)
  data/                          # Metadata CSVs (example labels/phenotypes)
  image/model.jpg                # Architecture figure
  requestment.txt                # Python dependencies (see notes below)
```

## Environment

### Dependencies

Install Python packages:

```bash
pip install -r requestment.txt
```

## Data Preparation

This repo expects two kinds of inputs:

1) **ROI time series** for each subject.

- Expected variable name: `ROISignals`
- Shape: `(T, N)` where `T` is timepoints and `N` is the number of ROIs (typically 116 for AAL).

2) **Label/phenotype metadata** used to map subject IDs to class labels.

- Example CSV files are under `data/` (their exact columns may differ across experiments).

### Expected Folder Layout (typical)

The scripts contain several experiment-specific default paths. A typical layout is:

```
data/
  Functional/
    closeset_.../                # known classes (ROI time series .mat)
    openset/CLASS_NAME/          # unknown class time series
  BrainNet.../
    <lambda_group>/raw/          # saved adjacency matrices (*.mat)
      harfbrain/raw/             # saved hemispheric/bipartite networks (*.mat)
  UpperFeat.../                  # saved upper features (*.h5)
```

If your layout differs, update the corresponding `--Fun_dir`, `--Save_dir`, `--mat_dir`, and `--upper_dir` arguments in scripts, or edit the default paths inside:

- `01_BrainNet_Generate.py`
- `02_UpperFeat_Generate.py`
- `03_Run_Openset.py`
- `utils/sparse_net_reader.py` (label/phenotype CSV path and ROI time series root)

## Running ORC-GNN

Run the pipeline from the `ORC-GNN/` directory.

### 1) Build brain networks (whole-brain + hemispheric + bipartite)

```bash
python 01_BrainNet_Generate.py
```

Key arguments:

- `--Fun_dir`: directory containing ROI time-series `.mat` files for known classes
- `--Save_dir`: output directory for generated brain networks
- `--Lambda_group`: group-lasso regularization strength (some configs use a string like `0.3_0.5`)
- `--Thres`, `--BiGraph_Ratio`, `--WoGraph_Ratio`: sparsification controls

Outputs:

- Whole-brain networks: `<Save_dir>/<Lambda_group>/raw/*_net_*.mat` with key `Brainnetwork`
- Hemispheric/bipartite networks: `<Save_dir>/<Lambda_group>/raw/harfbrain/raw/*.mat`

### 2) Generate upper features

```bash
python 02_UpperFeat_Generate.py
```

Outputs:

- `<data_dir>/*.h5` written via `deepdish`, each file stores `UpperFeat`.

### 3) Train and evaluate open-set recognition

```bash
python 03_Run_Openset.py
```

What it does (high level):

- Loads graph data via `utils/brainnetwork_reader.py` / `utils/sparse_net_reader.py`
- Performs cross-validation on known classes
- Learns class centroids and adaptive margins (`delta`) via boundary learning
- Detects unknown samples by rejecting those outside learned boundaries

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{li2025orc,
  title={ORC-GNN: A novel open set recognition based on graph neural network for multi-class classification of psychiatric disorders},
  author={Li, Yaqin and Dong, Yihong and Peng, Shoubo and Gao, Linlin and Xin, Yu},
  journal={Information Fusion},
  volume={117},
  pages={102887},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
