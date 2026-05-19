# DMR-ARF: Dynamic Memory Reservoir-driven Adaptive Random Forest


## Overview

DMR-ARF is an online streaming classifier designed for imbalanced data streams with concept drift. It extends Adaptive Random Forest (ARF) with a class-stratified replay buffer that periodically re-exposes the model to rare-class samples, preventing minority-class forgetting without introducing temporal data leakage.

**Three key contributions:**
1. Empirical evidence of temporal data leakage in static SMOTE applied to streaming traffic data
2. The DMR mechanism: per-class sliding-window replay buffers (FIFO eviction) with periodic random experience replay
3. Rigorous prequential evaluation that strictly respects chronological order

**Main result:** On 385,000+ real-world US accident records, DMR-ARF raises fatal-accident (Severity 1) recall from 0% to 44% compared with the streaming baseline, while remaining the only streaming method with G-mean > 0.

---

## Requirements

```
python >= 3.12
river >= 0.21
scikit-learn >= 1.4
pandas >= 2.0
numpy >= 1.26
matplotlib >= 3.8
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

The repository intentionally does not include datasets, generated results, figures, virtual environments, or IDE files. A fresh clone contains only source code, dependency metadata, README, license, and Git ignore rules.

**Quick reproducible run without local data files:**

```bash
python experiments/run_all.py
```

By default, `run_all.py` runs datasets that are available through `river` and do not require files in `data/`: `Electricity`, `Phishing`, `Bananas`, and `ImageSegments`.

**Full paper reproduction:**

Prepare the external datasets below under `data/`, then edit `CONFIG['DATASETS']` in `experiments/run_all.py` to include the full list shown in that file.

| Dataset | Required file | Notes |
|---------|---------------|-------|
| US Accidents | `data/US_Accidents_March23.csv` or `data/accidents_model_ready.csv` | Download the raw Kaggle CSV. If only the raw file is present, `data/loader.py` automatically creates `accidents_model_ready.csv` using the fixed seed and preprocessing pipeline. |
| INSECTS | `data/INSECTS-abrupt_balanced_norm.csv` | Required only for the full multi-dataset run. |
| Airlines | `data/airlines.csv` | Optional; disabled in the default configuration. |
| KDDCup99 | `data/kddcup99_10_data.gz` | Optional local cache; otherwise fetched through scikit-learn. |
| CoverType | `data/covtype.data.gz` | Optional local cache; otherwise fetched through scikit-learn. |
| Electricity | `data/electricity.csv` | Optional local cache; otherwise loaded from `river`. |

Generated outputs are written to `results/`, which is ignored by Git.

---

## Project Structure

```
DMR-ARF/
├── data/
│   └── loader.py            # Dataset loading and preprocessing
├── models/
│   ├── dmr_arf.py           # DMR-ARF implementation
│   └── baselines.py         # Baseline models
├── experiments/
│   ├── run_all.py           # Main experiment runner
│   ├── ablation.py          # Hyperparameter ablation study
│   └── stats_test.py        # Friedman + Nemenyi statistical tests
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Usage

**Run the default reproducible experiment set:**

```bash
python experiments/run_all.py
```

**Run ablation study:**

```bash
python experiments/ablation.py
```

The ablation study uses US Accidents, so it requires either `data/US_Accidents_March23.csv` or `data/accidents_model_ready.csv`.

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 200 | Buffer capacity per minority class |
| T | 10 | Replay interval (instances) |
| B | 5 | Replay batch size per class |
| K | 30 | Number of ARF trees |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2025dmrarf,
  title   = {Dynamic Memory Reservoir-Driven Online Warning for 
             Streaming Traffic Accident Severity Prediction},
  author  = {Huang, Hongming and Xiang, Zijia},
  journal = {Applied Intelligence},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.

---
