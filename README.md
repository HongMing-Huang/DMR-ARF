# DMR-ARF: Dynamic Memory Reservoir-driven Adaptive Random Forest



## Overview

DMR-ARF is an online streaming classifier designed for imbalanced data streams with concept drift. It extends Adaptive Random Forest (ARF) with a class-stratified replay buffer that periodically re-exposes the model to rare-class samples, preventing minority-class forgetting without introducing temporal data leakage.

**Three key contributions:**
1. Empirical evidence of temporal data leakage in static SMOTE applied to streaming traffic data
2. The DMR mechanism: per-class replay buffers with recency-biased random replacement
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
├── results/                 # Output figures and tables
├── requirements.txt
└── README.md
```

---

## Usage

**Run main experiments (US Accidents dataset):**

```bash
python experiments/run_all.py
```

**Run ablation study:**

```bash
python experiments/ablation.py
```

**Dataset:** The US Accidents dataset is publicly available at  
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents  
Download and place the CSV file under `data/` before running.

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

