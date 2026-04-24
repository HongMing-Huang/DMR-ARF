# paper_refactored — 项目结构说明

## 目录结构

```
.
├── data/
│   ├── loader.py          # 统一数据集加载器（所有数据集从这里读取）
│   ├── US_Accidents_March23.csv
│   ├── accidents_clean.csv
│   ├── accidents_model_ready.csv
│   ├── airlines.csv       # 需手动下载（见下方链接）
│   └── insects.arff       # 如后续启用可放在这里
├── models/
│   ├── dmr_arf.py         # DMR-ARF 主模型（你的方法）
│   └── baselines.py       # 所有对比基线
├── experiments/
│   ├── run_all.py         # 主入口：跑所有数据集 × 所有模型
│   ├── ablation.py        # 消融实验（M/T/B 超参数）
│   └── stats_test.py      # Friedman + Nemenyi 统计检验
├── results/               # 自动生成，存放所有实验结果
├── figures/               # 已整理的论文图表与导出图片
└── legacy/                # 旧版探索/原型脚本归档
```

---

## 安装依赖

```bash
pip install river xgboost scikit-learn pandas numpy matplotlib seaborn scipy scikit-posthocs
```

---

## 数据集下载

| 数据集 | 操作 |
|---|---|
| US_Accidents | 已整理到 `data/accidents_model_ready.csv` |
| Electricity | 已下载到 `data/electricity.zip` 并解压为 `data/electricity.csv` |
| Airlines | 已整理到 `data/airlines.csv`。项目内保留了原始 `data/airlines.arff`；README 里旧的 OpenML 直链并不是可直接用的 CSV 下载链接 |
| KDDCup99 | 已下载到 `data/kddcup99_10_data.gz`（10% 子集原始压缩源） |
| CoverType | 已下载到 `data/covtype.data.gz`（原始压缩源） |
| INSECTS（abrupt drift） | 已下载到 `data/INSECTS-abrupt_balanced_norm.csv`；旧 OpenML ARFF 链接当前仍是 404 |

---

## 运行顺序

```bash
# Step 1：跑主实验（约 1-3 小时）
python experiments/run_all.py

# Step 2：消融实验（约 30-60 分钟）
python experiments/ablation.py

# Step 3：统计检验（秒级）
python experiments/stats_test.py
```

---

## 已整理内容

- 主实验代码已归位到 `data/`、`models/`、`experiments/`
- 数据文件已归位到 `data/`
- 图表产物已归位到 `figures/`
- 旧版探索、预处理和原型代码已归档到 `legacy/`

---

## 结果文件说明

```
results/
├── master_summary.csv           # 所有模型 × 所有数据集汇总表
├── US_Accidents/
│   ├── summary.csv              # 该数据集上所有模型的指标
│   └── rolling_DMR-ARF.csv     # DMR-ARF 的滚动 Macro F1 曲线
├── ablation/
│   ├── ablation_buffer_size.csv
│   ├── ablation_replay_interval.csv
│   └── ablation_replay_batch.csv
└── stats/
    ├── average_ranks.csv
    ├── nemenyi_pvalues.csv
    └── critical_difference.pdf
```
