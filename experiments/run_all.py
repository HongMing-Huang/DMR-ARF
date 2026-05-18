"""
experiments/run_all.py
一键运行所有实验，自动保存结果到 results/ 目录

用法：
    cd paper_refactored
    python experiments/run_all.py

可选参数（直接修改下方 CONFIG）：
    DATASETS    : 要跑的数据集名称列表
    MODELS      : 要跑的模型名称列表
    DMR_PARAMS  : DMR-ARF 默认参数
"""

import sys
import time
from pathlib import Path

import pandas as pd

# 把项目根目录加入 Python 路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader    import DATASET_REGISTRY
from models.dmr_arf import DMRARF
from models.baselines import BASELINE_REGISTRY

# ─────────────────────────────────────────────
# 实验配置（只改这里）
# ─────────────────────────────────────────────
CONFIG = {
    # Quick reproducible run: no local data files are required.
    # For the full paper run, replace this with:
    # ['US_Accidents', 'Electricity', 'KDDCup99', 'CoverType',
    #  'Phishing', 'Bananas', 'ImageSegments', 'INSECTS']
    # after preparing the external data files listed in README.md.
    'DATASETS': [
        'Electricity',
        'Phishing',
        'Bananas',
        'ImageSegments',
    ],

    # 要对比的基线
    'BASELINES': [
        'ARF_NoMemory',
        'OzaBagging_ADWIN',
        'LeveragingBagging',
        'ARF_OverSampling',
        'Static_XGBoost',
    ],

    # DMR-ARF 参数
    'DMR_PARAMS': {
        'n_models':        30,
        'memory_size':     200,
        'replay_interval': 10,
        'replay_batch':    5,
        'seed':            42,
    },

    # 结果保存目录
    'RESULTS_DIR': ROOT / 'results',

    # 是否打印详细进度
    'VERBOSE': True,
}


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def run_experiment(dataset_name: str, results_dir: Path):
    """对单个数据集跑所有模型，保存结果"""
    print(f"\n{'='*70}")
    print(f"  数据集：{dataset_name}")
    print(f"{'='*70}")

    loader_fn = DATASET_REGISTRY.get(dataset_name)
    if loader_fn is None:
        print(f"  [跳过] 未找到数据集 {dataset_name} 的加载器")
        return []

    all_results = []

    # ── 运行 DMR-ARF ──────────────────────────
    try:
        train_stream, test_stream, info = loader_fn()
        dmr = DMRARF(
            minority_classes=info['minority_classes'],
            **CONFIG['DMR_PARAMS'],
        )
        result = dmr.run(
            train_stream, test_stream, info,
            verbose=CONFIG['VERBOSE'],
        )
        all_results.append(result)
    except Exception as e:
        print(f"  [错误] DMR-ARF on {dataset_name}: {e}")

    # ── 运行基线 ──────────────────────────────
    for baseline_name in CONFIG['BASELINES']:
        baseline_fn = BASELINE_REGISTRY.get(baseline_name)
        if baseline_fn is None:
            continue
        try:
            train_stream, test_stream, info = loader_fn()
            result = baseline_fn(
                train_stream, test_stream, info,
                verbose=CONFIG['VERBOSE'],
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [错误] {baseline_name} on {dataset_name}: {e}")

    # ── 保存单数据集结果 ──────────────────────
    ds_dir = results_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        row = {
            'model':      r.model_name,
            'macro_f1':   round(r.macro_f1, 4),
            'gmean':      round(r.gmean,    4),
            'auc':        round(r.auc,      4) if r.auc == r.auc else 'nan',
            'train_time': round(r.train_time_sec, 1),
        }
        # 加入每类 F1
        for cls, val in r.per_class_f1.items():
            row[f'f1_class{cls}'] = round(val, 4)
        rows.append(row)

        # 保存每个模型的 rolling F1
        if r.rolling_macro_f1:
            pd.Series(r.rolling_macro_f1).to_csv(
                ds_dir / f'rolling_{r.model_name}.csv', index=False
            )

    # 汇总表
    summary = pd.DataFrame(rows)
    summary.to_csv(ds_dir / 'summary.csv', index=False)
    print(f"\n  ✅ 结果已保存到 {ds_dir}/summary.csv")
    print(summary.to_string(index=False))

    return all_results


def main():
    results_dir = Path(CONFIG['RESULTS_DIR'])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_results = {}
    total_start = time.time()

    for ds_name in CONFIG['DATASETS']:
        results = run_experiment(ds_name, results_dir)
        all_dataset_results[ds_name] = results

    # ── 跨数据集总汇总表 ──────────────────────
    all_rows = []
    for ds_name, results in all_dataset_results.items():
        for r in results:
            all_rows.append({
                'dataset':    ds_name,
                'model':      r.model_name,
                'macro_f1':   round(r.macro_f1, 4),
                'gmean':      round(r.gmean,    4),
                'auc':        round(r.auc,      4) if r.auc == r.auc else 'nan',
            })

    if all_rows:
        master = pd.DataFrame(all_rows)
        master.to_csv(results_dir / 'master_summary.csv', index=False)
        print(f"\n{'='*70}")
        print("  跨数据集总汇总表：")
        print(master.pivot_table(
            index='model', columns='dataset',
            values='macro_f1', aggfunc='first'
        ).to_string())
        print(f"\n  总耗时：{(time.time()-total_start)/60:.1f} 分钟")
        print(f"  完整结果保存于：{results_dir}/master_summary.csv")


if __name__ == '__main__':
    main()
