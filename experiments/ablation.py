"""
experiments/ablation.py
消融实验：在 US_Accidents 上系统测试 DMR 超参数的影响

运行方式：
    python experiments/ablation.py

输出：results/ablation/ 目录下的 CSV 表格 + 热力图
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader    import load_us_accidents
from models.dmr_arf import DMRARF

# ─────────────────────────────────────────────
# 消融配置：每次只变一个参数，其余固定
# ─────────────────────────────────────────────
BASE_PARAMS = dict(n_models=30, replay_interval=10, replay_batch=5, seed=42)

ABLATION_CONFIGS = {
    # 变 buffer size M
    'buffer_size': {
        'param': 'memory_size',
        'values': [50, 100, 200, 500, 1000],
        'fixed': BASE_PARAMS,
    },
    # 变 replay interval T
    'replay_interval': {
        'param': 'replay_interval',
        'values': [5, 10, 20, 50, 100],
        'fixed': dict(n_models=30, memory_size=200, replay_batch=5, seed=42),
    },
    # 变 replay batch B
    'replay_batch': {
        'param': 'replay_batch',
        'values': [1, 3, 5, 10, 20],
        'fixed': dict(n_models=30, memory_size=200, replay_interval=10, seed=42),
    },
}

METRICS = ['macro_f1', 'gmean', 'auc']


def run_single(params: dict, minority_classes: list) -> dict:
    train_stream, test_stream, info = load_us_accidents()
    model = DMRARF(minority_classes=minority_classes, **params)
    result = model.run(train_stream, test_stream, info, verbose=False)
    # 还要记录 Severity 1 的 F1 和 Recall
    sev1_f1  = result.per_class_f1.get(0, 0.0)
    sev1_rec = result.per_class_recall.get(0, 0.0)
    return {
        'macro_f1':  result.macro_f1,
        'gmean':     result.gmean,
        'auc':       result.auc,
        'sev1_f1':   sev1_f1,
        'sev1_rec':  sev1_rec,
    }


def main():
    out_dir = ROOT / 'results' / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先跑一次获取 minority_classes
    _, _, info = load_us_accidents()
    minority = info['minority_classes']

    for study_name, cfg in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  消融实验：{study_name}")
        print(f"{'='*60}")

        rows = []
        for val in cfg['values']:
            params = dict(cfg['fixed'])
            params[cfg['param']] = val

            print(f"  {cfg['param']}={val} ...", end='', flush=True)
            metrics = run_single(params, minority)
            row = {cfg['param']: val, **metrics}
            rows.append(row)
            print(f"  Macro F1={metrics['macro_f1']:.4f}  "
                  f"Sev1-F1={metrics['sev1_f1']:.4f}  "
                  f"Sev1-Recall={metrics['sev1_rec']:.4f}")

        df = pd.DataFrame(rows)
        csv_path = out_dir / f'ablation_{study_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✅ 保存到 {csv_path}")

        # ── 生成折线图 ────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        param_label = cfg['param']

        for ax, metric in zip(axes, ['macro_f1', 'sev1_f1', 'sev1_rec']):
            ax.plot(df[cfg['param']], df[metric], 'o-', linewidth=2,
                    markersize=7, color='#4C72B0')
            ax.set_xlabel(param_label, fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'Effect of {param_label} on {metric}', fontsize=11)
            ax.grid(alpha=0.3, linestyle='--')

            # 标注最优值
            best_idx = df[metric].idxmax()
            ax.axvline(x=df[cfg['param']].iloc[best_idx],
                       color='red', linestyle='--', alpha=0.5,
                       label=f'Best={df[cfg["param"]].iloc[best_idx]}')
            ax.legend(fontsize=9)

        plt.suptitle(f'Ablation: {study_name}', fontsize=13, y=1.02)
        plt.tight_layout()
        fig_path = out_dir / f'ablation_{study_name}.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.savefig(str(fig_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  ✅ 图表保存到 {fig_path}")

    print(f"\n所有消融实验完成！结果在 {out_dir}")


if __name__ == '__main__':
    main()
