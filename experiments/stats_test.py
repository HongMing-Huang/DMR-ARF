"""
experiments/stats_test.py
Friedman + Nemenyi 统计显著性检验，生成 Critical Difference 图

前提：先跑完 run_all.py，生成 results/master_summary.csv

运行方式：
    python experiments/stats_test.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def friedman_test(data: np.ndarray):
    """data shape: (n_datasets, n_models)，返回 stat, p_value"""
    from scipy.stats import friedmanchisquare
    return friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])


def nemenyi_test(data: np.ndarray):
    """返回 p-value 矩阵 (n_models x n_models)"""
    try:
        import scikit_posthocs as sp
        df = pd.DataFrame(data)
        return sp.posthoc_nemenyi_friedman(df)
    except ImportError:
        print("  [提示] 安装 scikit-posthocs: pip install scikit-posthocs")
        return None


def plot_cd_diagram(avg_ranks: dict, n_datasets: int, alpha: float = 0.05,
                    save_path: Path = None):
    """
    绘制 Critical Difference 图
    avg_ranks: {'model_name': average_rank, ...}
    """
    import math

    k = len(avg_ranks)
    cd = _critical_difference(k, n_datasets, alpha)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # 排名轴（从左到右：rank 1 = 最好）
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    spacing = k / (k + 1)
    positions = {}

    for i, (name, rank) in enumerate(sorted_models):
        x = 0.5 + (i + 1) * spacing
        positions[name] = x
        ax.text(x, 1.5, name, ha='center', va='bottom', fontsize=9, rotation=30)
        ax.plot(x, 1.2, 'o', color='#4C72B0', markersize=8)
        ax.text(x, 1.0, f'{rank:.2f}', ha='center', va='top', fontsize=8)

    # CD 标注
    ax.annotate(
        f'CD = {cd:.3f} (α={alpha})',
        xy=(0.5, 0.3), fontsize=10,
        xycoords='axes fraction', ha='left',
    )

    ax.set_title('Critical Difference Diagram (Nemenyi test)', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.savefig(str(save_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
        print(f"  CD图保存到 {save_path}")
    plt.close()


def _critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Nemenyi CD 值（近似公式）
    q_alpha 来自 studentized range distribution / sqrt(2)
    """
    # 常用 q_alpha 值（k 个分类器，alpha=0.05）
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_table.get(k, 2.728)
    import math
    return q * math.sqrt(k * (k + 1) / (6 * n))


def main():
    results_file = ROOT / 'results' / 'master_summary.csv'
    if not results_file.exists():
        print(f"[错误] 找不到 {results_file}，请先运行 run_all.py")
        return

    out_dir = ROOT / 'results' / 'stats'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_file)
    print(f"  读取 {len(df)} 条记录，数据集数={df['dataset'].nunique()}")

    # 每个数据集、每个模型的 macro_f1
    pivot = df.pivot_table(index='dataset', columns='model', values='macro_f1')
    print("\n  Macro F1 透视表：")
    print(pivot.to_string())

    # ── Friedman 检验 ─────────────────────────
    data = pivot.values  # (n_datasets, n_models)
    stat, p = friedman_test(data)
    print(f"\n  Friedman 检验：chi2={stat:.4f}, p={p:.4f}")
    if p < 0.05:
        print("  → p < 0.05，各模型性能存在显著差异，继续 Nemenyi 事后检验")
    else:
        print("  → p >= 0.05，无显著差异（可能数据集太少）")

    # ── 平均排名 ──────────────────────────────
    ranks = pivot.rank(axis=1, ascending=False)  # F1 越高排名越好
    avg_ranks = ranks.mean().to_dict()
    print("\n  各模型平均排名（越小越好）：")
    for model, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"    {model:<25} {rank:.3f}")

    # 保存排名表
    rank_df = pd.DataFrame([avg_ranks]).T.rename(columns={0: 'avg_rank'})
    rank_df = rank_df.sort_values('avg_rank')
    rank_df.to_csv(out_dir / 'average_ranks.csv')

    # ── Nemenyi 事后检验 ──────────────────────
    nemenyi_p = nemenyi_test(data)
    if nemenyi_p is not None:
        nemenyi_p.columns = pivot.columns
        nemenyi_p.index   = pivot.columns
        nemenyi_p.to_csv(out_dir / 'nemenyi_pvalues.csv')
        print("\n  Nemenyi p-value 矩阵已保存")

        # ── 热力图 ────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(
            nemenyi_p, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=0.1, ax=ax,
            linewidths=0.5,
        )
        ax.set_title('Nemenyi Post-hoc Test p-values\n(green = significant difference)', fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / 'nemenyi_heatmap.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(out_dir / 'nemenyi_heatmap.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  热力图保存到 {out_dir}/nemenyi_heatmap.pdf")

    # ── CD 图 ─────────────────────────────────
    n_datasets = len(pivot)
    plot_cd_diagram(
        avg_ranks, n_datasets, alpha=0.05,
        save_path=out_dir / 'critical_difference.pdf',
    )

    print(f"\n  统计检验完成！结果保存于 {out_dir}")


if __name__ == '__main__':
    main()
