"""
models/dmr_arf.py
DMR-ARF：Dynamic Memory Reservoir-driven Adaptive Random Forest
重构版本 —— 支持多数据集、可配置参数、返回完整结果对象
"""

import time
import random
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any

from river import forest, metrics
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ─────────────────────────────────────────────
# 结果容器
# ─────────────────────────────────────────────
@dataclass
class ExperimentResult:
    dataset_name:    str
    model_name:      str
    macro_f1:        float
    gmean:           float
    auc:             float          # macro OvR AUC
    per_class_f1:    Dict[int, float] = field(default_factory=dict)
    per_class_recall:Dict[int, float] = field(default_factory=dict)
    rolling_macro_f1:List[float]    = field(default_factory=list)
    train_time_sec:  float          = 0.0
    n_train:         int            = 0
    params:          Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def compute_gmean(y_true, y_pred) -> float:
    """几何平均数（G-mean），适用于多类不平衡"""
    cm = confusion_matrix(y_true, y_pred)
    # 每类的 recall = TP / (TP + FN)
    per_class_recall = []
    for i in range(len(cm)):
        denom = cm[i].sum()
        per_class_recall.append(cm[i, i] / denom if denom > 0 else 0.0)
    per_class_recall = np.array(per_class_recall)
    # 过滤掉 0（该类在测试集中不存在）
    nonzero = per_class_recall[per_class_recall > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.prod(nonzero) ** (1.0 / len(nonzero)))


def compute_auc(y_true, y_prob, n_classes) -> float:
    """宏平均 One-vs-Rest AUC"""
    try:
        from sklearn.preprocessing import label_binarize
        classes = list(range(n_classes))
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1:          # 二分类退化
            return roc_auc_score(y_true, y_prob[:, 1])
        return roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────
# DMR-ARF 主类
# ─────────────────────────────────────────────
class DMRARF:
    """
    Dynamic Memory Reservoir-driven Adaptive Random Forest

    参数
    ----
    n_models        : ARF 中 Hoeffding Tree 的数量
    memory_size     : 每个少数类的 buffer 容量 M
    replay_interval : 每隔 T 条样本触发一次回放
    replay_batch    : 每次每个少数类回放 B 条样本
    minority_classes: 需要保护的少数类标签列表（0 起始）
    seed            : 随机种子
    """

    def __init__(
        self,
        n_models: int = 30,
        memory_size: int = 200,
        replay_interval: int = 10,
        replay_batch: int = 5,
        minority_classes: List[int] = None,
        seed: int = 42,
    ):
        self.n_models = n_models
        self.memory_size = memory_size
        self.replay_interval = replay_interval
        self.replay_batch = replay_batch
        self.minority_classes = minority_classes or []
        self.seed = seed
        random.seed(seed)

        # 内部状态
        self._model = None
        self._memory: Dict[int, deque] = {}
        self._counter = 0

    # ── 初始化 ──────────────────────────────
    def _init(self):
        self._model = forest.ARFClassifier(
            n_models=self.n_models,
            seed=self.seed,
            leaf_prediction="nba",   # 增强少数类灵敏度
        )
        self._memory = {
            c: deque(maxlen=self.memory_size)
            for c in self.minority_classes
        }
        self._counter = 0

    # ── 单步更新 ─────────────────────────────
    def _step(self, x: dict, y: int):
        """处理一条样本：学习 + 水库更新 + 可能触发回放"""
        # Step B：正常学习
        self._model.learn_one(x, y)

        # Step C：水库蓄水
        if y in self._memory:
            self._memory[y].append((x, y))

        # Step D：高频回放
        self._counter += 1
        if self._counter % self.replay_interval == 0:
            for cls, buf in self._memory.items():
                if len(buf) > 0:
                    batch = random.sample(
                        list(buf),
                        min(len(buf), self.replay_batch)
                    )
                    for xm, ym in batch:
                        self._model.learn_one(xm, ym)

    # ── 主训练+评估流程 ───────────────────────
    def run(
        self,
        train_stream,
        test_stream,
        dataset_info: dict,
        rolling_window: int = 10_000,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        执行完整实验流程，返回 ExperimentResult

        参数
        ----
        train_stream   : 训练数据迭代器，每次 yield (x_dict, y_int)
        test_stream    : 测试数据迭代器
        dataset_info   : loader.py 返回的 info 字典
        rolling_window : 滚动 Macro F1 的窗口大小
        verbose        : 是否打印进度
        """
        self._init()

        dataset_name = dataset_info['name']
        n_classes    = dataset_info['n_classes']

        rolling_f1   = metrics.MacroF1()
        rolling_log  = []
        step_counter = 0
        n_train      = 0

        if verbose:
            print(f"\n[DMR-ARF] 数据集={dataset_name} | "
                  f"M={self.memory_size} T={self.replay_interval} B={self.replay_batch}")
            print("─" * 60)

        start = time.time()

        # ── Prequential 训练 ──────────────────
        for x, y in train_stream:
            # Step A：先预测，再评估
            y_pred = self._model.predict_one(x)
            if y_pred is not None:
                rolling_f1.update(y, y_pred)

            # Step B-D：学习 + 水库 + 回放
            self._step(x, y)
            n_train += 1
            step_counter += 1

            # 记录滚动 F1
            if step_counter % rolling_window == 0:
                val = rolling_f1.get()
                rolling_log.append(round(val, 4))
                if verbose:
                    print(f"  已处理 {step_counter:>7d} 条 | 滚动 Macro F1 = {val:.4f}")

        train_time = time.time() - start

        # ── 静态测试集评估 ────────────────────
        y_true_list, y_pred_list, y_prob_list = [], [], []

        for x, y in test_stream:
            pred = self._model.predict_one(x)
            if pred is None:
                pred = max(self.minority_classes, default=0)
            y_true_list.append(y)
            y_pred_list.append(pred)

            # 概率（用于 AUC）
            proba = self._model.predict_proba_one(x)
            prob_vec = [proba.get(c, 0.0) for c in range(n_classes)]
            y_prob_list.append(prob_vec)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_prob = np.array(y_prob_list)

        # ── 计算指标 ──────────────────────────
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        gmean    = compute_gmean(y_true, y_pred)
        auc      = compute_auc(y_true, y_prob, n_classes)

        # 每类 F1 和 Recall
        report  = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0,
        )
        per_f1  = {int(k): v['f1-score'] for k, v in report.items()
                   if k not in ('accuracy', 'macro avg', 'weighted avg')}
        per_rec = {int(k): v['recall']   for k, v in report.items()
                   if k not in ('accuracy', 'macro avg', 'weighted avg')}

        if verbose:
            print(f"\n{'='*60}")
            print(f"[DMR-ARF] {dataset_name} 结果汇总")
            print(f"  Macro F1 = {macro_f1:.4f} | G-mean = {gmean:.4f} | AUC = {auc:.4f}")
            print(f"  训练耗时 = {train_time:.1f}s | 训练样本数 = {n_train}")
            print(classification_report(
                y_true, y_pred,
                target_names=dataset_info.get('class_names'),
                zero_division=0,
            ))

        return ExperimentResult(
            dataset_name     = dataset_name,
            model_name       = 'DMR-ARF',
            macro_f1         = macro_f1,
            gmean            = gmean,
            auc              = auc,
            per_class_f1     = per_f1,
            per_class_recall = per_rec,
            rolling_macro_f1 = rolling_log,
            train_time_sec   = train_time,
            n_train          = n_train,
            params           = {
                'n_models':        self.n_models,
                'memory_size':     self.memory_size,
                'replay_interval': self.replay_interval,
                'replay_batch':    self.replay_batch,
            },
        )
