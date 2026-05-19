"""
models/baselines.py
所有对比基线模型，统一接口：run(train_stream, test_stream, dataset_info) -> ExperimentResult
"""

import time
import numpy as np
from sklearn.metrics import f1_score, classification_report

from river import forest, ensemble, imblearn
from models.dmr_arf import ExperimentResult, compute_gmean, compute_auc


def _run_prequential(model, train_stream, test_stream, dataset_info,
                     model_name, rolling_window=10_000, verbose=True):
    """通用 Prequential 评估框架，所有流式基线共用"""
    from river import metrics as river_metrics
    n_classes = dataset_info['n_classes']
    rolling_f1 = river_metrics.Rolling(
        river_metrics.MacroF1(),
        window_size=rolling_window
    )
    rolling_log = []
    n_train = 0

    if verbose:
        print(f"\n[{model_name}] 数据集={dataset_info['name']}")

    start = time.time()

    for x, y in train_stream:
        y_pred = model.predict_one(x)
        if y_pred is not None:
            rolling_f1.update(y, y_pred)
        model.learn_one(x, y)
        n_train += 1
        if n_train % rolling_window == 0:
            val = rolling_f1.get()
            rolling_log.append(round(val, 4))
            if verbose:
                print(f"  {n_train:>7d} 条 | 滚动 MacroF1={val:.4f}")

    train_time = time.time() - start

    y_true_list, y_pred_list, y_prob_list = [], [], []
    for x, y in test_stream:
        pred = model.predict_one(x)
        if pred is None:
            pred = 0
        y_true_list.append(y)
        y_pred_list.append(pred)
        proba = model.predict_proba_one(x) or {}
        prob_vec = [proba.get(c, 0.0) for c in range(n_classes)]
        y_prob_list.append(prob_vec)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    gmean    = compute_gmean(y_true, y_pred, labels=list(range(n_classes)))
    auc      = compute_auc(y_true, y_prob, n_classes)

    report  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_f1  = {int(k): v['f1-score'] for k, v in report.items()
               if k not in ('accuracy', 'macro avg', 'weighted avg')}
    per_rec = {int(k): v['recall']   for k, v in report.items()
               if k not in ('accuracy', 'macro avg', 'weighted avg')}

    if verbose:
        print(f"  Macro F1={macro_f1:.4f} G-mean={gmean:.4f} AUC={auc:.4f}")
        print(classification_report(
            y_true, y_pred,
            target_names=dataset_info.get('class_names'),
            zero_division=0,
        ))

    return ExperimentResult(
        dataset_name=dataset_info['name'], model_name=model_name,
        macro_f1=macro_f1, gmean=gmean, auc=auc,
        per_class_f1=per_f1, per_class_recall=per_rec,
        rolling_macro_f1=rolling_log,
        train_time_sec=train_time, n_train=n_train,
    )


# ─────────────────────────────────────────────
# 基线 1：ARF（无记忆水库）
# ─────────────────────────────────────────────
def run_arf_baseline(train_stream, test_stream, dataset_info, verbose=True):
    model = forest.ARFClassifier(n_models=30, seed=42, leaf_prediction="nba")
    return _run_prequential(
        model, train_stream, test_stream, dataset_info,
        model_name='ARF_NoMemory', verbose=verbose
    )


# ─────────────────────────────────────────────
# 基线 2：OzaBagging + ADWIN
# ─────────────────────────────────────────────
def run_oza_bagging(train_stream, test_stream, dataset_info, verbose=True):
    from river.tree import HoeffdingTreeClassifier
    model = ensemble.ADWINBaggingClassifier(
        model=HoeffdingTreeClassifier(),
        n_models=30,
        seed=42,
    )
    return _run_prequential(
        model, train_stream, test_stream, dataset_info,
        model_name='OzaBagging_ADWIN', verbose=verbose
    )


# ─────────────────────────────────────────────
# 基线 3：Leveraging Bagging
# ─────────────────────────────────────────────
def run_leveraging_bagging(train_stream, test_stream, dataset_info, verbose=True):
    from river.tree import HoeffdingTreeClassifier
    model = ensemble.LeveragingBaggingClassifier(
        model=HoeffdingTreeClassifier(),
        n_models=30,
        seed=42,
    )
    return _run_prequential(
        model, train_stream, test_stream, dataset_info,
        model_name='LeveragingBagging', verbose=verbose
    )


# ─────────────────────────────────────────────
# 基线 4：ARF + Online Random OverSampler
# ─────────────────────────────────────────────
def run_arf_oversampling(train_stream, test_stream, dataset_info, verbose=True):
    base_model = forest.ARFClassifier(n_models=30, seed=42, leaf_prediction="nba")
    n_classes = dataset_info['n_classes']
    desired_dist = {c: 1 / n_classes for c in range(n_classes)}
    model = imblearn.RandomOverSampler(
        classifier=base_model,
        desired_dist=desired_dist,
        seed=42,
    )
    return _run_prequential(
        model, train_stream, test_stream, dataset_info,
        model_name='ARF_OverSampling', verbose=verbose
    )


# ─────────────────────────────────────────────
# 基线 5：Static XGBoost（离线）
# ─────────────────────────────────────────────
def run_static_xgboost(train_stream, test_stream, dataset_info, verbose=True):
    """
    静态 XGBoost 基线：先收集所有训练数据，再 fit，再在测试集上 predict
    不使用 Prequential，代表传统离线方法
    """
    from xgboost import XGBClassifier
    import pandas as pd

    # 收集训练数据
    X_train_list, y_train_list = [], []
    for x, y in train_stream:
        X_train_list.append(list(x.values()))
        y_train_list.append(y)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    n_classes = dataset_info['n_classes']
    model = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    X_test_list, y_test_list = [], []
    for x, y in test_stream:
        X_test_list.append(list(x.values()))
        y_test_list.append(y)

    X_test = np.array(X_test_list)
    y_true = np.array(y_test_list)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    gmean    = compute_gmean(y_true, y_pred, labels=list(range(n_classes)))
    auc      = compute_auc(y_true, y_prob, n_classes)

    report  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_f1  = {int(k): v['f1-score'] for k, v in report.items()
               if k not in ('accuracy', 'macro avg', 'weighted avg')}
    per_rec = {int(k): v['recall']   for k, v in report.items()
               if k not in ('accuracy', 'macro avg', 'weighted avg')}

    if verbose:
        print(f"\n[Static_XGBoost] {dataset_info['name']} | "
              f"Macro F1={macro_f1:.4f} G-mean={gmean:.4f} AUC={auc:.4f}")

    return ExperimentResult(
        dataset_name=dataset_info['name'], model_name='Static_XGBoost',
        macro_f1=macro_f1, gmean=gmean, auc=auc,
        per_class_f1=per_f1, per_class_recall=per_rec,
        rolling_macro_f1=[],
        train_time_sec=train_time, n_train=len(y_train),
    )


# ─────────────────────────────────────────────
# 基线注册表
# ─────────────────────────────────────────────
BASELINE_REGISTRY = {
    'ARF_NoMemory':      run_arf_baseline,
    'OzaBagging_ADWIN':  run_oza_bagging,
    'LeveragingBagging': run_leveraging_bagging,
    'ARF_OverSampling':  run_arf_oversampling,
    'Static_XGBoost':    run_static_xgboost,
}
