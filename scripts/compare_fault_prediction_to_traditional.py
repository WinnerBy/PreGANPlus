#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比“故障预测/异常检测”能力：本文编码器 vs 传统方法可用的简单统计检测器。

核心思想：
- 传统启发式方法没有显式故障预测输出，因此使用统一的“事件代理标签”进行对比：
  以 metrics_with_interval.csv 中 slaviolations > 0 定义“性能退化/故障事件”（interval 级）。
- 为所有方法构造可比较的风险评分（risk score）：
  1) 统计阈值检测器（Baseline detector）：基于 time_series.npy 的 98 分位阈值，按主机 3 维指标判定异常主机数
  2) 编码器风险评分（Encoder）：仅对本文三种 GAN 方法计算（PreGAN/FPE_16, PreGANPlus/Transformer_16, PreGANPlusEnhanced/Transformer_16）
- 指标：
  - 同步检测：score_t 区分 event_t （AUROC、AP）
  - 一步提前：score_t 区分 event_{t+1}（AUROC、AP）

输出：
- final_results/summary/fault_prediction_comparison.csv
- final_results/summary/fault_prediction_comparison.md

运行：
  在 conda env pregan 中运行（含 numpy/torch/scipy）。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.PreGANSrc.src.utils import load_dataset, load_model, freeze


SCENARIO = "RPiEdge_BWGD2_100_16_16_1000_10000_300_5"
PERCENTILE = 98  # keep consistent with recovery/PreGANSrc/src/constants.py


@dataclass
class Row:
    method: str
    risk_source: str  # baseline_threshold / encoder
    intervals: int
    event_rate: float
    auroc_sync: Optional[float]
    ap_sync: Optional[float]
    auroc_next: Optional[float]
    ap_next: Optional[float]
    notes: str


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Mann–Whitney U based AUROC with tie handling via average ranks."""
    labels = labels.astype(int)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    scores = scores.reshape(-1).astype(float)
    n_pos = int(labels.sum())
    n = labels.size
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")  # ascending
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Average Precision (area under PR curve) without sklearn dependency."""
    labels = labels.astype(int).reshape(-1)
    scores = scores.astype(float).reshape(-1)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)  # desc
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / (tp + fp + 1e-12)
    # AP: average precision at each true positive
    ap = precision[y == 1].mean() if (y == 1).any() else None
    return None if ap is None else float(ap)


def load_event_series(run_dir: Path) -> Tuple[np.ndarray, str]:
    metrics_csv = run_dir / "metrics_with_interval.csv"
    df = pd.read_csv(metrics_csv)
    # keep last 100 intervals to match thesis stage-4 evaluation
    if len(df) > 100:
        df = df.iloc[-100:].copy()
    y = (df["slaviolations"].fillna(0).to_numpy() > 0).astype(int)
    return y, f"slaviolations>0 on last {len(df)} intervals"


def baseline_threshold_score(run_dir: Path) -> np.ndarray:
    """Return score per interval: number of anomalous hosts (0..16) based on percentile rule."""
    time_path = run_dir / "time_series.npy"
    x = np.load(time_path)  # [T, 48]
    if x.shape[0] > 100:
        x = x[-100:]
    # thresholds per dim across time
    thr = np.percentile(x, PERCENTILE, axis=0)
    anom_per_dim = x > thr  # [T, 48]
    # per host (3 dims)
    anom_host = []
    for i in range(0, x.shape[1], 3):
        anom_host.append(np.logical_or.reduce(anom_per_dim[:, i : i + 3], axis=1))
    anom_host = np.stack(anom_host, axis=1)  # [T, 16]
    score = anom_host.sum(axis=1).astype(float)  # [T]
    return score


def encoder_score(run_dir: Path, ckpt_folder: Path, ckpt_name: str, model_class: str) -> np.ndarray:
    """Return score per interval: mean anomaly probability across hosts (0..1)."""
    model, _, _, _ = load_model(str(ckpt_folder), ckpt_name, model_class)
    model.eval()
    freeze(model)

    # load dataset (includes normalization)
    time_windows, schedule_series, _, _ = load_dataset(str(run_dir), model)
    if time_windows.shape[0] > 100:
        time_windows = time_windows[-100:]
        schedule_series = schedule_series[-100:]

    if not isinstance(schedule_series, torch.Tensor):
        schedule_series = torch.tensor(schedule_series).double()
    else:
        schedule_series = schedule_series.double()
    time_windows = time_windows.double()

    scores = []
    with torch.no_grad():
        for t in range(time_windows.shape[0]):
            source_anomaly, _ = model(time_windows[t], schedule_series[t])
            if isinstance(source_anomaly, list):
                source_anomaly = torch.stack(source_anomaly, dim=0)  # [N,1,2]
            if isinstance(source_anomaly, torch.Tensor) and source_anomaly.dim() == 3 and source_anomaly.shape[-1] == 2:
                source_anomaly = source_anomaly.squeeze(1)  # [N,2]
            # anomaly probability (class 1)
            probs = torch.softmax(source_anomaly, dim=-1)[:, 1]
            scores.append(float(probs.mean().item()))
    return np.array(scores, dtype=float)


def eval_scores(y: np.ndarray, score: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    # sync
    auroc_s = roc_auc(y, score)
    ap_s = average_precision(y, score)
    # next-step (predict y_{t+1} using score_t)
    if len(y) >= 2:
        y_next = y[1:]
        s_prev = score[:-1]
        auroc_n = roc_auc(y_next, s_prev)
        ap_n = average_precision(y_next, s_prev)
    else:
        auroc_n = ap_n = None
    return auroc_s, ap_s, auroc_n, ap_n


def main() -> None:
    final_results = REPO_ROOT / "final_results"
    methods_all = ["CMODLB", "DFTM", "ECLB", "PCFT", "PreGAN", "PreGANPlus", "PreGANPlusEnhanced"]

    encoder_cfg = {
        "PreGAN": (REPO_ROOT / "recovery/PreGANSrc/checkpoints", "simulator_FPE_16.ckpt", "FPE_16"),
        "PreGANPlus": (REPO_ROOT / "recovery/PreGANSrc/checkpointsplus", "simulator_Transformer_16.ckpt", "Transformer_16"),
        "PreGANPlusEnhanced": (REPO_ROOT / "recovery/PreGANSrc/checkpointsplus", "simulator_Transformer_16.ckpt", "Transformer_16"),
    }

    rows: List[Row] = []
    event_notes: Dict[str, str] = {}

    for method in methods_all:
        data_root = final_results / "data" / method
        run_dirs = sorted(data_root.glob(f"run_*/{SCENARIO}"))
        if not run_dirs:
            rows.append(Row(method, "baseline_threshold", 0, 0.0, None, None, None, None, "run folder not found"))
            continue
        run_dir = run_dirs[0]
        y, note = load_event_series(run_dir)
        event_notes[method] = note

        # baseline threshold detector (available for all)
        s_base = baseline_threshold_score(run_dir)
        auroc_s, ap_s, auroc_n, ap_n = eval_scores(y, s_base)
        rows.append(
            Row(
                method=method,
                risk_source="baseline_threshold",
                intervals=len(y),
                event_rate=float(y.mean()) if len(y) else 0.0,
                auroc_sync=auroc_s,
                ap_sync=ap_s,
                auroc_next=auroc_n,
                ap_next=ap_n,
                notes="",
            )
        )

        # encoder score (only for GAN methods, but computed on their own runs)
        if method in encoder_cfg:
            ckpt_folder, ckpt_name, model_class = encoder_cfg[method]
            s_enc = encoder_score(run_dir, ckpt_folder, ckpt_name, model_class)
            auroc_s, ap_s, auroc_n, ap_n = eval_scores(y, s_enc)
            rows.append(
                Row(
                    method=method,
                    risk_source="encoder",
                    intervals=len(y),
                    event_rate=float(y.mean()) if len(y) else 0.0,
                    auroc_sync=auroc_s,
                    ap_sync=ap_s,
                    auroc_next=auroc_n,
                    ap_next=ap_n,
                    notes=f"{model_class}",
                )
            )

    out_csv = final_results / "summary" / "fault_prediction_comparison.csv"
    out_md = final_results / "summary" / "fault_prediction_comparison.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "risk_source",
                "intervals",
                "event_rate",
                "auroc_sync",
                "ap_sync",
                "auroc_next",
                "ap_next",
                "notes",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.risk_source,
                    r.intervals,
                    f"{r.event_rate:.6f}",
                    "" if r.auroc_sync is None else f"{r.auroc_sync:.6f}",
                    "" if r.ap_sync is None else f"{r.ap_sync:.6f}",
                    "" if r.auroc_next is None else f"{r.auroc_next:.6f}",
                    "" if r.ap_next is None else f"{r.ap_next:.6f}",
                    r.notes,
                ]
            )

    def fmt(x: Optional[float], nd: int = 4) -> str:
        return "" if x is None else f"{x:.{nd}f}"

    lines: List[str] = []
    lines.append("# 故障预测/异常检测与传统方法的可比性评估（由 final_results/data 解析生成）\n")
    lines.append("事件定义：以 `metrics_with_interval.csv` 中 `slaviolations>0` 表示 interval 级性能退化事件；统计口径取各运行结果的最后 100 个 interval。\n")
    lines.append(f"传统方法缺乏显式预测器，因此提供统一统计基线：按 time_series 的 {PERCENTILE} 分位阈值统计“异常主机数”。\n")
    lines.append("对于本文方法，同时报告编码器输出的风险评分（异常概率均值）对同一事件的区分能力。\n")
    lines.append("| 方法 | 风险信号 | intervals | 事件比例 | AUROC(同步) | AP(同步) | AUROC(一步提前) | AP(一步提前) | 备注 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.method} | {r.risk_source} | {r.intervals} | {fmt(r.event_rate)} | {fmt(r.auroc_sync)} | {fmt(r.ap_sync)} | {fmt(r.auroc_next)} | {fmt(r.ap_next)} | {r.notes} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out_csv)
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()

