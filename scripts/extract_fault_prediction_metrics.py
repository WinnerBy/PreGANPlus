#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 final_results/data 的已归档运行结果中，离线提取“故障预测/异常检测”指标。

特点：
- 不重跑仿真，仅使用 time_series.npy + schedule_series.npy + 编码器 checkpoint 做前向推理
- 使用与训练代码一致的弱监督标注规则（按维度 98 分位阈值，见 recovery/PreGANSrc/src/utils.py::form_test_dataset）

输出：
- final_results/summary/fault_prediction_eval_from_data.csv
- final_results/summary/fault_prediction_eval_from_data.md

运行建议：
  需要在 pregan conda 环境中运行（包含 numpy/torch）。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovery.PreGANSrc.src.utils import load_dataset, load_model, freeze


@dataclass
class EvalResult:
    method: str
    run_dir: str
    intervals: int
    trigger_rate: float
    avg_pred_anom_nodes: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    notes: str = ""


def safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def f1_from_pr(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def evaluate_encoder_on_run(
    method: str,
    run_folder: Path,
    ckpt_folder: Path,
    ckpt_name: str,
    model_class_name: str,
) -> EvalResult:
    # Load model
    model, _, _, _ = load_model(str(ckpt_folder), ckpt_name, model_class_name)
    model.eval()
    freeze(model)

    # Load data (includes weak labels)
    time_windows, schedule_series, anomaly_labels, _ = load_dataset(str(run_folder), model)

    # Ensure tensors
    if not isinstance(schedule_series, torch.Tensor):
        schedule_series = torch.tensor(schedule_series).double()
    else:
        schedule_series = schedule_series.double()

    time_windows = time_windows.double()
    anomaly_labels = np.asarray(anomaly_labels).astype(int)  # [T, N]

    # Align with stage-4 evaluation convention (typically 100 intervals)
    T_total = int(time_windows.shape[0])
    if T_total > 100:
        time_windows = time_windows[-100:]
        schedule_series = schedule_series[-100:]
        anomaly_labels = anomaly_labels[-100:]
    T = int(time_windows.shape[0])
    trigger_cnt = 0
    pred_anom_nodes_sum = 0

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for t in range(T):
            source_anomaly, _ = model(time_windows[t], schedule_series[t])
            # source_anomaly is either a Tensor [N, 2] or a list of length N with Tensor[2]
            if isinstance(source_anomaly, list):
                source_anomaly = torch.stack(source_anomaly, dim=0)
            # tolerate shapes like [N, 1, 2]
            if isinstance(source_anomaly, torch.Tensor) and source_anomaly.dim() == 3 and source_anomaly.shape[-1] == 2:
                source_anomaly = source_anomaly.squeeze(1)
            # -> predicted label per node
            pred = torch.argmax(source_anomaly, dim=-1).cpu().numpy().astype(int)
            y = anomaly_labels[t].astype(int)

            pred_anom_nodes = int(pred.sum())
            pred_anom_nodes_sum += pred_anom_nodes
            if pred_anom_nodes > 0:
                trigger_cnt += 1

            tp += int(((pred == 1) & (y == 1)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = f1_from_pr(precision, recall)

    notes = ""
    if T_total != T:
        notes = f"evaluated last {T} of {T_total} intervals"

    return EvalResult(
        method=method,
        run_dir=str(run_folder),
        intervals=T,
        trigger_rate=trigger_cnt / T if T else 0.0,
        avg_pred_anom_nodes=pred_anom_nodes_sum / T if T else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        notes=notes,
    )


def main() -> None:
    repo_root = REPO_ROOT
    final_results = repo_root / "final_results"

    # mapping: method -> (ckpt_folder, ckpt_name, model_class_name)
    encoders = {
        "PreGAN": (repo_root / "recovery/PreGANSrc/checkpoints", "simulator_FPE_16.ckpt", "FPE_16"),
        "PreGANPlus": (repo_root / "recovery/PreGANSrc/checkpointsplus", "simulator_Transformer_16.ckpt", "Transformer_16"),
        "PreGANPlusEnhanced": (repo_root / "recovery/PreGANSrc/checkpointsplus", "simulator_Transformer_16.ckpt", "Transformer_16"),
    }

    # use the single run directory under final_results/data/<method>/*/<scenario>/
    results: List[EvalResult] = []
    for method, (ckpt_folder, ckpt_name, model_class_name) in encoders.items():
        data_root = final_results / "data" / method
        run_dirs = sorted(data_root.glob("run_*/RPiEdge_BWGD2_100_16_16_1000_10000_300_5"))
        if not run_dirs:
            results.append(EvalResult(method, "", 0, 0.0, 0.0, None, None, None, notes="run folder not found"))
            continue
        run_folder = run_dirs[0]
        results.append(
            evaluate_encoder_on_run(
                method=method,
                run_folder=run_folder,
                ckpt_folder=ckpt_folder,
                ckpt_name=ckpt_name,
                model_class_name=model_class_name,
            )
        )

    out_csv = final_results / "summary" / "fault_prediction_eval_from_data.csv"
    out_md = final_results / "summary" / "fault_prediction_eval_from_data.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "run_dir",
                "intervals",
                "trigger_rate",
                "avg_pred_anom_nodes",
                "precision",
                "recall",
                "f1",
                "notes",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.method,
                    r.run_dir,
                    r.intervals,
                    "" if r.trigger_rate is None else f"{r.trigger_rate:.6f}",
                    "" if r.avg_pred_anom_nodes is None else f"{r.avg_pred_anom_nodes:.6f}",
                    "" if r.precision is None else f"{r.precision:.6f}",
                    "" if r.recall is None else f"{r.recall:.6f}",
                    "" if r.f1 is None else f"{r.f1:.6f}",
                    r.notes,
                ]
            )

    # write MD
    def fmt(x: Optional[float], nd: int = 4) -> str:
        return "" if x is None else f"{x:.{nd}f}"

    lines: List[str] = []
    lines.append("# 故障预测（异常检测）离线评估汇总（由 final_results/data 解析生成）\n")
    lines.append("说明：该结果不依赖仿真重跑，仅基于已归档的时间序列与编码器 checkpoint 进行前向推理。\n")
    lines.append("弱监督标签采用与训练代码一致的百分位阈值规则（按维度 98 分位）。\n")
    lines.append("| 方法 | intervals | 触发率 | 平均预测异常节点数 | Precision | Recall | F1 | run_dir | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in results:
        lines.append(
            f"| {r.method} | {r.intervals} | {fmt(r.trigger_rate)} | {fmt(r.avg_pred_anom_nodes)} | {fmt(r.precision)} | {fmt(r.recall)} | {fmt(r.f1)} | {r.run_dir} | {r.notes} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()

