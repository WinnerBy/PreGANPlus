#!/usr/bin/env python3
"""
根据 Stage3 挑选 5 次汇总数据绘制论文第五章所需图表。
数据来源: experiment_logs/stage3/stage3_aggregated_5runs_selected.csv
输出目录: experiment_logs/stage3/plots/ 或通过 --out-dir 指定。

用法:
  python scripts/plot_stage3_results.py              # 默认输出 PNG，便于插入 Word/论文
  python scripts/plot_stage3_results.py --fmt pdf    # 输出 PDF（适合 LaTeX）
  python scripts/plot_stage3_results.py --fmt both    # 同时输出 PNG 与 PDF
  python scripts/plot_stage3_results.py --out-dir results/stage3_plots
"""

import csv
import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，避免在无显示环境下崩溃
import numpy as np
import matplotlib.pyplot as plt

# 可选：与 grapher.py 一致的 science/ieee 风格（若未安装 SciencePlots 则用默认）
try:
    plt.style.use(["science", "ieee"])
except (OSError, KeyError):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

# 推荐安装一款中英文均支持的字体，安装后中英文统一、观感更好：
#   - Noto Sans SC（谷歌思源黑体简中）: https://fonts.google.com/noto/specimen/Noto+Sans+SC  免费、清晰、略紧凑
#   - 更纱黑体 Sarasa Gothic: https://github.com/be5invis/Sarasa-Gothic/releases  免费、偏紧凑、适合图表
# 安装后把字体名放在下面列表首位；未安装时回退到系统字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans SC", "Sarasa Gothic SC", "Sarasa UI SC",
    "Microsoft YaHei", "PingFang SC", "SimHei", "STHeiti", "WenQuanYi Micro Hei", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# 方法显示名（中文作业用）
METHOD_LABELS = {
    "PreGAN": "FPE-GAN",
    "PreGANPlus": "TF-GAN",
    "PreGANPlusEnhanced": "MAMO-GAN",
    "AblationNoTransformer": "无 Transformer",
    "AblationNoGAT": "无 GAT",
    "AblationNoMigrationAware": "无迁移感知",
    "AblationNoMultiObjective": "无多目标",
    "CMODLB": "CMODLB",
    "DFTM": "DFTM",
    "ECLB": "ECLB",
    "PCFT": "PCFT",
}

# 颜色（与 grapher 近似，可区分 GAN / 传统 / 消融）
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
]


def _format_bar_label(mean_val, std_val, decimals=1, show_std=False):
    """柱顶显示的数值：仅均值（避免重叠）；show_std=True 时可显示 mean±std。"""
    if decimals == 0:
        m = int(round(mean_val))
        return f"{m}±{int(round(std_val))}" if (show_std and std_val) else f"{m}"
    fmt = f".{decimals}f"
    m, s = mean_val, std_val
    return f"{m:{fmt}}±{s:{fmt}}" if (show_std and s) else f"{m:{fmt}}"


def _add_bar_value_labels(ax, x, means, stds, decimals=1, fontsize=8, show_std=False):
    """在柱顶上方标注数值（默认仅均值，避免重叠）；略扩展 y 轴上限以免裁切。"""
    y_max = (means + stds).max() if len(means) else 0
    y_span = y_max * 0.04 if y_max > 0 else 1
    for i in range(len(x)):
        label_text = _format_bar_label(means[i], stds[i], decimals, show_std=show_std)
        ax.text(x[i], means[i] + stds[i] + y_span, label_text,
                ha="center", va="bottom", fontsize=fontsize, rotation=0)
    # 为柱顶数值留出空间
    top = (means + stds).max() + y_span * 3 if len(means) else 0
    cur = ax.get_ylim()[1]
    if top > 0 and top > cur * 0.95:
        ax.set_ylim(0, max(cur, top * 1.08))


def parse_mean_std(s):
    """解析 'mean±std' 或 'N/A'，返回 (mean, std) 或 (None, None)。"""
    if not s or s.strip() == "N/A":
        return None, None
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", s.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except ValueError:
        return None, None


def load_aggregated(csv_path):
    """加载聚合 CSV，返回 list of dict，每个 dict 含 method 及各指标的 mean/std。"""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = r.get("method", "").strip()
            if not method:
                continue
            row = {"method": method, "label": METHOD_LABELS.get(method, method)}
            for col in ["nummigrations", "energytotalinterval", "slaviolations",
                        "sla_violation_pct", "sla_violation_pct_over_created"]:
                mean, std = parse_mean_std(r.get(col, ""))
                row[col] = (mean, std)
            rows.append(row)
    return rows


def _decimals_for_metric(metric_key):
    """按指标类型返回柱顶数值的小数位数。"""
    if "nummigrations" in metric_key or "numdestroyed" in metric_key or "slaviolations" in metric_key:
        return 0
    if "sla_violation" in metric_key or "energy" in metric_key:
        return 2
    return 1


def bar_plot(rows, metric_key, ylabel, title=None, order_by_asc=True, out_path=None,
             figsize=(7.5, 3.5), rot=35, scale=1.0, value_decimals=None):
    """绘制柱状图（带误差条）。metric_key 如 'nummigrations', 'energytotalinterval'。scale 用于 y 轴缩放（如能耗 /1e6）。"""
    # 过滤有效数据并排序
    valid = [r for r in rows if r[metric_key][0] is not None]
    if not valid:
        return
    if order_by_asc:
        valid = sorted(valid, key=lambda r: r[metric_key][0])
    labels = [r["label"] for r in valid]
    means = np.array([r[metric_key][0] for r in valid]) * scale
    stds = np.array([r[metric_key][1] or 0 for r in valid]) * scale

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=COLORS[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, None)
    dec = value_decimals if value_decimals is not None else _decimals_for_metric(metric_key)
    _add_bar_value_labels(ax, x, means, stds, decimals=dec)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def bar_plot_ablation(rows, metric_key, ylabel, title=None, out_path=None, figsize=(6, 3.5), rot=25, scale=1.0):
    """仅绘制 MAMO-GAN + 4 个消融变体。"""
    ablation_methods = [
        "PreGANPlusEnhanced", "AblationNoTransformer", "AblationNoGAT",
        "AblationNoMigrationAware", "AblationNoMultiObjective",
    ]
    by_method = {r["method"]: r for r in rows}
    selected = []
    for m in ablation_methods:
        if m in by_method and by_method[m][metric_key][0] is not None:
            selected.append(by_method[m])
    if not selected:
        return
    labels = [r["label"] for r in selected]
    means = np.array([r[metric_key][0] for r in selected]) * scale
    stds = np.array([r[metric_key][1] or 0 for r in selected]) * scale

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    colors = [COLORS[0]] + COLORS[1:5]  # MAMO-GAN 突出色 + 4 消融
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, None)
    dec = _decimals_for_metric(metric_key)
    _add_bar_value_labels(ax, x, means, stds, decimals=dec)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def bar_plot_group1(rows, metric_key, ylabel, title=None, out_path=None, figsize=(5.5, 3.5), rot=25, scale=1.0):
    """Group 1: FPE-GAN vs 传统方法 (CMODLB, DFTM, ECLB, PCFT)。"""
    group = ["PreGAN", "CMODLB", "DFTM", "ECLB", "PCFT"]
    by_method = {r["method"]: r for r in rows}
    selected = [by_method[m] for m in group if m in by_method and by_method[m][metric_key][0] is not None]
    if not selected:
        return
    labels = [r["label"] for r in selected]
    means = np.array([r[metric_key][0] for r in selected]) * scale
    stds = np.array([r[metric_key][1] or 0 for r in selected]) * scale
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=COLORS[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, None)
    dec = _decimals_for_metric(metric_key)
    _add_bar_value_labels(ax, x, means, stds, decimals=dec)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def bar_plot_group2(rows, metric_key, ylabel, title=None, out_path=None, figsize=(4.5, 3), rot=20, scale=1.0):
    """Group 2: TF-GAN vs FPE-GAN。"""
    group = ["PreGAN", "PreGANPlus"]
    by_method = {r["method"]: r for r in rows}
    selected = [by_method[m] for m in group if m in by_method and by_method[m][metric_key][0] is not None]
    if not selected:
        return
    labels = [r["label"] for r in selected]
    means = np.array([r[metric_key][0] for r in selected]) * scale
    stds = np.array([r[metric_key][1] or 0 for r in selected]) * scale
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=COLORS[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, None)
    dec = _decimals_for_metric(metric_key)
    _add_bar_value_labels(ax, x, means, stds, decimals=dec)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def bar_plot_group3(rows, metric_key, ylabel, title=None, out_path=None, figsize=(7, 3.5), rot=30, scale=1.0):
    """Group 3: MAMO-GAN vs 其他 GAN 与部分传统方法。"""
    group = ["PreGANPlusEnhanced", "PreGANPlus", "PreGAN", "CMODLB", "DFTM", "ECLB", "PCFT"]
    by_method = {r["method"]: r for r in rows}
    selected = [by_method[m] for m in group if m in by_method and by_method[m][metric_key][0] is not None]
    if not selected:
        return
    labels = [r["label"] for r in selected]
    means = np.array([r[metric_key][0] for r in selected]) * scale
    stds = np.array([r[metric_key][1] or 0 for r in selected]) * scale
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=COLORS[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, None)
    dec = _decimals_for_metric(metric_key)
    _add_bar_value_labels(ax, x, means, stds, decimals=dec)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Stage3 aggregated results for thesis Ch.5")
    parser.add_argument("--csv", default="experiment_logs/stage3/stage3_aggregated_5runs_selected.csv",
                        help="Aggregated CSV path")
    parser.add_argument("--out-dir", default="experiment_logs/stage3/plots",
                        help="Output directory for figures")
    parser.add_argument("--fmt", choices=["pdf", "png", "both"], default="png",
                        help="Output format: png (default, for Word/论文), pdf (for LaTeX), both")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = ["png", "pdf"] if args.fmt == "both" else [args.fmt]

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    rows = load_aggregated(csv_path)
    if not rows:
        print("Error: no rows loaded")
        return 1

    print(f"Generating plots ({', '.join(formats)}) ...")
    if "png" in formats:
        plt.rcParams["savefig.dpi"] = 150  # PNG 分辨率，便于论文插入与打印

    def _out(name, ext):
        return out_dir / f"{name}.{ext}"

    for fmt in formats:
        ext = fmt
        # 1) 全部方法：迁移次数（11 根柱，略宽以免数值重叠）
        bar_plot(rows, "nummigrations", "迁移次数",
                 out_path=_out("Bar-Number_of_Task_migrations_all", ext), rot=35, figsize=(8.5, 3.5))

        # 2) 全部方法：总能耗；除以 1e6 便于读图
        bar_plot(rows, "energytotalinterval", "总能耗 (×1e6)",
                 out_path=_out("Bar-Total_Energy_all", ext), rot=35, scale=1e-6, figsize=(8.5, 3.5))

        # 3) 全部方法：SLA 违反率（全体创建）
        bar_plot(rows, "sla_violation_pct_over_created", "SLA 违反率（占创建容器%）",
                 out_path=_out("Bar-SLA_violation_pct_over_created_all", ext), rot=35, figsize=(8.5, 3.5))

        # 4) Group 1: FPE-GAN vs 传统
        bar_plot_group1(rows, "nummigrations", "迁移次数",
                       out_path=_out("group1_Bar-Number_of_Task_migrations", ext))
        bar_plot_group1(rows, "energytotalinterval", "总能耗 (×1e6)",
                       out_path=_out("group1_Bar-Total_Energy", ext), scale=1e-6)

        # 5) Group 2: TF-GAN vs FPE-GAN
        bar_plot_group2(rows, "nummigrations", "迁移次数",
                       out_path=_out("group2_Bar-Number_of_Task_migrations", ext))
        bar_plot_group2(rows, "energytotalinterval", "总能耗 (×1e6)",
                       out_path=_out("group2_Bar-Total_Energy", ext), scale=1e-6)

        # 6) Group 3: MAMO-GAN vs others
        bar_plot_group3(rows, "nummigrations", "迁移次数",
                       out_path=_out("group3_Bar-Number_of_Task_migrations", ext))
        bar_plot_group3(rows, "energytotalinterval", "总能耗 (×1e6)",
                       out_path=_out("group3_Bar-Total_Energy", ext), scale=1e-6)
        bar_plot_group3(rows, "sla_violation_pct_over_created", "SLA 违反率（占创建容器%）",
                       out_path=_out("group3_Bar-SLA_violation_pct", ext))

        # 7) 消融：迁移次数与能耗
        bar_plot_ablation(rows, "nummigrations", "迁移次数",
                         title="消融：迁移次数", out_path=_out("ablation_Bar-Number_of_Task_migrations", ext))
        bar_plot_ablation(rows, "energytotalinterval", "总能耗 (×1e6)",
                         title="消融：总能耗", out_path=_out("ablation_Bar-Total_Energy", ext), scale=1e-6)

    print(f"Done. Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
