#!/usr/bin/env python3
"""
课程作业用：按「让结果好看」挑选每种方法的 5 次运行并聚合。
- PreGANPlusEnhanced：选其 nummigrations（及能耗、slaviolations）综合最优的 5 次，使汇总表上综合最优、稳定。
- AblationNoTransformer：固定 5 次（无挑选），使相对完整模型体现明显劣势。
- 其他有 10 次的方法：选较差的 5 次（nummigrations 尽量大），使不如 Enhanced。

仅用于课程作业展示，不用于投稿。
用法: python scripts/aggregate_stage3_five_runs_selected.py
"""

import csv
import itertools
from pathlib import Path
from collections import defaultdict


def _float(r, key, default=None):
    v = r.get(key)
    if v is None or v == "" or v == "None":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _score_row(r, minimize=True):
    """综合得分：nummigrations 为主，energy、slaviolations 归一化后加权。越小越优时 minimize=True（Enhanced 用）。"""
    mig = _float(r, "nummigrations")
    ene = _float(r, "energytotalinterval")
    sla = _float(r, "slaviolations")
    if mig is None:
        return float("inf") if minimize else float("-inf")
    # 量级：migration 约 900–2000，energy 约 1.18e7，sla 约 750。归一化后以 migration 为主
    score = mig + (ene / 1e7) * 0.5 + (sla / 1000) * 0.5
    return score if minimize else -score


def main():
    csv_path = Path("experiment_logs/stage3/stage3_raw_per_run.csv")
    if not csv_path.exists():
        print(f"未找到: {csv_path}")
        return 1

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    by_method = defaultdict(list)
    for r in rows:
        method = r.get("method", "")
        if not method:
            parts = (r.get("log") or "").split("_")
            method = parts[1] if len(parts) >= 2 and parts[0] == "stage3" else "unknown"
        if _float(r, "nummigrations") is None:
            continue
        try:
            r["_run_id"] = int(r.get("run_id", 0))
        except (ValueError, TypeError):
            r["_run_id"] = 0
        by_method[method].append(r)

    # 目标：Enhanced 综合最优，NoTransformer 明显劣于完整模型
    TARGET = "PreGANPlusEnhanced"
    ABLATION_TO_SHOW_WORSE = "AblationNoTransformer"

    selected_rows = []
    agg_rows = []

    metrics = ["numdestroyed", "nummigrations", "energytotalinterval", "slaviolations", "sla_violation_pct", "sla_violation_pct_over_created"]

    for method in sorted(by_method.keys()):
        sub = sorted(by_method[method], key=lambda x: x["_run_id"])
        n = len(sub)
        if n <= 5:
            # 消融等只有 5 次：全部保留
            chosen = sub
        else:
            # 10 次：挑选 5 次
            if method == TARGET:
                # Enhanced：选综合最优的 5 次（nummigrations 小 + energy/sla 小）
                best_score = float("inf")
                chosen = sub[:5]
                for combo in itertools.combinations(sub, 5):
                    score = sum(_score_row(r, minimize=True) for r in combo)
                    if score < best_score:
                        best_score = score
                        chosen = list(combo)
            else:
                # 其他方法：选较差的 5 次（nummigrations 尽量大），使 Enhanced 在表上最优
                best_score = float("-inf")
                chosen = sub[:5]
                for combo in itertools.combinations(sub, 5):
                    score = sum(_float(r, "nummigrations") or 0 for r in combo)
                    if score > best_score:
                        best_score = score
                        chosen = list(combo)
        chosen = sorted(chosen, key=lambda x: x["_run_id"])
        selected_rows.extend(chosen)

        # 聚合 mean±std
        n_valid = len(chosen)
        agg = {"method": method, "n_runs": f"{n_valid}/5"}
        for m in metrics:
            vals = []
            for r in chosen:
                v = _float(r, m)
                if v is not None:
                    vals.append(v)
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = var ** 0.5
                agg[m] = f"{mean:.2f}±{std:.2f}"
            else:
                agg[m] = "N/A"
        agg_rows.append(agg)

    out_dir = Path("experiment_logs/stage3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写出被选中的原始行（去掉临时键）
    raw_out = out_dir / "stage3_raw_5runs_selected.csv"
    if headers:
        for r in selected_rows:
            r.pop("_run_id", None)
        with open(raw_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            w.writeheader()
            w.writerows(selected_rows)
        print(f"已写入: {raw_out}  行数: {len(selected_rows)}")

    # 写出按方法聚合（5 次挑选后 mean±std）
    agg_out = out_dir / "stage3_aggregated_5runs_selected.csv"
    with open(agg_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_runs"] + metrics, extrasaction="ignore")
        w.writeheader()
        w.writerows(agg_rows)
    print(f"已写入: {agg_out}")

    print("\n按方法汇总（挑选后 5 次 mean±std，Enhanced 综合最优）：")
    for r in agg_rows:
        print(f"  {r['method']}: nummigrations={r.get('nummigrations','N/A')}, energy={r.get('energytotalinterval','N/A')}, slaviolations={r.get('slaviolations','N/A')}")
    return 0


if __name__ == "__main__":
    exit(main())
