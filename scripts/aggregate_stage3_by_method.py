#!/usr/bin/env python3
"""
从 stage3_summary_parsed.csv 按方法聚合，计算各方法的 mean±std（nummigrations, energytotalinterval, slaviolations 等）。
用法: python scripts/aggregate_stage3_by_method.py
"""

import csv
from pathlib import Path
from collections import defaultdict


def main():
    csv_path = Path("experiment_logs/stage3/stage3_summary_parsed.csv")
    if not csv_path.exists():
        print(f"未找到: {csv_path}")
        return 1

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 从 log 名提取方法（若 CSV 无 method 列）: stage3_<Method>_runNN_...
    def method_from_log(log_name):
        parts = log_name.split("_")
        if len(parts) >= 2 and parts[0] == "stage3":
            return parts[1]
        return "unknown"

    by_method = defaultdict(list)
    for r in rows:
        method = r.get("method") or method_from_log(r["log"])
        by_method[method].append(r)

    metrics = ["numdestroyed", "nummigrations", "energytotalinterval", "slaviolations", "sla_violation_pct", "sla_violation_pct_over_created"]
    out_rows = []
    for method in sorted(by_method.keys()):
        sub = by_method[method]
        valid = [r for r in sub if r.get("nummigrations") not in (None, "", "None")]
        n_total, n_valid = len(sub), len(valid)
        if n_valid == 0:
            out_rows.append({
                "method": method,
                "n_runs": f"{n_valid}/{n_total}",
                "numdestroyed": "N/A",
                "nummigrations": "N/A",
                "energytotalinterval": "N/A",
                "slaviolations": "N/A",
                "sla_violation_pct": "N/A",
            })
            continue
        agg = {}
        for m in metrics:
            vals = []
            for r in valid:
                v = r.get(m)
                if v is None or v == "" or v == "None":
                    continue
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            if vals:
                mean = sum(vals) / len(vals)
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = variance ** 0.5
                agg[m] = f"{mean:.2f}±{std:.2f}"
            else:
                agg[m] = "N/A"
        out_rows.append({
            "method": method,
            "n_runs": f"{n_valid}/{n_total}",
            **agg,
        })

    # 打印
    print("=" * 100)
    print("Stage3 按方法聚合 (mean±std)")
    print("=" * 100)
    for r in out_rows:
        print(f"\n【{r['method']}】 有效运行: {r['n_runs']}")
        print(f"  numdestroyed = {r['numdestroyed']}")
        print(f"  nummigrations = {r['nummigrations']}")
        print(f"  energytotalinterval = {r['energytotalinterval']}")
        print(f"  slaviolations = {r['slaviolations']}")
        print(f"  sla_violation_pct(%) = {r['sla_violation_pct']}")
        print(f"  sla_violation_pct_over_created(%) = {r.get('sla_violation_pct_over_created', 'N/A')}")

    # 写入 CSV
    out_path = Path("experiment_logs/stage3/stage3_aggregated_by_method.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["method", "n_runs"] + metrics
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n已写入: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
