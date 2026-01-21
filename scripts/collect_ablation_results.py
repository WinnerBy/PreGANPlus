#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect ablation results from metrics_with_interval.csv and summarize.
"""
import argparse
from pathlib import Path
import pandas as pd


def load_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Guard against empty files
    if df.empty:
        return None
    total_energy_kwh = df["energytotalinterval"].sum() / 1000.0
    migrations = df["nummigrations"].sum()
    # avg response time: match stage plotting logic (only intervals with numdestroyed>0)
    mask = df["numdestroyed"] > 0
    avg_response_time = df.loc[mask, "avgresponsetime"].mean() if mask.any() else 0.0
    total_destroyed = df["numdestroyed"].sum()
    total_sla = df["slaviolations"].sum()
    sla_violation_rate = (total_sla / total_destroyed) * 100.0 if total_destroyed > 0 else 0.0
    return {
        "total_energy_kwh": total_energy_kwh,
        "avg_response_time": avg_response_time,
        "migrations": migrations,
        "sla_violation_rate_pct": sla_violation_rate,
        "total_destroyed": total_destroyed,
        "total_sla_violations": total_sla,
    }


def find_metrics_files(run_dir: Path):
    return list(run_dir.rglob("metrics_with_interval.csv"))


def summarize(data_dir: Path):
    rows = []
    for method_dir in sorted([d for d in data_dir.iterdir() if d.is_dir()]):
        method = method_dir.name
        for run_dir in sorted([d for d in method_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]):
            metrics_files = find_metrics_files(run_dir)
            if not metrics_files:
                continue
            # Use the first match (there should be only one)
            metrics = load_metrics(metrics_files[0])
            if metrics is None:
                continue
            row = {
                "method": method,
                "run": run_dir.name,
                "metrics_path": str(metrics_files[0]),
                **metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize ablation results")
    parser.add_argument("--data-dir", type=str, required=True, help="Ablation data directory")
    parser.add_argument("--output-csv", type=str, default="ablation_summary.csv")
    parser.add_argument("--output-md", type=str, default="ablation_summary.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = summarize(data_dir)
    if df.empty:
        print("No metrics found.")
        return

    # Per-run summary
    df.to_csv(args.output_csv, index=False)

    # Per-method mean summary
    summary = df.groupby("method")[["total_energy_kwh", "avg_response_time",
                                    "migrations", "sla_violation_rate_pct"]].mean().reset_index()
    summary = summary.sort_values("method")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("| 方法 | 总能耗(kWh) | 平均响应时间(s) | 迁移次数 | SLA违约率(%) |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: |\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['method']} | {row['total_energy_kwh']:.2f} | "
                    f"{row['avg_response_time']:.2f} | {row['migrations']:.0f} | "
                    f"{row['sla_violation_rate_pct']:.2f} |\n")

    print(f"Wrote per-run CSV: {args.output_csv}")
    print(f"Wrote per-method MD: {args.output_md}")


if __name__ == "__main__":
    main()
