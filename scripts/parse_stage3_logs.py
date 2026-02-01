#!/usr/bin/env python3
"""
从 Stage3 日志中解析：
1. 末尾 Summation 指标（numdestroyed, nummigrations, energytotalinterval, slaviolations 等）
2. 每步的 P, R, F1 行，汇总为整次运行的宏平均 P/R/F1（及“有正例步”上的宏平均）

用法：
  python3 scripts/parse_stage3_logs.py experiment_logs/stage3
  python3 scripts/parse_stage3_logs.py experiment_logs/stage3 --steps 600 --containers-per-step 10   # 无日志时用配置推算 total_created 与 sla_violation_pct_over_created
  python3 scripts/parse_stage3_logs.py experiment_logs/stage3 --glob "stage3_PreGAN_run*.log"
"""

import re
import sys
import argparse
from pathlib import Path

# 与 main.py 一致：Stage3 仿真 600 步，每步约 10 个新容器（BWGD2 gauss(10,1.5)）
DEFAULT_NUM_SIM_STEPS = 600
DEFAULT_NEW_CONTAINERS = 10


def method_and_run_from_log(name):
    """从日志名 stage3_<Method>_run<NN>_<timestamp> 提取 method 与 run_id。"""
    m = re.match(r"stage3_(.+)_run(\d+)_", name)
    if m:
        return m.group(1), int(m.group(2))
    return "unknown", 0


def parse_summation(log_path):
    """从日志末尾解析 Summation、Average energy、Total containers created、Overall SLA violation rate 行，返回 dict。"""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = text.strip().split("\n")
    out = {}
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Average energy"):
            m = re.search(r"Average energy \(.*\)\s*=\s*([\d.]+)", line)
            if m:
                out["avg_energy_per_destroyed"] = float(m.group(1))
            continue
        m = re.match(r"Summation\s+(\w+)\s*=\s*([\d.]+)", line)
        if m:
            key, val = m.group(1), m.group(2)
            try:
                out[key] = int(val) if "." not in val or val.endswith(".0") else float(val)
            except ValueError:
                out[key] = float(val)
        else:
            # 遇到非 Summation 行就停（避免误匹配前面的）
            if out and not line.startswith("Summation"):
                break
    # 解析 Stats 新增行：总创建容器数、两种整体 SLO 违反率（若存在）
    m_created = re.search(r"Total containers created \(sum newcontainers\)\s*=\s*(\d+)", text)
    if m_created:
        out["total_containers_created"] = int(m_created.group(1))
    m_over_created = re.search(r"Overall SLA violation rate \(slaviolations/total_created[^)]*\)\s*=\s*([\d.]+)\s*%", text)
    if m_over_created:
        out["sla_violation_pct_over_created"] = float(m_over_created.group(1))
    m_over_destroyed = re.search(r"Overall SLA violation rate \(slaviolations/numdestroyed[^)]*\)\s*=\s*([\d.]+)\s*%", text)
    if m_over_destroyed:
        out["sla_violation_pct_completed"] = float(m_over_destroyed.group(1))
    return out


def parse_prf1(log_path):
    """解析所有 'P = x, R = y, F1 = z' 行，返回 (P_list, R_list, F1_list)。"""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    # 兼容可能的 ANSI 或空格
    pattern = re.compile(r"P\s*=\s*([\d.]+)\s*,\s*R\s*=\s*([\d.]+)\s*,\s*F1\s*=\s*([\d.]+)")
    ps, rs, f1s = [], [], []
    for m in pattern.finditer(text):
        ps.append(float(m.group(1)))
        rs.append(float(m.group(2)))
        f1s.append(float(m.group(3)))
    return ps, rs, f1s


def main():
    parser = argparse.ArgumentParser(description="解析 Stage3 日志并可选按配置推算 SLA 百分比")
    parser.add_argument("log_dir", nargs="?", default="experiment_logs/stage3", help="日志目录")
    parser.add_argument("--glob", type=str, default="stage3_*.log", help="日志文件名 glob")
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_SIM_STEPS, help="仿真步数（用于推算 total_containers_created）")
    parser.add_argument("--containers-per-step", type=int, default=DEFAULT_NEW_CONTAINERS, help="每步新容器数（用于推算 total_containers_created）")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    glob_pat = args.glob

    log_files = sorted(log_dir.glob(glob_pat))
    if not log_files:
        print(f"未找到匹配的日志: {log_dir / glob_pat}")
        return 1

    total_created_from_config = args.steps * args.containers_per_step
    rows = []
    for log_path in log_files:
        name = log_path.stem
        summation = parse_summation(log_path)
        # 无日志中的 total_containers_created 时，用配置推算（不重跑实验即可得到「在全体创建中的违反率」）
        if summation.get("total_containers_created") is None:
            summation["total_containers_created"] = total_created_from_config
        if summation.get("sla_violation_pct_over_created") is None and summation.get("slaviolations") is not None and summation.get("total_containers_created"):
            summation["sla_violation_pct_over_created"] = summation["slaviolations"] / summation["total_containers_created"] * 100
        ps, rs, f1s = parse_prf1(log_path)
        n_steps = len(ps)
        if n_steps == 0:
            macro_p = macro_r = macro_f1 = None
            macro_p_pos = macro_r_pos = macro_f1_pos = None
            n_pos_steps = 0
        else:
            macro_p = sum(ps) / n_steps
            macro_r = sum(rs) / n_steps
            macro_f1 = sum(f1s) / n_steps
            pos_mask = [(p + r + f) > 0 for p, r, f in zip(ps, rs, f1s)]
            n_pos_steps = sum(pos_mask)
            if n_pos_steps > 0:
                macro_p_pos = sum(p for p, m in zip(ps, pos_mask) if m) / n_pos_steps
                macro_r_pos = sum(r for r, m in zip(rs, pos_mask) if m) / n_pos_steps
                macro_f1_pos = sum(f for f, m in zip(f1s, pos_mask) if m) / n_pos_steps
            else:
                macro_p_pos = macro_r_pos = macro_f1_pos = None
        # 总体 SLO 违反率：优先用日志中打印的 Overall (slaviolations/numdestroyed)，否则自算
        sla_pct = summation.get("sla_violation_pct_completed")
        if sla_pct is None and summation.get("numdestroyed"):
            sla_pct = summation.get("slaviolations", 0) / summation.get("numdestroyed", 1) * 100
        method, run_id = method_and_run_from_log(name)
        row = {
            "method": method,
            "run_id": run_id,
            "log": name,
            "numdestroyed": summation.get("numdestroyed"),
            "nummigrations": summation.get("nummigrations"),
            "energytotalinterval": summation.get("energytotalinterval"),
            "slaviolations": summation.get("slaviolations"),
            "sla_violation_pct": round(sla_pct, 2) if sla_pct is not None else None,
            "total_containers_created": summation.get("total_containers_created"),
            "sla_violation_pct_over_created": round(summation.get("sla_violation_pct_over_created"), 2) if summation.get("sla_violation_pct_over_created") is not None else None,
            "avg_energy_per_destroyed": summation.get("avg_energy_per_destroyed"),
            "n_steps": n_steps,
            "macro_P": round(macro_p, 4) if macro_p is not None else None,
            "macro_R": round(macro_r, 4) if macro_r is not None else None,
            "macro_F1": round(macro_f1, 4) if macro_f1 is not None else None,
            "n_steps_with_detection": n_pos_steps,
            "macro_P_pos": round(macro_p_pos, 4) if macro_p_pos is not None else None,
            "macro_R_pos": round(macro_r_pos, 4) if macro_r_pos is not None else None,
            "macro_F1_pos": round(macro_f1_pos, 4) if macro_f1_pos is not None else None,
        }
        rows.append(row)

    # 打印表格
    print("=" * 100)
    print("Stage3 运行汇总（从日志解析）")
    print("=" * 100)
    for r in rows:
        print(f"\n【{r['log']}】")
        print(f"  系统指标: numdestroyed={r['numdestroyed']}, nummigrations={r['nummigrations']}, "
              f"energytotalinterval={r['energytotalinterval']}, slaviolations={r['slaviolations']}, "
              f"SLO违反率={r['sla_violation_pct']}%, avg_energy_per_destroyed={r['avg_energy_per_destroyed']}")
        print(f"  故障检测(P/R/F1): 共 {r['n_steps']} 步; 宏平均 P={r['macro_P']}, R(召回率)={r['macro_R']}, F1={r['macro_F1']}")
        if r["n_steps_with_detection"] and r["macro_P_pos"] is not None:
            print(f"  仅在有检测的步上: {r['n_steps_with_detection']} 步; 宏平均 P={r['macro_P_pos']}, R={r['macro_R_pos']}, F1={r['macro_F1_pos']}")

    # CSV 输出到同目录（含 method/run_id 的原始数据，便于制表与复现）
    headers = ["method", "run_id", "log", "numdestroyed", "nummigrations", "energytotalinterval", "slaviolations",
              "sla_violation_pct", "total_containers_created", "sla_violation_pct_over_created", "avg_energy_per_destroyed",
              "n_steps", "macro_P", "macro_R", "macro_F1", "n_steps_with_detection", "macro_P_pos", "macro_R_pos", "macro_F1_pos"]
    csv_path = log_dir / "stage3_summary_parsed.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print(f"\n已写入: {csv_path}")

    # 原始数据 CSV（每次运行一行，便于论文制表）
    raw_path = log_dir / "stage3_raw_per_run.csv"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print(f"已写入: {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
