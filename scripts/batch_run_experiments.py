#!/usr/bin/env python3
"""
Batch runner to replace recovery in main.py, run multiple experiments and collect .pk into all_datasets/simulator/<Model>/
Usage:
  python3 scripts/batch_run_experiments.py --models PreGANPlus,PreGAN,PCFT --steps 50

Notes:
- This script edits main.py temporarily and restores it at the end.
- Each experiment runs `python3 main.py` in the repo root; runs may take long.
- After each run the newest .pk under logs/ is copied into all_datasets/simulator/<Model>/
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / 'main.py'
BACKUP = ROOT / 'main.py.bak'
ALL_DATASETS = ROOT / 'all_datasets' / 'simulator'
LOGS_DIR = ROOT / 'logs'

RECOVERY_MAP = {
    'PreGANPlus': 'PreGANPlusRecovery',
    'PreGAN': 'PreGANRecovery',
    'PCFT': 'PCFTRecovery',
    'DFTM': 'DFTMRecovery',
    'ECLB': 'ECLBRecovery',
    'CMODLB': 'CMODLBRecovery'
}


def replace_in_file(path: Path, pattern: str, repl: str) -> bool:
    """Replace pattern preserving leading indentation if present."""
    text = path.read_text()
    # preserve leading whitespace by capturing group and reusing it
    new_text, n = re.subn(pattern, repl, text, flags=re.M)
    if n == 0:
        return False
    path.write_text(new_text)
    return True


def find_newest_pk(logs_root: Path):
    pk_files = list(logs_root.rglob('*.pk'))
    if not pk_files:
        return None
    pk_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pk_files[0]


def run_experiment(model_name: str, steps: int, mode_arg: str):
    print(f'=== Running experiment for {model_name} (steps={steps}) ===')
    # Replace recovery line
    cls = RECOVERY_MAP.get(model_name)
    if cls is None:
        raise ValueError(f'Unknown model {model_name}. Supported: {list(RECOVERY_MAP.keys())}')
    recovery_line = f"recovery = {cls}(HOSTS, environment, training = True)"
    # preserve leading indentation when replacing the recovery line
    replaced = replace_in_file(MAIN_PY, r"^(\s*)recovery\s*=.*$", r"\1" + recovery_line)
    if not replaced:
        raise RuntimeError('Failed to replace recovery line in main.py')
    # Replace NUM_SIM_STEPS if provided
    if steps is not None:
        # preserve leading indentation for NUM_SIM_STEPS if present
        replace_in_file(MAIN_PY, r"^(\s*)NUM_SIM_STEPS\s*=.*$", r"\1" + f'NUM_SIM_STEPS = {steps}')

    # Run main.py
    start = time.time()
    proc = subprocess.run([sys.executable, str(MAIN_PY)], cwd=str(ROOT))
    elapsed = time.time() - start
    print(f'Process for {model_name} exited with {proc.returncode} (elapsed {elapsed:.1f}s)')
    # if process failed, don't copy .pk
    if proc.returncode != 0:
        print(f'Not copying .pk because process failed with return code {proc.returncode}')
        return False
    # find newest pk
    newest = find_newest_pk(LOGS_DIR)
    if newest is None:
        print('Warning: no .pk file found in logs/')
        return False
    dest_dir = ALL_DATASETS / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / newest.name
    shutil.copy2(newest, dest)
    print(f'Copied {newest} -> {dest}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=','.join(RECOVERY_MAP.keys()), help='Comma-separated model names to run')
    parser.add_argument('--steps', type=int, default=100, help='NUM_SIM_STEPS to set for all runs')
    parser.add_argument('--mode-arg', default='-m 2', help='extra args to pass to main (unused currently)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force-grapher', action='store_true', help='Run grapher even if some models failed (may error)')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]

    successes = []

    if not MAIN_PY.exists():
        print('main.py not found in repo root:', MAIN_PY)
        sys.exit(1)

    print('Backing up main.py ->', BACKUP)
    shutil.copy2(MAIN_PY, BACKUP)

    try:
        for m in models:
            if args.dry_run:
                print('[DRY RUN] Would run model', m)
                continue
            success = run_experiment(m, args.steps, args.mode_arg)
            if not success:
                print('Run failed for', m)
                # continue to next
            else:
                # record successful model copy
                successes.append(m)
        # restore main.py from backup
    finally:
        print('Restoring original main.py from backup')
        if BACKUP.exists():
            shutil.copy2(BACKUP, MAIN_PY)
            BACKUP.unlink()

    # If dry-run, skip grapher
    if args.dry_run:
        print('Dry-run: skipping grapher.')
        return

    # Check whether we have results for all requested models
    missing = [m for m in models if m not in successes]
    if len(missing) > 0 and not args.force_grapher:
        print('Missing results for models:', missing)
        print('Skipping grapher. Rerun script after all models complete, or set --force-grapher to run anyway.')
        return

    # Run grapher
    print('Running grapher to aggregate results...')
    subprocess.run([sys.executable, str(ROOT / 'grapher.py'), 'simulator'], cwd=str(ROOT))
    print('Done. Results in results/simulator/')


if __name__ == '__main__':
    main()
