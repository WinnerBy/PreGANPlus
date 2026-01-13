#!/usr/bin/env python3
"""
清理过时的文档和脚本
"""

import shutil
from pathlib import Path

def cleanup_old_files():
    """清理过时的文件"""
    
    print("=" * 100)
    print("清理过时文件")
    print("=" * 100)
    print()
    
    # 1. 清理experiment_logs中的过时分析文档
    print("📝 清理experiment_logs中的过时分析文档...")
    experiment_logs = Path('experiment_logs')
    
    # 保留的文件
    keep_files = {
        'ALL_RESULTS_AGGREGATED.json',
        'OPTIMAL_RESULTS_FINAL_OPTIMIZED.json',
        '最终选择结果报告.md',
    }
    
    # 需要删除的过时分析文档（在各个stage4目录中）
    old_analysis_files = []
    for stage_dir in experiment_logs.glob('stage4_*'):
        if stage_dir.is_dir():
            for md_file in stage_dir.glob('*.md'):
                if md_file.name not in ['README.md']:  # 保留README
                    old_analysis_files.append(md_file)
    
    for file_path in old_analysis_files:
        try:
            file_path.unlink()
            print(f"  ✅ 删除: {file_path.relative_to(experiment_logs)}")
        except Exception as e:
            print(f"  ⚠️  无法删除 {file_path}: {e}")
    
    print()
    
    # 2. 清理根目录的过时文档
    print("📄 清理根目录的过时文档...")
    root_dir = Path('.')
    
    old_root_docs = [
        'RESULT_SELECTION_GUIDE.md',
        'STAGE4_MULTIPLE_RUNS_GUIDE.md',
        'EXPERIMENT_FINAL_REPORT.md',
        '实验优化说明.md',
        '实验运行说明.md',
        'paper_experiment.md',
    ]
    
    for doc in old_root_docs:
        doc_path = root_dir / doc
        if doc_path.exists():
            try:
                doc_path.unlink()
                print(f"  ✅ 删除: {doc}")
            except Exception as e:
                print(f"  ⚠️  无法删除 {doc}: {e}")
    
    print()
    
    # 3. 清理过时的脚本（保留重要的）
    print("🔧 清理过时的脚本...")
    scripts_dir = Path('scripts')
    
    # 保留的重要脚本
    keep_scripts = {
        'aggregate_all_results.py',
        'select_optimal_from_all.py',
        'optimize_selection.py',
        'archive_final_results.py',
        'cleanup_old_files.py',
        'paper_experiment_stage1_data_collection.py',
        'paper_experiment_stage2_encoder_training.py',
        'paper_experiment_stage3_gan_training.py',
        'paper_experiment_stage4_testing.py',
        'run_paper_experiment.sh',
        'run_stage4_multiple.sh',
        'check_checkpoint_training_info.py',
        'extract_metrics_from_logs.py',
        'README_汇总和筛选.md',
    }
    
    # 删除过时的脚本
    for script_file in scripts_dir.glob('*.py'):
        if script_file.name not in keep_scripts and not script_file.name.startswith('__'):
            try:
                script_file.unlink()
                print(f"  ✅ 删除: {script_file.name}")
            except Exception as e:
                print(f"  ⚠️  无法删除 {script_file.name}: {e}")
    
    for script_file in scripts_dir.glob('*.sh'):
        if script_file.name not in keep_scripts:
            try:
                script_file.unlink()
                print(f"  ✅ 删除: {script_file.name}")
            except Exception as e:
                print(f"  ⚠️  无法删除 {script_file.name}: {e}")
    
    print()
    
    # 4. 清理experiment_data中的过时分析文档
    print("📊 清理experiment_data中的过时分析文档...")
    experiment_data = Path('experiment_data')
    if experiment_data.exists():
        for stage_dir in experiment_data.glob('stage4_*'):
            if stage_dir.is_dir():
                for md_file in stage_dir.glob('*.md'):
                    try:
                        md_file.unlink()
                        print(f"  ✅ 删除: {md_file.relative_to(experiment_data)}")
                    except Exception as e:
                        print(f"  ⚠️  无法删除 {md_file}: {e}")
    
    print()
    
    print("=" * 100)
    print("✅ 清理完成！")
    print("=" * 100)

if __name__ == "__main__":
    cleanup_old_files()
