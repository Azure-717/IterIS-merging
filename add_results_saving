#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Add comprehensive result saving functionality to IterIS_plus.py"""

import re

# 读取原文件
with open('IterIS_plus.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 确保导入了 os 模块
if 'import os' not in content:
    content = content.replace('import gc', 'import os\nimport gc')

# 2. 在 main 函数开始处添加辅助函数
# 找到 def main(): 后的第一个 parser = 之前的位置
helper_functions = '''
def generate_output_filename(task_type, use_mats, use_camr, use_dcs):
    """Generate unique filename based on task type and activation flags."""
    output_dir = f'outputs_{task_type.lower()}'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base filename
    mats_str = 'MATS1' if use_mats else 'MATS0'
    camr_str = 'CAMR1' if use_camr else 'CAMR0'
    dcs_str = 'DCS1' if use_dcs else 'DCS0'
    base_name = f'{mats_str}_{camr_str}_{dcs_str}'
    
    # Check for existing files and add number if needed
    counter = 0
    while True:
        if counter == 0:
            filename = f'{output_dir}/{base_name}.txt'
        else:
            filename = f'{output_dir}/{base_name}_{counter}.txt'
        
        if not os.path.exists(filename):
            return filename
        counter += 1


def format_results_table(task_type, task_targets, eval_results, config_data, 
                        use_mats, use_camr, use_dcs, elapsed_time):
    """Format evaluation results into a nice table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"IterIS++ EXPERIMENT RESULTS - {task_type}")
    lines.append("=" * 80)
    lines.append("")
    
    # Configuration section
    lines.append("CONFIGURATION:")
    lines.append("-" * 80)
    lines.append(f"Task Type:              {task_type}")
    lines.append(f"Model:                  {config_data[task_type]['model_name']}")
    lines.append(f"Tasks:                  {', '.join(task_targets)}")
    lines.append(f"Max Iterations:         {config_data[task_type]['max_iter']}")
    lines.append(f"LoRA Rank:              {config_data[task_type]['rank']}")
    lines.append(f"LoRA Alpha:             {config_data[task_type]['lora_alpha']}")
    lines.append(f"Alpha 1 (reg):          {config_data[task_type]['alpha_1']}")
    lines.append(f"Alpha 2 (reg):          {config_data[task_type]['alpha_2']}")
    lines.append(f"Sample Numbers:         {config_data[task_type]['samples_num']}")
    lines.append("")
    lines.append("IterIS++ Innovations:")
    lines.append(f"  - MATS (Anderson Acceleration):    {'✓ ENABLED' if use_mats else '✗ DISABLED'}")
    lines.append(f"  - CAMR (Curvature-Aware Reg):      {'✓ ENABLED' if use_camr else '✗ DISABLED'}")
    lines.append(f"  - DCS (Dynamic Sample Weighting):  {'✓ ENABLED' if use_dcs else '✗ DISABLED'}")
    if use_mats:
        lines.append(f"    - MATS history size:             {config_data[task_type].get('mats_history_size', 5)}")
        lines.append(f"    - MATS regularization:           {config_data[task_type].get('mats_regularization', 1e-6)}")
    if use_camr:
        lines.append(f"    - CAMR alpha:                    {config_data[task_type].get('camr_alpha', config_data[task_type]['alpha_1'])}")
        lines.append(f"    - CAMR beta:                     {config_data[task_type].get('camr_beta', 1e-8)}")
    if use_dcs:
        lines.append(f"    - DCS sigma:                     {config_data[task_type].get('dcs_sigma', 1.0)}")
    lines.append(f"  - Convergence threshold:           {config_data[task_type].get('convergence_threshold', 1e-6)}")
    lines.append("")
    lines.append(f"Total Training Time:    {elapsed_time:.2f} seconds")
    lines.append("")
    
    # Results section
    lines.append("EVALUATION RESULTS:")
    lines.append("=" * 80)
    
    if task_type == 'TASKS_blip_base':
        # Vision & Language task format
        lines.append("Method                Acc(pos, neg)     CIDEr    B-1    B-2    B-3    B-4")
        lines.append("-" * 80)
        
        # Parse results for each task
        for task in task_targets:
            if task in eval_results:
                result = eval_results[task]
                lines.append(f"{task:20}  {result}")
        
        lines.append("-" * 80)
        lines.append("IterIS++ (This Run)   [Results from evaluation above]")
        
    elif task_type == 'GLUE_t5':
        # GLUE task format
        lines.append("Task      Metric    Score")
        lines.append("-" * 80)
        
        for task in task_targets:
            if task in eval_results:
                result = eval_results[task]
                lines.append(f"{task:10} {result}")
        
        lines.append("-" * 80)
        
    elif task_type == 'EMOTION_t5_large':
        # Emotion task format
        lines.append("Task             Metric    Score")
        lines.append("-" * 80)
        
        for task in task_targets:
            if task in eval_results:
                result = eval_results[task]
                lines.append(f"{task:15}  {result}")
        
        lines.append("-" * 80)
    
    lines.append("=" * 80)
    lines.append("")
    lines.append("* Best performance indicated where applicable")
    lines.append("")
    
    return '\\n'.join(lines)

'''

# 在 def main(): 之后插入辅助函数
main_pattern = r'(def main\(\):.*?"""Main entry point for IterIS\+\+\.""")'
replacement = r'\1\n' + helper_functions
content = re.sub(main_pattern, replacement, content, flags=re.DOTALL)

# 3. 在 main 函数中，Model evaluation 之前添加结果收集初始化
# 找到 "# Model evaluation" 前面添加结果收集
eval_pattern = r'(\s+)(# Model evaluation\s+for task_name in task_targets:)'
eval_replacement = r'\1# Generate output filename\n\1output_filename = generate_output_filename(task_type, use_mats, use_camr, use_dcs)\n\1\n\1# Collect evaluation results\n\1eval_results = {}\n\1\n\1\2'
content = re.sub(eval_pattern, eval_replacement, content)

# 4. 在最后的 gc.collect() 之后添加结果保存代码
save_code = '''
    
    # Save results to file
    result_text = format_results_table(
        task_type=task_type,
        task_targets=task_targets,
        eval_results=eval_results,
        config_data=config_data,
        use_mats=use_mats,
        use_camr=use_camr,
        use_dcs=use_dcs,
        elapsed_time=elapsed_time
    )
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(result_text)
    
    print(f"\\n{'=' * 80}")
    print(f"Results saved to: {output_filename}")
    print(f"{'=' * 80}")
'''

# 在最后一个 gc.collect() 之后插入
last_gc_pattern = r'(gc\.collect\(\)\s*\n)(\s*if __name__)'
content = re.sub(last_gc_pattern, r'\1' + save_code + r'\n\2', content)

# 写回文件
with open('IterIS_plus.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Successfully added result saving functionality!")
print("✓ Added helper functions for filename generation and result formatting")
print("✓ Results will be saved to outputs_<task_type>/ directory")
