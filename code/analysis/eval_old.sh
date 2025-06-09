#!/usr/bin/env bash
#SBATCH --job-name=dwm
#SBATCH --output=logs/sample/horizon8_t/slurm_%A_hopper-medium-replay-v2.out 
#SBATCH --gres=gpu:a100:1           # 每个节点分配 1 个 A100 GPU
#SBATCH --cpus-per-task=4                        # ✅ 分配 32 GB 内存
#SBATCH --nodes=1                           # 单节点
#SBATCH --ntasks=1                   # 单任务
#SBATCH --time=1:00:00              # 最大运行时间(HH:MM:SS)

# ===== 环境准备 =====
source ~/.bashrc
conda activate newdiff

# ===== 进入代码目录 =====
cd /home/atuin/b241dd/b241dd12/workarea/dwm/code/analysis

# ===== 启动任务 =====
python eval.py