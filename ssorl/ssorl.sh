#!/usr/bin/env bash
#SBATCH --job-name=SSORL_batch
#SBATCH --output=logs/%x_%j.out  # %x 表示job名，%j 表示job ID
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --partition=major   

#SBATCH --mail-type=BEGIN,END,FAIL         # 📩 发送通知的时机（开始 & 结束）
#SBATCH --mail-user=hxhx6584989@gmail.com    # 📬 你接收通知的邮箱地址

# ==== 参数设置 ====
MODEL=td3bc # td3bc, cql
ENV_NAME="hopper-medium-expert-v2" #9
EXPERIMENT="ssorl" # dwm, ssorl
TRAJ_LEN=8 # only 8
SEED=500 # 10,80,90,100,400，20，30，40，50，60，70，80，200，300，500
RL=True # 

# ===== 环境准备 =====
source ~/.bashrc
conda activate dwm

# ===== 进入代码目录 =====
cd /home/stud/xhan/projects/ba/ssorl

# ===== 启动任务 =====
python main.py model=$MODEL experiment=$EXPERIMENT env=$ENV_NAME traj_len=$TRAJ_LEN seed=$SEED RL=$RL