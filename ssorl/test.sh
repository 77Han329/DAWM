#!/usr/bin/env bash
#SBATCH --job-name=SSORL_batch
#SBATCH --output=logs/slurm_%A_ssorl.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

# ===== 环境准备 =====
source ~/.bashrc
conda activate dwm

python
import mujoco_py
# TASK_ID=$SLURM_PROCID

# MODELS=("td3bc" "td3bc" "cql" "cql")
# ENVS=("hopper-medium-v2" "walker2d-medium-v2" "hopper-medium-v2" "walker2d-medium-v2")

# MODEL=${MODELS[$TASK_ID]}
# ENV=${ENVS[$TASK_ID]}
# EXP="ssorl"

# export CUDA_VISIBLE_DEVICES=$TASK_ID

# echo ">>> Running $MODEL on $ENV with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# python main.py model=$MODEL env=$ENV experiment=$EXP
