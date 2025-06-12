#!/usr/bin/env bash
#SBATCH --job-name=train_dawm
#SBATCH --output=slurmlogs/train/%x_%j.out 
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00  # 2 天
# ===== 环境准备 =====
source ~/.bashrc
conda activate dawm

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0C/di97zuq/project/DAWM/code

cd /dss/dsshome1/0C/di97zuq/project/DAWM/code/analysis || { echo "❌ cd failed"; exit 1; }

# ===== 参数接收 =====
ENV=$1
HORIZON=$2
RETURNSCALE=$3
BATCH_SIZE=$4
LEARNING_RATE=$5
N_TRAIN_STEPS=$6
CONFIG_FILE="new_temp_config_${ENV}_${HORIZON}_${RETURNSCALE}.jsonl"

# 固定参数

# ===== 检查 GPU 状态 =====
echo "🔍 Checking GPU availability..."
python - <<EOF
import torch
if torch.cuda.is_available():
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("❌ torch.cuda.is_available() == False — CUDA device not available")
EOF

# ===== 调试输出 =====
echo "📂 当前目录: $(pwd)"
echo "🔍 查找 new_train.jsonl..."
ls -l new_train.jsonl || { echo "❌ 找不到 new_train.jsonl"; exit 1; }

# ===== 生成 JSONL 文件 =====
python <<EOF
import json, os

env_new = "$ENV"
horizon_new = int($HORIZON)
return_scale_new = float($RETURNSCALE)
batch_size_new = $BATCH_SIZE
learning_rate_new = $LEARNING_RATE
n_train_steps_new = $N_TRAIN_STEPS

with open("new_train.jsonl", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    config = json.loads(line)
    env_old = config.get("env", "")
    horizon_old = config.get("horizon", "")
    seed = config.get("seed", 100)

    # 更新训练参数
    config["env"] = env_new
    config["dataset"] = env_new
    config["horizon"] = horizon_new
    config["returns_scale"] = return_scale_new
    config["batch_size"] = batch_size_new
    config["learning_rate"] = learning_rate_new
    config["n_train_steps"] = n_train_steps_new

    # 构造路径
    subdir = f"horizon_{horizon_new}/batch_{batch_size_new}/lr_{learning_rate_new}/steps_{n_train_steps_new}/{env_new}/{seed}"
    config["RUN.prefix"] = f"diffuser/default_inv/{subdir}"
    config["RUN.job_name"] = subdir
    config["bucket"] = f"/dss/dsshome1/0C/di97zuq/project/DAWM/weights/{subdir}"

    # 创建输出目录（可选）
    os.makedirs(config["bucket"], exist_ok=True)

    new_lines.append(json.dumps(config))

with open("$CONFIG_FILE", "w") as f:
    for line in new_lines:
        f.write(line + "\\n")
EOF

# ===== 检查是否写成功 =====
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Failed to generate config file: $CONFIG_FILE"
    exit 1
fi

echo "✅ Using config file: $CONFIG_FILE"

# ===== 启动训练 =====
python train.py "$CONFIG_FILE"