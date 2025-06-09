#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --output=slurmlogs/train/%x_%j.out 
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=START,END,FAIL,REQUEUE          # Notifications for job done & fail
#SBATCH --mail-user=Li.Yusong@campus.lmu.de,Xiao.Han@campus.lmu.de  # Send-to email address (replace with your email)
# ===== 环境准备 =====

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
source ~/.bashrc
conda activate newdiff
cd /home/atuin/b241dd/b241dd12/workarea/dwm/code/analysis

# ===== 参数接收 =====
ENV=$1
HORIZON=$2
RETURNSCALE=$3

# 创建唯一配置文件名
CONFIG_FILE="temp_config_${ENV}_${HORIZON}_${RETURNSCALE}.jsonl"

# ===== 生成 JSONL 文件 =====
python <<EOF
import json

env_new = "$ENV"
horizon_new = int($HORIZON)
return_scale_new = float($RETURNSCALE)

with open("default_inv_train.jsonl", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    config = json.loads(line)
    env_old = config.get("env", "")
    horizon_old = config.get("horizon", "")

    config["env"] = env_new
    config["dataset"] = env_new
    config["horizon"] = horizon_new
    config["returns_scale"] = return_scale_new

    if "RUN.prefix" in config:
        s = config["RUN.prefix"]
        s = s.replace(f"horizon_{horizon_old}", f"horizon_{horizon_new}")
        s = s.replace(env_old, env_new)
        config["RUN.prefix"] = s

    if "RUN.job_name" in config:
        s = config["RUN.job_name"]
        s = s.replace(f"horizon_{horizon_old}", f"horizon_{horizon_new}")
        s = s.replace(env_old, env_new)
        config["RUN.job_name"] = s

    if "bucket" in config:
        parts = config["bucket"].rstrip("/").split("/")
        parts[-1] = str(horizon_new)
        config["bucket"] = "/".join(parts)

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