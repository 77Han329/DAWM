#!/usr/bin/env bash

# ===== 环境准备 =====
source ~/.bashrc
conda activate dawm

# 添加 PYTHONPATH 以确保模块能找到
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0C/di97zuq/project/DAWM/code

cd /dss/dsshome1/0C/di97zuq/project/DAWM/code/analysis || { echo "❌ cd failed"; exit 1; }

# ===== 参数接收 =====
ENV=$1
HORIZON=$2
RETURNSCALE=$3
CONFIG_FILE="temp_config_${ENV}_${HORIZON}_${RETURNSCALE}.jsonl"

# ===== 检查 GPU 是否可用 =====
echo "🔍 正在检查 GPU 可用性..."
python - <<EOF
import torch
if torch.cuda.is_available():
    print(f"✅ GPU 可用：{torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 警告：当前 PyTorch 无法使用 GPU，任务将使用 CPU 执行。请检查 CUDA 和 PyTorch 是否匹配。")
EOF

# ===== 调试输出 =====
echo "📂 当前目录: $(pwd)"
echo "🔍 查找 default_inv_train.jsonl..."
ls -l default_inv_train.jsonl || { echo "❌ 找不到 default_inv_train.jsonl"; exit 1; }

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