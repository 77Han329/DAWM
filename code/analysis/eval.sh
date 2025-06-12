#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --output=slurmlogs/sample/horizon8/%x_%j.out
#SBATCH --gres=gpu:a100:1           # 每个节点分配 1 个 A100 GPU
#SBATCH --cpus-per-task=4                        # ✅ 分配 32 GB 内存
#SBATCH --nodes=1                           # 单节点
#SBATCH --ntasks=1                   # 单任务
#SBATCH --time=24:00:00              # 最大运行时间(HH:MM:SS)
# ===== 环境准备 =====



source ~/.bashrc
conda activate dwm

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# ===== 进入代码目录 =====
cd /dss/dsshome1/0C/di97zuq/project/DAWM/code/analysis
# ===== 启动任务 =====

ENV=$1         # e.g., hopper-medium-v2
HORIZON=$2     # e.g., 8
RTG=$3         # e.g., 0.4

# 生成新的 JSONL 配置文件
CONFIG_FILE=$(python <<EOF
import json

env_new = "$ENV"
horizon_new = int($HORIZON)
rtg_new = float($RTG)

with open("default_inv_sample.jsonl", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    config = json.loads(line)

    # # 替换基本字段
    # config["env"] = env
    # config["dataset"] = env
    # config["horizon"] = horizon
    # config["RTG"] = rtg
    # 记录旧值

    env_old = config.get("env", "")
    horizon_old = config.get("horizon", "")

    # 修改核心字段
    config["env"] = env_new
    config["dataset"] = env_new
    config["horizon"] = horizon_new
    config["RTG"] = rtg_new

    # 替换 RUN.prefix 中的 env 和 horizon
    if "RUN.prefix" in config:
        # config["RUN.prefix"] = config["RUN.prefix"]
        # config["RUN.prefix"] = config["RUN.prefix"].replace(config.get("env", ""), env)
        # config["RUN.prefix"] = config["RUN.prefix"].replace(f"horizon_{config.get('horizon', horizon)}", f"horizon_{horizon}")
        s = config["RUN.prefix"]
        s = s.replace(f"horizon_{horizon_old}", f"horizon_{horizon_new}")
        s = s.replace(env_old, env_new)
        config["RUN.prefix"] = s

        # print(config["RUN.prefix"])

    # 替换 RUN.job_name 中的 env 和 horizon
    if "RUN.job_name" in config:
        # config["RUN.job_name"] = config["RUN.job_name"]
        # config["RUN.job_name"] = config["RUN.job_name"].replace(config.get("env", ""), env)
        # config["RUN.job_name"] = config["RUN.job_name"].replace(f"horizon_{config.get('horizon', horizon)}", f"horizon_{horizon}")
        s = config["RUN.job_name"]
        s = s.replace(f"horizon_{horizon_old}", f"horizon_{horizon_new}")
        s = s.replace(env_old, env_new)
        config["RUN.job_name"] = s
        # print(config["RUN.job_name"])

    # 替换 bucket 最后一位为 horizon 数字
    if "bucket" in config:
        parts = config["bucket"].rstrip("/").split("/")
        parts[-1] = str(horizon_new)
        config["bucket"] = "/".join(parts)
        # print(config["bucket"])
    
    # assert 0

    new_lines.append(json.dumps(config))

config_file=f"temp_config_{env_new}_{horizon_new}_{rtg_new}.jsonl"

with open(config_file, "w") as f:
    for line in new_lines:
        f.write(line + "\n")

print(config_file)

EOF
)

echo "Using config file: $CONFIG_FILE"
python eval.py "$CONFIG_FILE"