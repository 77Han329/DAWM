import os
import pandas as pd
import numpy as np

# === 配置 ===
horizon = 8
base_root = f"exp_2e6/Horizon_{horizon}"
algorithms = ["td3bc"]
environments = [
    "hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2",
    "hopper-medium-replay-v2", "halfcheetah-medium-replay-v2", "walker2d-medium-replay-v2",
    "hopper-medium-expert-v2", "halfcheetah-medium-expert-v2", "walker2d-medium-expert-v2"
]
subfolders = ["dwm", "ssorl"]

# === D4RL baseline 分数 ===
d4rl_scores = {
    "hopper-medium-v2": {"random": -20.27, "expert": 3234.3},
    "halfcheetah-medium-v2": {"random": -280.178953, "expert": 12135.0},
    "walker2d-medium-v2": {"random": 1.629008, "expert": 4592.3},
    "hopper-medium-replay-v2": {"random": -20.27, "expert": 3234.3},
    "halfcheetah-medium-replay-v2": {"random": -280.178953, "expert": 12135.0},
    "walker2d-medium-replay-v2": {"random": 1.629008, "expert": 4592.3},
    "hopper-medium-expert-v2": {"random": -20.27, "expert": 3234.3},
    "halfcheetah-medium-expert-v2": {"random": -280.178953, "expert": 12135.0},
    "walker2d-medium-expert-v2": {"random": 1.629008, "expert": 4592.3},
}

# === 初始化结构 ===
best_scores_by_subfolder = {sub: {} for sub in subfolders}
best_seed_avg_scores = {sub: {} for sub in subfolders}

# === 主循环：遍历 ssorl / dwm 和所有环境 ===
for subfolder in subfolders:
    for env in environments:
        expert = d4rl_scores[env]["expert"]
        random = d4rl_scores[env]["random"]
        denom = expert - random

        best_top1_mean = -np.inf
        best_top1_std = None
        best_top1_seed = None

        seed_avg_scores = {}

        for seed in os.listdir(base_root):
            seed_path = os.path.join(base_root, seed)
            if not os.path.isdir(seed_path):
                continue

            for algo in algorithms:
                result_csv = os.path.join(seed_path, algo, env, subfolder, "results.csv")
                if os.path.exists(result_csv):
                    df = pd.read_csv(result_csv)
                    if "evaluation/return_mean" not in df.columns:
                        continue

                    # Top1 performance
                    top1_idx = df["evaluation/return_mean"].nlargest(1).index[0]
                    top1_return = df["evaluation/return_mean"].iloc[top1_idx]
                    top1_std = df["evaluation/return_std"].iloc[top1_idx] if "evaluation/return_std" in df.columns else 0.0

                    norm_top1 = (top1_return - random) / denom
                    norm_std = top1_std / denom

                    if norm_top1 > best_top1_mean:
                        best_top1_mean = norm_top1
                        best_top1_std = norm_std
                        best_top1_seed = seed

                    # Average performance for current seed
                    normed_returns = (df["evaluation/return_mean"] - random) / denom
                    avg_score = normed_returns.mean()
                    seed_avg_scores[seed] = avg_score

        # 最佳 top-1 表现
        best_scores_by_subfolder[subfolder][env] = {
            "score": best_top1_mean,
            "std": best_top1_std,
            "seed": best_top1_seed
        }

        # 平均表现最好的 seed
        if seed_avg_scores:
            best_seed_avg = max(seed_avg_scores.items(), key=lambda x: x[1])
            best_seed_avg_scores[subfolder][env] = {
                "avg_score": best_seed_avg[1],
                "seed": best_seed_avg[0]
            }
        else:
            best_seed_avg_scores[subfolder][env] = {
                "avg_score": None,
                "seed": None
            }

# === 汇总输出为 DataFrame ===
top1_df = []
avg_df = []

for subfolder in subfolders:
    for env in environments:
        top1 = best_scores_by_subfolder[subfolder][env]
        avg = best_seed_avg_scores[subfolder][env]

        # 计算 top1_seed 对应的全体平均表现
        top1_seed = top1["seed"]
        expert = d4rl_scores[env]["expert"]
        random = d4rl_scores[env]["random"]
        denom = expert - random
        seed_path = os.path.join(base_root, top1_seed) if top1_seed else None
        avg_score_under_top1_seed = None

        if top1_seed and os.path.isdir(seed_path):
            result_csv = os.path.join(seed_path, algorithms[0], env, subfolder, "results.csv")
            if os.path.exists(result_csv):
                df = pd.read_csv(result_csv)
                if "evaluation/return_mean" in df.columns:
                    normed_returns = (df["evaluation/return_mean"] - random) / denom
                    avg_score_under_top1_seed = normed_returns.mean()

        top1_df.append([
            subfolder, env,
            top1["seed"], top1["score"], top1["std"],
            avg_score_under_top1_seed
        ])
        avg_df.append([
            subfolder, env,
            avg["seed"], avg["avg_score"]
        ])

# === 最终表格合并 ===
df_top1 = pd.DataFrame(top1_df, columns=[
    "method", "env", "best_seed_top1", "top1_score", "top1_std", "avg_score_under_top1_seed"
])
df_avg = pd.DataFrame(avg_df, columns=[
    "method", "env", "best_seed_avg", "avg_score"
])

# === 输出到屏幕 ===
print("\n🎯 Best Seed and Scores Summary:\n")
print(df_top1.merge(df_avg, on=["method", "env"]).to_string(index=False))

# === 如需保存到文件，可取消下面注释 ===
# df_top1.merge(df_avg, on=["method", "env"]).to_csv("best_scores_summary.csv", index=False)