import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 配置 ===
sns.set(style="whitegrid", font_scale=1.2)
horizon = 4
algorithms = ["td3bc"]
environments = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
subfolders = ["dwm", "oracle", "dwm_h", "baseline", "ssorl"]

base_dir = f"exp-local/Horizon_{horizon}"
plot_dir = os.path.join("exp-local", "plots", f"horizon_{horizon}")
os.makedirs(plot_dir, exist_ok=True)

# 固定 x 轴步数
fixed_x = list(range(1, 21))

# 开始绘图
for algo in algorithms:
    for env in environments:
        plt.figure(figsize=(10, 6))

        for subfolder in subfolders:
            result_path = os.path.join(base_dir, algo, env, subfolder, "results.csv")

            if not os.path.exists(result_path):
                print(f"⚠️ Missing file: {result_path}")
                continue

            df = pd.read_csv(result_path)

            if "evaluation/return_mean" not in df.columns:
                print(f"⚠️ No evaluation/return_mean in {result_path}")
                continue

            # 限制长度为前 20 次
            df = df.head(20)

            # 对齐到固定 x
            x = fixed_x[:len(df)]
            y = df["evaluation/return_mean"].values
            std = df["evaluation/return_std"].values if "evaluation/return_std" in df.columns else np.zeros_like(y)

            # 绘图
            plt.plot(x, y, label=subfolder)
            plt.fill_between(x, y - std, y + std, alpha=0.2)

            print(f"✅ Loaded: {algo} - {env} - {subfolder}")

        plt.title(f"{algo} on {env} (Horizon {horizon})")
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.xticks(fixed_x)
        plt.legend()
        plt.tight_layout()

        # 保存图像
        save_name = f"{algo}_{env.replace('-', '_')}_curve_h{horizon}.pdf"
        save_path = os.path.join(plot_dir, save_name)
        plt.savefig(save_path)
        plt.close()

        print(f"📈 Saved plot: {save_path}")

print("🎯 All return curves saved to:", plot_dir)