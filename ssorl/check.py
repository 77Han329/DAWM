import numpy as np
import os

# ====== 用户可配置参数 ======
env_name = "hopper-medium-replay-v2"
traj_len = 4
experiment_name = "dwm"  # 或 "dwmt"
folder = "/home/stud/xhan/projects/ba/dwm/mb_dataset"

# ====== 判断 RTG 文件夹名 ======
if env_name.startswith("halfcheetah"):
    rtg_folder = "RTG_0.4"
elif env_name.startswith("hopper") or env_name.startswith("walker"):
    rtg_folder = "RTG_0.8"
else:
    raise ValueError(f"Unsupported environment name: {env_name}")

# ====== 决定前缀路径名 ======
if experiment_name == "dwm":
    prefix_folder = str(traj_len)
elif experiment_name == "dwmt":
    prefix_folder = f"1.{traj_len}"
else:
    raise ValueError(f"Unsupported experiment name: {experiment_name}")

# ====== 拼接路径 ======
horizon_folder = f"horizon_{traj_len}"
action_path = os.path.join(folder, prefix_folder, env_name, horizon_folder, rtg_folder, "actions.npy")

# ====== 加载和打印信息 ======
if not os.path.exists(action_path):
    print(f"❌ File not found: {action_path}")
else:
    print(f"📂 Found action file at: {action_path}")
    actions = np.load(action_path)

    print(f"\n✅ Loaded actions with shape: {actions.shape}")
    print(f"📐 Dtype: {actions.dtype}")
    print(f"🔍 ndim: {actions.ndim}")

    if actions.size == 0:
        print("⚠️ Warning: actions.npy is empty.")
    elif actions.ndim == 1:
        print("⚠️ Warning: actions has shape (N,), reshaping to (N, 1).")
        actions = actions.reshape(-1, 1)
        print(f"➡️ New shape after reshape: {actions.shape}")

    print("\n📊 First 5 action entries:")
    print(actions[:5])