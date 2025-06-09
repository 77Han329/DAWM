import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D

root_dir = "/home/stud/xhan/projects/ba/ssorl/st"
max_samples = 10000
max_dims_to_plot = 8

def plot_joint_pair_distribution_2d(first_pairs, pred_pairs, save_path, env_name):
    pca = PCA(n_components=2)
    data_all = np.vstack([first_pairs, pred_pairs])
    data_pca = pca.fit_transform(data_all)

    first_pca = data_pca[:len(first_pairs)]
    pred_pca = data_pca[len(first_pairs):]

    plt.figure(figsize=(6, 6))
    plt.scatter(first_pca[:, 0], first_pca[:, 1], alpha=0.5, label="(s₀, a₀)", s=3)
    plt.scatter(pred_pca[:, 0], pred_pca[:, 1], alpha=0.5, label="(s₁:₇, â₁:₇)", s=3)
    plt.title(f"Joint State-Action Pair Distribution (2D PCA) - {env_name}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved 2D PCA plot to {save_path}")
    plt.close()

def plot_joint_pair_distribution_3d(first_pairs, pred_pairs, save_path, env_name):
    pca = PCA(n_components=3)
    data_all = np.vstack([first_pairs, pred_pairs])
    data_pca = pca.fit_transform(data_all)

    first_pca = data_pca[:len(first_pairs)]
    pred_pca = data_pca[len(first_pairs):]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(first_pca[:, 0], first_pca[:, 1], first_pca[:, 2], alpha=0.5, label="(s₀, a₀)", s=5)
    ax.scatter(pred_pca[:, 0], pred_pca[:, 1], pred_pca[:, 2], alpha=0.5, label="(s₁:₇, â₁:₇)", s=5)
    ax.set_title(f"Joint State-Action Pair Distribution (3D PCA) - {env_name}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved 3D PCA plot to {save_path}")
    plt.close()

def plot_joint_pair_distribution_tsne_2d(first_pairs, pred_pairs, save_path, env_name):
    data_all = np.vstack([first_pairs, pred_pairs])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    data_tsne = tsne.fit_transform(data_all)

    first_tsne = data_tsne[:len(first_pairs)]
    pred_tsne = data_tsne[len(first_pairs):]

    plt.figure(figsize=(6, 6))
    plt.scatter(first_tsne[:, 0], first_tsne[:, 1], alpha=0.5, label="(s₀, a₀)", s=3)
    plt.scatter(pred_tsne[:, 0], pred_tsne[:, 1], alpha=0.5, label="(s₁:₇, â₁:₇)", s=3)
    plt.title(f"Joint State-Action Pair Distribution (t-SNE 2D) - {env_name}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved t-SNE 2D plot to {save_path}")
    plt.close()

def plot_joint_pair_distribution_umap_2d(first_pairs, pred_pairs, save_path, env_name):
    data_all = np.vstack([first_pairs, pred_pairs])
    reducer = umap.UMAP(n_components=2, random_state=42)
    data_umap = reducer.fit_transform(data_all)

    first_umap = data_umap[:len(first_pairs)]
    pred_umap = data_umap[len(first_pairs):]

    plt.figure(figsize=(6, 6))
    plt.scatter(first_umap[:, 0], first_umap[:, 1], alpha=0.5, label="(s₀, a₀)", s=3)
    plt.scatter(pred_umap[:, 0], pred_umap[:, 1], alpha=0.5, label="(s₁:₇, â₁:₇)", s=3)
    plt.title(f"Joint State-Action Pair Distribution (UMAP 2D) - {env_name}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved UMAP 2D plot to {save_path}")
    plt.close()

def check_shapes_and_plot(root):
    for env_name in os.listdir(root):
        env_path = os.path.join(root, env_name)
        if not os.path.isdir(env_path):
            continue

        horizon_path = os.path.join(env_path, "horizon_8")
        print(f"\n=== Environment: {env_name} ===")

        first_path = os.path.join(horizon_path, "state_action_pairs.pt")
        pred_path = os.path.join(horizon_path, "state_action_pairs_predicted.pt")

        if not (os.path.exists(first_path) and os.path.exists(pred_path)):
            print("Missing required files. Skipping.")
            continue

        first_tensor = torch.load(first_path)[:max_samples]
        pred_tensor = torch.load(pred_path)[:max_samples]

        print(f"state_action_pairs.pt: shape = {tuple(first_tensor.shape)}")
        print(f"state_action_pairs_predicted.pt: shape = {tuple(pred_tensor.shape)}")

        first_np = first_tensor.numpy()
        pred_np = pred_tensor.numpy()
        dim = first_np.shape[1]
        plot_dims = min(dim, max_dims_to_plot)

        # --- 原始直方图 ---
        plt.figure(figsize=(4 * plot_dims, 5))
        for i in range(plot_dims):
            plt.subplot(1, plot_dims, i + 1)
            plt.hist(first_np[:, i], bins=100, alpha=0.5, label="s0,a0")
            plt.hist(pred_np[:, i], bins=100, alpha=0.5, label="s1-7,a1-7")
            plt.title(f"Dim {i}")
            plt.xlabel("Value")
            plt.ylabel("Freq")
            if i == 0:
                plt.legend()
        plt.suptitle(f"Distribution Comparison - {env_name}", fontsize=16)
        plt.tight_layout()
        save_path_hist = os.path.join(horizon_path, "distribution_comparison.png")
        plt.savefig(save_path_hist)
        print(f"Saved histogram to {save_path_hist}")
        plt.close()

        # --- 可视化方法们 ---
        plot_joint_pair_distribution_2d(first_np, pred_np,
            os.path.join(horizon_path, "joint_distribution_pca_2d.png"), env_name)
        
        plot_joint_pair_distribution_3d(first_np, pred_np,
            os.path.join(horizon_path, "joint_distribution_pca_3d.png"), env_name)

        plot_joint_pair_distribution_tsne_2d(first_np, pred_np,
            os.path.join(horizon_path, "joint_distribution_tsne_2d.png"), env_name)

        plot_joint_pair_distribution_umap_2d(first_np, pred_np,
            os.path.join(horizon_path, "joint_distribution_umap_2d.png"), env_name)

if __name__ == "__main__":
    check_shapes_and_plot(root_dir)