"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pickle
import random

import numpy as np
import torch


class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        # To avoid generating another copy of trajs, keep sampling_ind
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        action_range,
        max_episode_len,
        num_past_transitions=0,
        sampling_ind=None,
        action_available_ind=None,
    ):

        super(TransitionDataset, self).__init__()
        self.trajs = trajectories
        self.transitions = self.__get_transitions__(
            state_dim,
            act_dim,
            state_mean,
            state_std,
            action_range,
            max_episode_len,
            num_past_transitions,
        )

        self.sampling_ind = sampling_ind

    def __get_transitions__(
        self,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        action_range,
        max_episode_len,
        num_past_transitions,
    ):

        transitions = []

        for traj in self.trajs:
            traj_transitions = []

            rtg = discount_cumsum(traj["rewards"], 1.0)
            avg_rtg = rtg / (max_episode_len - np.arange(len(rtg)))

            for si in range(0, traj["rewards"].shape[0]):
                traj_transitions.append(
                    get_transition(
                        traj,
                        avg_rtg,
                        si,
                        state_dim,
                        act_dim,
                        state_mean,
                        state_std,
                        action_range,
                        num_past_transitions,
                    )
                )
            transitions.append(traj_transitions)

        return transitions

    def __getitem__(self, index):
        # need to assign sampling_ind before calling this
        traj_transitions = self.transitions[self.sampling_ind[index]]
        si = random.randint(0, len(traj_transitions) - 1)
        return traj_transitions[si]

    def __len__(self):
        return len(self.sampling_ind)


class TransitionDatasetForShuffling(TransitionDataset):
    def __init__(
        self,
        trajectories,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        action_range,
        max_episode_len,
        num_past_transitions,
    ):

        super(TransitionDatasetForShuffling, self).__init__(
            trajectories,
            state_dim,
            act_dim,
            state_mean,
            state_std,
            action_range,
            max_episode_len,
            num_past_transitions=num_past_transitions,
        )
        self.transitions = [
            transition for traj_trans in self.transitions for transition in traj_trans
        ]

    def __getitem__(self, index):
        return self.transitions[index]

    def __len__(self):
        return len(self.transitions)


class TransformSamplingTransitions:
    def __init__(
        self,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        action_range,
        max_episode_len,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood.
        self.action_range = action_range
        self.max_episode_len = max_episode_len

    def __call__(self, traj):

        si = random.randint(0, traj["rewards"].shape[0] - 1)

        rtg = discount_cumsum(traj["rewards"], 1.0)
        avg_rtg = rtg / (self.max_episode_len - np.arange(len(rtg)))

        return get_transition(
            traj,
            avg_rtg,
            si,
            self.state_dim,
            self.act_dim,
            self.state_mean,
            self.state_std,
            self.action_range,
        )


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
        max_episode_len,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood.
        self.action_range = action_range

        self.max_episode_len = max_episode_len

    def __call__(self, traj):
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        ss = traj["observations"][si : si + self.max_len].reshape(-1, self.state_dim)
        aa = traj["actions"][si : si + self.max_len].reshape(-1, self.act_dim)
        rr = traj["rewards"][si : si + self.max_len].reshape(-1, 1)
        if "terminals" in traj:
            dd = traj["terminals"][si : si + self.max_len]  # .reshape(-1)
        else:
            dd = traj["dones"][si : si + self.max_len]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = ss.shape[0]

        timesteps = np.arange(si, si + tlen)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= self.max_episode_len] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= self.max_episode_len] = (
            self.max_episode_len - 1
        )  # padding cutoff

        rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        act_len = aa.shape[0]
        if tlen != act_len:
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), ss])
        ss = (ss - self.state_mean) / self.state_std

        # aa = np.concatenate([np.ones((self.max_len - tlen, self.act_dim)) * -10.0, aa])
        aa = np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa])
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr])
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 2, dd])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg])
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps])
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])

        ss = torch.from_numpy(ss).to(dtype=torch.float32)
        aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*self.action_range)
        rr = torch.from_numpy(rr).to(dtype=torch.float32)
        dd = torch.from_numpy(dd).to(dtype=torch.long)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long)
        ordering = torch.from_numpy(ordering).to(dtype=torch.long)
        padding_mask = torch.from_numpy(padding_mask)
        return ss, aa, rr, dd, rtg, timesteps, ordering, padding_mask
        # return ss, aa, rr, dd, rtg, ordering, mask


def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    max_episode_len,
    action_available_ind=None,
    num_workers=24,
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters

    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    sampling_ind = sample_trajs(traj_lens, sample_size)

    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
        max_episode_len=max_episode_len,
    )

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)

    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )


def create_transition_dataloader(
    dataset,
    batch_size,
    num_iters,
    num_workers=12,
):
    """
    Collect all the transitions from the trajectories and shuffle
    """

    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    traj_lens = np.array(
        [len(traj_transitions) for traj_transitions in dataset.transitions]
    )
    sampling_ind = sample_trajs(traj_lens, sample_size)
    dataset.sampling_ind = sampling_ind

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(traj_lens, sample_size):
    # reweight sampling so we sample according to timesteps instead of trajectories
    # traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)

    inds = np.random.choice(
        np.arange(len(traj_lens)),
        size=sample_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )
    return inds


def get_transition(
    traj,
    avg_rtg,
    si,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    action_range,
    num_past_transitions=0,
):
    if num_past_transitions == 0:
        ss = traj["observations"][si].reshape(-1)

    elif num_past_transitions > 0:
        traj_len = len(traj["observations"])
        # need to offset by 1 due to python syntax
        upper = min(traj_len, (si + 1) + 1)
        max_len = num_past_transitions + 2
        lower = max(0, si - num_past_transitions)
        ss = traj["observations"][lower:upper].reshape(-1, state_dim)
        tlen = ss.shape[0]
        ss = np.concatenate([np.zeros((max_len - tlen, state_dim)), ss])
        ss = ss.reshape(-1, state_dim)

    ss = (ss - state_mean) / state_std
    ss = torch.from_numpy(ss).to(dtype=torch.float32)

    next_ss = traj["next_observations"][si].reshape(-1)
    next_ss = (next_ss - state_mean) / state_std
    next_ss = torch.from_numpy(next_ss).to(dtype=torch.float32)

    aa = traj["actions"][si].reshape(-1)
    aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*action_range)

    rr = traj["rewards"][si].reshape(-1)
    rr = torch.from_numpy(rr).to(dtype=torch.float32)

    avg_rtgg = avg_rtg[si].reshape(-1)
    avg_rtgg = torch.from_numpy(avg_rtgg).to(dtype=torch.float32)

    not_done = 1.0 - traj["terminals"][si].reshape(-1)
    not_done = torch.from_numpy(not_done).to(dtype=torch.float32)

    return ss, next_ss, aa, rr, avg_rtgg, not_done


def setup_action_label(
    trajectories,
    action_available_threshold,
    action_available_perc,
):
    n = len(trajectories)
    size_upper_bound = int(n * action_available_perc)

    lower = 0
    upper = int(n * action_available_threshold)
    size = min(int(upper - lower), size_upper_bound)
    action_available_ind = np.random.choice(
        np.arange(lower, upper),
        size=size,
        replace=False,
    )
    action_unavailable_in_dist = np.array(
        [x for x in np.arange(lower, upper) if x not in action_available_ind]
    )
    action_unavailable_out_dist = np.arange(upper, n)

    return action_available_ind, action_unavailable_in_dist, action_unavailable_out_dist


def load_dataset(
    env_name,
    action_available_perc=1.0,
    action_available_threshold=0.5,
    folder="/home/stud/xhan/projects/ba/ssorl/dataset",
):
    # load dataset
    dataset_path = f"{folder}/{env_name}.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
    print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
    print("=" * 50)

    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    trajectories = [trajectories[ii] for ii in sorted_inds]

    (
        action_available_ind,
        action_unavailable_in_dist,
        action_unavailable_out_dist,
    ) = setup_action_label(
        trajectories,
        action_available_threshold,
        action_available_perc,
    )

    return (
        trajectories,
        state_mean,
        state_std,
        action_available_ind,
        action_unavailable_in_dist,
        action_unavailable_out_dist,
    )



def load_dwm_dataset(env_name, experiment_name, traj_len=8, folder="/home/stud/xhan/projects/ba/dwm/mb_dataset_2e6"):
    import numpy as np
    import os

    # 判断 RTG 设置
    if env_name.startswith("halfcheetah"):
        rtg_folder = "RTG_0.4"
    elif env_name.startswith("hopper") or env_name.startswith("walker"):
        rtg_folder = "RTG_0.8"
    else:
        raise ValueError(f"Unsupported environment name for RTG config: {env_name}")

    # 设置路径前缀：根据 experiment_name 决定
    if experiment_name == "dwm":
        prefix_folder = str(traj_len)
    elif experiment_name == "dwmt":
        prefix_folder = f"1.{traj_len}"
    else:
        raise ValueError(f"Unsupported experiment name: {experiment_name}")

    # 构造完整路径
    root_folder = os.path.join(folder, prefix_folder)
    horizon_folder = f"horizon_{traj_len}"
    env_folder = os.path.join(root_folder, env_name, horizon_folder, rtg_folder)

    obs_path = os.path.join(env_folder, "observations.npy")
    act_path = os.path.join(env_folder, "actions.npy")
    rew_path = os.path.join(env_folder, "rewards.npy")

    # 加载数据
    observations = np.load(obs_path).astype(np.float32)         # shape: (N, traj_len, obs_dim)
    rewards = np.load(rew_path).astype(np.float32).squeeze(-1)  # shape: (N, traj_len)
    actions = np.load(act_path).astype(np.float32)     

    num_traj = observations.shape[0]
    obs_dim = observations.shape[2]
    act_dim = actions.shape[1]

    trajectories = []
    print("📂 Loading trajectories from", obs_path)
    print("📊 Total number of trajectories:", num_traj)

    for i in range(num_traj):
        traj_obs = observations[i]  # shape: (traj_len, obs_dim)
        traj_next_obs = np.vstack([
            traj_obs[1:],
            np.zeros((1, obs_dim), dtype=np.float32)
        ])
        traj_rew = rewards[i]       # shape: (traj_len,)
        traj_act = np.full((traj_len, act_dim), np.nan, dtype=np.float32)
        traj_act[0] = actions[i]    # only a₀ known

        traj_term = np.zeros((traj_len,), dtype=bool)
        traj_term[-1] = True

        trajectories.append({
            "observations": traj_obs,
            "next_observations": traj_next_obs,
            "actions": traj_act,
            "rewards": traj_rew,
            "terminals": traj_term,
        })

    # Normalization
    states = np.concatenate([traj["observations"] for traj in trajectories], axis=0)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6

    return trajectories, state_mean, state_std



import os
import torch
import numpy as np

import os
import numpy as np
import torch

def save_predicted_trajs(env, horizon, trajs, save_root="/home/stud/xhan/projects/ba/ssorl/hopperme_dawm_dataset_final"):
    """
    保存完整的 predicted trajs（包含 observations, actions, rewards）列表为 .pt 文件。
    每个元素是一个 dict，包含 keys: 'observations', 'actions', 'rewards'
    
    文件保存路径结构为: {save_root}/{env}/horizon_{horizon}/predicted_trajs.pt
    
    如果文件已存在，则跳过保存。
    """
    # 拼接保存目录路径
    output_dir = os.path.join(save_root, env, f"horizon_{horizon}")
    os.makedirs(output_dir, exist_ok=True)

    # 最终保存路径
    output_path = os.path.join(output_dir, "predicted_trajs.pt")

    # 如果文件已存在，跳过保存
    if os.path.exists(output_path):
        print(f"⚠️ File already exists at {output_path}. Skipping save.")
        return

    print(f"✅ Will save {len(trajs)} trajectories to {output_path}\n")

    if len(trajs) > 0:
        print("📊 Trajectory structure summary (first 3 shown):")
        for idx, traj in enumerate(trajs[:3]):
            obs_len = len(traj.get("observations", []))
            act_len = len(traj.get("actions", []))
            rew_len = len(traj.get("rewards", []))
            print(f"  Traj {idx + 1}: obs={obs_len}, acts={act_len}, rews={rew_len}")

        print("\n📐 Example shapes:")
        sample = trajs[0]
        print(f"  - observation shape: {np.array(sample['observations']).shape}")
        print(f"  - action shape:      {np.array(sample['actions']).shape}")
        print(f"  - reward shape:      {np.array(sample['rewards']).shape}")
    else:
        print("⚠️ Warning: trajs list is empty!")

    # 保存文件
    torch.save(trajs, output_path)
    print(f"\n✅ Saved to {output_path}")
    
def save_ssorl_predicted_trajs(env, horizon, trajs, save_root="/home/stud/xhan/projects/ba/ssorl/hopperme_ssorl_dataset_final"):
    """
    保存完整的 predicted trajs（包含 observations, actions, rewards）列表为 .pt 文件。
    每个元素是一个 dict，包含 keys: 'observations', 'actions', 'rewards'
    
    文件保存路径结构为: {save_root}/{env}/horizon_{horizon}/predicted_trajs.pt
    
    如果文件已存在，则跳过保存。
    """
    # 拼接保存目录路径
    output_dir = os.path.join(save_root, env, f"horizon_{horizon}")
    os.makedirs(output_dir, exist_ok=True)

    # 最终保存路径
    output_path = os.path.join(output_dir, "predicted_trajs.pt")

    # 如果文件已存在，跳过保存
    if os.path.exists(output_path):
        print(f"⚠️ File already exists at {output_path}. Skipping save.")
        return

    print(f"✅ Will save {len(trajs)} trajectories to {output_path}\n")

    if len(trajs) > 0:
        print("📊 Trajectory structure summary (first 3 shown):")
        for idx, traj in enumerate(trajs[:3]):
            obs_len = len(traj.get("observations", []))
            act_len = len(traj.get("actions", []))
            rew_len = len(traj.get("rewards", []))
            print(f"  Traj {idx + 1}: obs={obs_len}, acts={act_len}, rews={rew_len}")

        print("\n📐 Example shapes:")
        sample = trajs[0]
        print(f"  - observation shape: {np.array(sample['observations']).shape}")
        print(f"  - action shape:      {np.array(sample['actions']).shape}")
        print(f"  - reward shape:      {np.array(sample['rewards']).shape}")
    else:
        print("⚠️ Warning: trajs list is empty!")

    # 保存文件
    torch.save(trajs, output_path)
    print(f"\n✅ Saved to {output_path}")
    
    
def save_dwmt_predicted_trajs(env, horizon, trajs, save_root="/home/stud/xhan/projects/ba/ssorl/dawmt_dataset"):
    """
    保存完整的 predicted trajs（包含 observations, actions, rewards）列表为 .pt 文件。
    每个元素是一个 dict，包含 keys: 'observations', 'actions', 'rewards'
    
    文件保存路径结构为: {save_root}/{env}/horizon_{horizon}/predicted_trajs.pt
    
    如果文件已存在，则跳过保存。
    """
    # 拼接保存目录路径
    output_dir = os.path.join(save_root, env, f"horizon_{horizon}")
    os.makedirs(output_dir, exist_ok=True)

    # 最终保存路径
    output_path = os.path.join(output_dir, "predicted_trajs.pt")

    # 如果文件已存在，跳过保存
    if os.path.exists(output_path):
        print(f"⚠️ File already exists at {output_path}. Skipping save.")
        return

    print(f"✅ Will save {len(trajs)} trajectories to {output_path}\n")

    if len(trajs) > 0:
        print("📊 Trajectory structure summary (first 3 shown):")
        for idx, traj in enumerate(trajs[:3]):
            obs_len = len(traj.get("observations", []))
            act_len = len(traj.get("actions", []))
            rew_len = len(traj.get("rewards", []))
            print(f"  Traj {idx + 1}: obs={obs_len}, acts={act_len}, rews={rew_len}")

        print("\n📐 Example shapes:")
        sample = trajs[0]
        print(f"  - observation shape: {np.array(sample['observations']).shape}")
        print(f"  - action shape:      {np.array(sample['actions']).shape}")
        print(f"  - reward shape:      {np.array(sample['rewards']).shape}")
    else:
        print("⚠️ Warning: trajs list is empty!")

    # 保存文件
    torch.save(trajs, output_path)
    print(f"\n✅ Saved to {output_path}")