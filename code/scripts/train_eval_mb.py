
from ml_logger import logger
import torch
import numpy as np
import os
import importlib
import gym
import time
import math
from RL.models import td3bc
import RL
from RL.trainers import td3bc_trainer
from config.mb_config import MBConfig
from diffuser.utils.arrays import to_torch, to_np, to_device, batch_to_device
from diffuser.datasets.d4rl import suppress_output
from torch.utils.data import Dataset, DataLoader
from RL.evaluate_mb import create_td3bc_eval_fn, create_vec_eval_episodes_fn




# load sampled data
if __name__ == "__main__":

    class TrajectoryDataset(Dataset):

        def __init__(self, obs_file, act_file, rew_file):
            # Load the .npy files
            self.obs = np.load(obs_file)  # Shape: [n, 8, obs_dim]
            self.act = np.load(act_file)  # Shape: [n, act_dim]
            self.rew = np.load(rew_file)  # Shape: [n, 8, rew_dim]

            assert len(self.obs) == len(self.act) == len(self.rew)

        def __len__(self):
            # Return the number of samples available (n)
            return len(self.obs)

        def __getitem__(self, idx):
            # Return the observation, action, and reward for the given index (batch)

            obs = self.obs[idx]  # Shape [8, 11]
            act = self.act[idx]  # Shape [3]
            rew = self.rew[idx] # Shape [8, 1]

            return (
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(act, dtype=torch.float32),
                torch.tensor(rew, dtype=torch.float32),
            )
        
    env_spec = RL.rl_utils.get_env_spec(MBConfig.env)


    if MBConfig.mb_model == "td3bc":
        # actor and critic are sent to the device in the initialization function
        model = td3bc.TD3BC(
            name=MBConfig.mb_model,
            state_dim=env_spec.state_dim,
            act_dim=env_spec.act_dim,
            action_range=env_spec.action_range,
        )
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)

        # defined inside TD3BC training code
        loss_fn = None
        trainer = td3bc_trainer.TD3BCTrainer(
            model=model,
            sim_horizon=MBConfig.sim_horizon,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=MBConfig.device,
        )


    obs_path = f'/home/atuin/b241dd/b241dd10/dwm/mb_dataset/{MBConfig.env}/horizon_{MBConfig.horizon}/RTG_{MBConfig.RTG}/observations.npy'
    act_path = f'/home/atuin/b241dd/b241dd10/dwm/mb_dataset/{MBConfig.env}/horizon_{MBConfig.horizon}/RTG_{MBConfig.RTG}/actions.npy'
    rewards_path = f'/home/atuin/b241dd/b241dd10/dwm/mb_dataset/{MBConfig.env}/horizon_{MBConfig.horizon}/RTG_{MBConfig.RTG}/rewards.npy'

    dataset = TrajectoryDataset(obs_path, act_path, rewards_path)

    total_indices = len(dataset)
    print(total_indices)

    dataloader = DataLoader(dataset, batch_size=MBConfig.mb_batch_size, shuffle=True)

    num_updates = int(math.ceil(total_indices / MBConfig.mb_batch_size))
    print("steps in 1 epoch:", num_updates)

    n_epochs = int(math.ceil(MBConfig.mb_n_train_steps/num_updates))
    print(n_epochs)
    # assert 0

    for epoch in range(n_epochs):

        train_outputs = trainer.train_iteration(
                dataloader=dataloader, loss_fn=loss_fn, num_updates=num_updates,
        )
        print(train_outputs)
        assert 0
        
        
    # print("\n\n Make Eval Env\n\n")
    # eval_env = utils.make_eval_env(MBConfig.env, MBConfig.seed + 100)
    # eval_fn = create_td3bc_eval_fn(
    #     eval_env, cfg.num_eval_episodes, state_mean, state_std
    # )


