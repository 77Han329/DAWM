"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from https://github.com/sfujim/TD3_BC,
which is licensed under the MIT License.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
import RL.rl_utils as rl_utils


class TD3BCTrainer:
    def __init__(
        self,
        model,
        sim_horizon,
        actor_optimizer,
        critic_optimizer,
        device="cuda",
    ):
        self.model = model
        self.sim_horizon = sim_horizon
        self.device = device

        self.model.actor_optimizer = actor_optimizer
        self.model.critic_optimizer = critic_optimizer
        
        self.discount_factors = self.model.discount ** torch.arange(self.sim_horizon+1, dtype=torch.float32)

    def train_iteration(self, dataloader, loss_fn=None, num_updates=1000):
        self.model.actor.train()
        self.model.critic.train()

        train_start = time.time()
        critic_losses, actor_losses = [], []

        total_it = 0
        for step in range(num_updates):
            for _, batch in enumerate(dataloader):
                states, action, rewards = rl_utils.to_torch(
                    batch, device=self.device
                )
                not_done=1

                with torch.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = (torch.randn_like(action) * self.model.policy_noise).clamp(
                        -self.model.noise_clip, self.model.noise_clip
                    )
                    # print(states.shape)
                    # print(action.shape)
                    # print(rewards.shape)
                    next_action = (self.model.actor_target(states[:, self.sim_horizon]) + noise).clamp(
                        -self.model.max_action, self.model.max_action
                    )
                    # print("next_action: ", next_action.shape)

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.model.critic_target(
                        states[:, self.sim_horizon], next_action
                    )
                    target_Q = torch.min(target_Q1, target_Q2)
                    # print("target Q: ", target_Q.shape)

                    # discount_factors = self.model.discount ** torch.arange(self.sim_horizon+1, dtype=torch.float32)
                    discount_factors = self.discount_factors.to(self.device)
                    disc_rewards = rewards[:,:self.sim_horizon+1] * discount_factors[:, None]
                    acc_disc_rewards = disc_rewards.sum(dim=1)
                    # print("acc_disc_rewards: " , acc_disc_rewards.shape)

                    # target_Q = acc_disc_rewards + not_done * (self.model.discount ** self.sim_horizon) * target_Q
                    target_Q = acc_disc_rewards + (self.model.discount ** self.sim_horizon) * target_Q

                # Get current Q estimates
                current_Q1, current_Q2 = self.model.critic(states[:,0], action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q
                )
                # Optimize the critic
                self.model.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.model.critic_optimizer.step()
                critic_losses.append(critic_loss.detach().cpu().item())

                # Delayed policy updates
                if total_it % self.model.policy_freq == 0:

                    # Compute actor loss
                    pi = self.model.actor(states[:,0])
                    Q = self.model.critic.Q1(states[:,0], pi)
                    lmbda = self.model.alpha / Q.abs().mean().detach()

                    # maximize Q function + BC regularization
                    actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
                    # print(actor_loss)

                    # Optimize the actor
                    self.model.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.model.actor_optimizer.step()
                    actor_losses.append(actor_loss.detach().cpu().item())

                    # Update the frozen target models
                    for param, target_param in zip(
                        self.model.critic.parameters(),
                        self.model.critic_target.parameters(),
                    ):
                        target_param.data.copy_(
                            self.model.tau * param.data
                            + (1 - self.model.tau) * target_param.data
                        )

                    for param, target_param in zip(
                        self.model.actor.parameters(),
                        self.model.actor_target.parameters(),
                    ):
                        target_param.data.copy_(
                            self.model.tau * param.data
                            + (1 - self.model.tau) * target_param.data
                        )
                total_it += 1
                if total_it >= num_updates:
                    done_training = True
                    break
            
        logs = {}
        logs["time/training"] = time.time() - train_start
        logs["training/critic_loss"] = np.mean(critic_losses)
        logs["training/actor_loss"] = np.mean(actor_losses)

        return logs