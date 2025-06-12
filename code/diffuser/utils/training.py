import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
import wandb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=64,
        train_lr=1e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=10000,
        save_freq=100000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                wandb.log(metrics)
                logger.log_metrics_summary(metrics, default_stats='mean')
               

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()

            self.step += 1
        


    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#



    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        env = self.dataset.env_str

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)  # (8, 99)
        conditions = to_np(batch.conditions)


        ## [ batch_size x trajectory_dim ]
        normed_obs = []
        normed_actions = []
        rewards = []
        normed_obs.append(trajectories[:, :self.dataset.observation_dim])
        normed_acts = trajectories[:, self.dataset.observation_dim: (self.dataset.observation_dim + self.dataset.action_dim)]
        rews = trajectories[:, (self.dataset.observation_dim + self.dataset.action_dim): (self.dataset.observation_dim + self.dataset.action_dim + self.dataset.reward_dim)]
        sub_trajectories = trajectories[:, (self.dataset.observation_dim + self.dataset.action_dim + self.dataset.reward_dim):]


        for n in range(self.dataset.horizon):
                if n > 0:
                    normed_obs.append(sub_trajectories[:, :self.dataset.observation_dim])
                    sub_trajectories = sub_trajectories[:, self.dataset.observation_dim:]
                    rewards.append(sub_trajectories[:, :self.dataset.reward_dim])
                    sub_trajectories = sub_trajectories[:, self.dataset.reward_dim:]

        obs = self.dataset.normalizer.unnormalize(np.array(normed_obs), 'observations')
        acts = self.dataset.normalizer.unnormalize(normed_acts, 'actions')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath_obs = os.path.join('training_reference', f'{env}', 'images', f'sample-reference.png')
        self.renderer.composite(savepath_obs, obs)
        actions_file_path = '/dss/dsshome1/0C/di97zuq/project/DAWM/training_reference/{}/action-reference.npy'.format(env)
        act_directory = os.path.dirname(actions_file_path)
        os.makedirs(act_directory, exist_ok=True)
        np.save(act_directory, acts)





    # def render_reference(self):
    #     '''
    #         renders training points
    #     '''
    #     self.vis_dataloader = cycle(torch.utils.data.DataLoader(
    #         self.dataset, batch_size=3213, num_workers=0, shuffle=True, pin_memory=True
    #     ))
    #     env = self.dataset.env_str
    #     batch = next(self.vis_dataloader)
    #     batch = batch_to_device(batch, device=self.device)
    #     trajectories = to_np(batch.trajectories)
    #     actions = trajectories[:, :, :self.dataset.action_dim:]
    #     actions = self.dataset.normalizer.unnormalize(actions, 'actions')
    #     rewards = trajectories[:, :, self.dataset.action_dim: self.dataset.action_dim + 1]

    #     actions_file_path = '/home/wiss/li/workarea/decision-diffuser/{}/actions/actions.npy'.format(env)
    #     rewards_file_path = '/home/wiss/li/workarea/decision-diffuser/{}/rewards/rewards.npy'.format(env)
    #     act_directory = os.path.dirname(actions_file_path)
    #     rew_directory = os.path.dirname(rewards_file_path)

    #     # Create the directory if it doesn't exist
    #     os.makedirs(act_directory, exist_ok=True)
    #     os.makedirs(rew_directory, exist_ok=True)

    #     np.save(act_directory, actions)
    #     np.save(rew_directory, rewards)
    #     ## get a temporary dataloader to load a single batch
    #     # dataloader_tmp = cycle(torch.utils.data.DataLoader(
    #     #     self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
    #     # ))
    #     # batch = dataloader_tmp.__next__()
    #     # dataloader_tmp.close()

    #     ## get trajectories and condition at t=0 from batch
    #     # trajectories = to_np(batch.trajectories)
    #     # conditions = to_np(batch.conditions[0])[:,None]

    #     ## [ batch_size x horizon x observation_dim ]
    #     normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
    #     observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

    #     # from diffusion.datasets.preprocessing import blocks_cumsum_quat
    #     # # observations = conditions + blocks_cumsum_quat(deltas)
    #     # observations = conditions + deltas.cumsum(axis=1)

    #     #### @TODO: remove block-stacking specific stuff
    #     # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
    #     # observations = blocks_add_kuka(observations)
    #     ####

    #     # savepath = os.path.join('images', f'sample-reference.png')
    #     iter_num = trajectories.shape[0]  
    #     step_num = trajectories.shape[1]      
    #     for i in range(iter_num):
    #         for j in range(step_num):
    #             savepath = '/home/wiss/li/workarea/decision-diffuser/{}/images/img_{}_{}.png'.format(env, i, j)
    #             self.renderer.composite(savepath, np.expand_dims(observations[i][j], axis=0), partial=True)   

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            env = self.dataset.env_str
            conditions = to_device(batch.conditions, self.device)

            ## repeat each item in conditions `n_samples` times
            conditions = einops.repeat(conditions, 'b d -> (repeat b) d', repeat=n_samples)

            ref_obs = conditions[:, : self.dataset.observation_dim]
            ref_actions = conditions[:, self.dataset.observation_dim:]

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)
            ref_obs = to_np(ref_obs)
            ref_actions = to_np(ref_actions)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = []
            rewards = []
            obs_0 = samples[:, :self.dataset.observation_dim]
            normed_observations.append(obs_0)
            sub_samples = samples[:, (self.dataset.observation_dim + self.dataset.action_dim):]
            re_0 = sub_samples[:, :self.dataset.reward_dim]
            sub_samples = sub_samples[:, self.dataset.reward_dim:]
            rewards.append(re_0)

            for n in range(self.dataset.horizon):
                if n > 0:
                    ob = sub_samples[:, :self.dataset.observation_dim]
                    normed_observations.append(ob)
                    sub_samples = sub_samples[:, self.dataset.observation_dim:]
                    re = sub_samples[:, :self.dataset.reward_dim]
                    rewards.append(re)
                    sub_samples = sub_samples[:, self.dataset.reward_dim:]

            ## [ n_samples x (ref_dim + sample_dim)]
            # normed_observations = np.concatenate([
            #     ref_obs,
            #     normed_observations
            # ], axis=-1)

            assert len(normed_observations) == self.dataset.horizon == len(rewards)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.array(normed_observations)
            ref_actions = np.array(ref_actions)
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            ref_acts = self.dataset.normalizer.unnormalize(ref_actions, 'actions')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####



            savepath = os.path.join('training_sample', f'{env}', 'images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)
            actions_file_path = f'/dss/dsshome1/0C/di97zuq/project/DAWM/training_sample/{env}/action-reference-{i}.npy'
            rewards_file_path = f'/dss/dsshome1/0C/di97zuq/project/DAWM/training_sample/{env}/sample-rewards-{i}.npy'
            act_directory = os.path.dirname(actions_file_path)
            os.makedirs(act_directory, exist_ok=True)
            np.save(act_directory, ref_acts)
            rew_directory = os.path.dirname(rewards_file_path)
            os.makedirs(act_directory, exist_ok=True)
            np.save(rew_directory, rewards)