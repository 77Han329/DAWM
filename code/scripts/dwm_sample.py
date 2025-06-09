import diffuser.utils as utils
from ml_logger import logger
import torch
import numpy as np
import os
import gym
import math
from tqdm import tqdm
from config.mb_config import MBConfig
from diffuser.utils.arrays import to_torch, to_np, to_device, batch_to_device
from diffuser.datasets.d4rl import suppress_output


def cycle(dl):
    while True:
        for data in dl:
            yield data

def dwm_sample(**deps):
    from ml_logger import logger, RUN
    from config.mb_config import MBConfig

    RUN._update(deps)
    MBConfig._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(MBConfig), RUN=vars(RUN))

    MBConfig.device = 'cuda'

    if MBConfig.predict_epsilon:
        prefix = f'predict_epsilon_{MBConfig.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{MBConfig.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(MBConfig.bucket, logger.prefix, 'checkpoint')
    
    print("reading weights from : ",loadpath)
    if MBConfig.save_checkpoints:
        #loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
        raise(NotImplementedError)
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=MBConfig.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(MBConfig.seed)

    dataset_config = utils.Config(
        MBConfig.loader,
        savepath='dataset_config.pkl',
        env=MBConfig.dataset,
        horizon=MBConfig.horizon,
        normalizer=MBConfig.normalizer,
        preprocess_fns=MBConfig.preprocess_fns,
        use_padding=MBConfig.use_padding,
        max_path_length=MBConfig.max_path_length,
        include_returns=MBConfig.include_returns,
        returns_scale=MBConfig.returns_scale,
        RTG = MBConfig.RTG
    )

    render_config = utils.Config(
        MBConfig.renderer,
        savepath='render_config.pkl',
        env=MBConfig.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    reward_dim = dataset.reward_dim
    indices = dataset.indices

    model_config = utils.Config(
        MBConfig.model,
        savepath='model_config.pkl',
        horizon=MBConfig.horizon,
        env=MBConfig.env,
        transition_dim=1,
        action_dim=action_dim,
        # trajectory_dim=(observation_dim + action_dim + reward_dim) + ((Config.horizon - 1) * (observation_dim + reward_dim)),
        # cond_dim=observation_dim + action_dim,
        dim_mults=MBConfig.dim_mults,
        dim=MBConfig.dim,
        returns_condition=MBConfig.returns_condition,
        device=MBConfig.device,
    )

    diffusion_config = utils.Config(
        MBConfig.diffusion,
        savepath='diffusion_config.pkl',
        horizon=MBConfig.horizon,
        env=MBConfig.env,
        observation_dim=observation_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        trajectory_dim=(observation_dim + action_dim + reward_dim) + ((MBConfig.horizon - 1) * (observation_dim + reward_dim)),
        n_timesteps=MBConfig.n_diffusion_steps,
        loss_type=MBConfig.loss_type,
        clip_denoised=MBConfig.clip_denoised,
        predict_epsilon=MBConfig.predict_epsilon,
        # hidden_dim=MBConfig.hidden_dim,
        ## loss weighting removed
        returns_condition=MBConfig.returns_condition,
        device=MBConfig.device,
        condition_guidance_w=MBConfig.condition_guidance_w,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=MBConfig.batch_size,
        train_lr=MBConfig.learning_rate,
        gradient_accumulate_every=MBConfig.gradient_accumulate_every,
        ema_decay=MBConfig.ema_decay,
        sample_freq=MBConfig.sample_freq,
        save_freq=MBConfig.save_freq,
        log_freq=MBConfig.log_freq,
        label_freq=int(MBConfig.n_train_steps // MBConfig.n_saves),
        save_parallel=MBConfig.save_parallel,
        bucket=MBConfig.bucket,
        n_reference=MBConfig.n_reference,
        train_device=MBConfig.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    env = dataset.env_str

    device = MBConfig.device
    
    obs_save_path = f'/home/atuin/b241dd/b241dd12/workarea/dwm/mb_dataset_2e6/{MBConfig.horizon}/{env}/horizon_{MBConfig.horizon}/RTG_{MBConfig.RTG}/observations.npy'
    act_save_path = f'/home/atuin/b241dd/b241dd12/workarea/dwm/mb_dataset_2e6/{MBConfig.horizon}/{env}/horizon_{MBConfig.horizon}/RTG_{MBConfig.RTG}/actions.npy'
    rewards_save_path = f'/home/atuin/b241dd/b241dd12/workarea/dwm/mb_dataset_2e6/{MBConfig.horizon}/{env}/horizon_{MBConfig.horizon}/RTG_{MBConfig.RTG}/rewards.npy'

    obs_directory = os.path.dirname(obs_save_path)
    act_directory = os.path.dirname(act_save_path)
    rew_directory = os.path.dirname(rewards_save_path)

    os.makedirs(obs_directory, exist_ok=True)
    os.makedirs(act_directory, exist_ok=True)
    os.makedirs(rew_directory, exist_ok=True)

    assert trainer.ema_model.condition_guidance_w == MBConfig.condition_guidance_w

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=MBConfig.mb_batch_size, shuffle=False)
    
    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=MBConfig.mb_batch_size, num_workers=0, shuffle=False, pin_memory=True
        ))

    obs = np.empty((0, MBConfig.horizon, observation_dim))
    acts = np.empty((0, action_dim))
    rewards = np.empty((0, MBConfig.horizon, reward_dim))

    processed_samples = 0
    
    num_batches = math.ceil(indices.shape[0] / MBConfig.mb_batch_size)
    print('num_batches: ', num_batches)
    
    # assert 0
    with tqdm(total=num_batches, desc="Sampling trajectories", unit="batch") as pbar:
        for i in range(num_batches):
            batch = next(dataloader)
            batch = batch_to_device(batch, device=device)
        
            conditions, returns = batch.conditions, batch.returns
            # print('batch condition shape: ', conditions.shape)  # [128, 14]
            samples = trainer.ema_model.conditional_sample(conditions, returns=returns)  # Sample from trained DWM
            print('samples: ', samples.dtype)
            # print( 'samples has shape: ', samples.shape)  # [128, 99]
            # assert 0
            
            samples = to_np(samples)  # (128, 99)
            print("to_np samples: ", samples.dtype)
            conditions = to_np(conditions)  # (128, 14)
        
            normed_observations_sim = []
            rewards_sim = []
            # normed_acts = []

            # references
            o_0 = conditions[:, :observation_dim]  # (128, 11)
            print("o_0: ", o_0.dtype)
            normed_acts = conditions[:, observation_dim:]  # (128, 3)
            normed_observations_sim.append(o_0)
            # normed_acts.append(a_0)

            # output samples
            sub_samples = samples[:, (observation_dim + action_dim):]
            r_0 = sub_samples[:, :reward_dim]
            sub_samples = sub_samples[:, reward_dim:]
            rewards_sim.append(r_0)

            for n in range(MBConfig.horizon):
                if n > 0:
                    o_n = sub_samples[:, :observation_dim]
                    normed_observations_sim.append(o_n)
                    sub_samples = sub_samples[:, observation_dim:]
                    r_n = sub_samples[:, :reward_dim]
                    rewards_sim.append(r_n)
                    sub_samples = sub_samples[:, reward_dim:]

            # print(len(normed_observations_sim))
            # print(normed_observations_sim[0].shape)
            normed_observations_sim = np.stack(normed_observations_sim, axis=0).transpose(1, 0, 2)  # [8, 128, 11] -> [128, 8, 11]
            # assert normed_observations_sim.shape == (128,8,11)
            # assert normed_acts.shape == (128,3)
            rewards_sim = np.stack(rewards_sim, axis=0).transpose(1, 0, 2)  # [8, 128, 1] -> [128, 8, 1]
            # assert rewards_sim.shape == (128,8,1)


            observations = dataset.normalizer.unnormalize(normed_observations_sim, 'observations')
            actions = dataset.normalizer.unnormalize(normed_acts, 'actions')

            obs = np.concatenate([obs, observations], axis=0)  # [n, 8, 11]
            acts = np.concatenate([acts, actions], axis=0)  # [n, 3]
            rewards = np.concatenate([rewards, rewards_sim], axis=0)  # [n, 8, 1]

            processed_samples += len(samples)
        
            np.save(obs_save_path, obs)
            np.save(act_save_path, acts)
            np.save(rewards_save_path, rewards)

            pbar.update(1)
            
            # assert 0

    logger.print(f"Finished sampling {processed_samples} trajectories from DWM using {env} dataset with RTG conditioning of {MBConfig.RTG}")

    # save all samples to local dir as npy files
    np.save(obs_save_path, obs)
    np.save(act_save_path, acts)
    np.save(rewards_save_path, rewards)
