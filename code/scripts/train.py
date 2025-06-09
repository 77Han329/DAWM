import diffuser.utils as utils
import torch
import math
import wandb
import os

def main(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config
    
    RUN._update(deps)
    Config._update(deps)
    wandb.init(
        project="dawm",
        entity="dawm",
        name=f"{Config.env}_h{Config.horizon}",
        config=vars(Config)
    )
    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    reward_dim = dataset.reward_dim
    total_indices = len(dataset)
    n_steps_per_epoch = math.ceil(total_indices//Config.batch_size)

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    # Config.model = 'models.TemporalUnet'
    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        env=Config.env,
        transition_dim=1,
        # trajectory_dim=(observation_dim + action_dim + reward_dim) + ((Config.horizon - 1) * (observation_dim + reward_dim)),
        action_dim=action_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
        device=Config.device,
    )
    # Config.diffusion = 'models.GaussianDiffusion'
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        env=Config.env,
        observation_dim=observation_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        # transition_dim_=1,
        trajectory_dim=(observation_dim + action_dim + reward_dim) + ((Config.horizon - 1) * (observation_dim + reward_dim)),
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(n_steps_per_epoch // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, renderer)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(*batch)

    # =========================Zongyue_DataGen_Start===============================#
    # trainer.render_reference()
    # data_test = batch[0]  # first data in the batch
    # data_test = data_test.unsqueeze(0)
    # data_diffusion_original = data_test.unsqueeze(2)
    # renderer.render_diffusion(None, data_diffusion_original)
    # =========================Zongyue_DataGen_End=================================#

    loss.backward()
    logger.print('✓')

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#
    
    # wandb.init(project="DAWM", entity="bathesis")
    
    # n_epochs = int(Config.n_train_steps // n_steps_per_epoch)
    # #n_epochs = int(1e6 // n_steps_per_epoch)
    
    # #loadcheckpoint for further training steps. Comment if from scratch.
    # loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    
    # if Config.save_checkpoints:
    #     #loadpath = os.path.joisn(loadpath, f'state_{self.step}.pt')
    #     # raise(NotImplementedError)
    # else:
    #     loadpath = os.path.join(loadpath, 'state.pt')
    
    # state_dict = torch.load(loadpath, map_location=Config.device)
    
    # trainer.step = state_dict['step']
    # trainer.model.load_state_dict(state_dict['model'])
    # trainer.ema_model.load_state_dict(state_dict['ema'])


    # for i in range(n_epochs):
    #     logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
    #     trainer.train(n_train_steps=n_steps_per_epoch)
        
    # logger.print('Training completed with 1e6 iterations.')
    # trainer.save()

    # Initialize wandb
    # wandb.init(project="DAWM", entity="bathesis", name=Config.env)

    # # Define total training steps and compute number of epochs
    # total_train_steps = int(1e6)
    # n_epochs = int(total_train_steps // n_steps_per_epoch)

    # # Prepare checkpoint path
    # checkpoint_dir = os.path.join(Config.bucket, logger.prefix)
    # checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint', 'state.pt')

    # # Try to load checkpoint if available
    # if os.path.exists(checkpoint_file):
    #     print(f'✅ Loading checkpoint from {checkpoint_file}')
    #     state_dict = torch.load(checkpoint_file, map_location=Config.device)
    #     trainer.step = state_dict['step']
    #     trainer.model.load_state_dict(state_dict['model'])
    #     trainer.ema_model.load_state_dict(state_dict['ema'])
    # else:
    #     print(f'⚠️ No checkpoint found at {checkpoint_file}, starting from scratch.')

    # # Main training loop
    # for i in range(n_epochs):
    #     logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
    #     trainer.train(n_train_steps=n_steps_per_epoch)

    # logger.print('✅ Training completed with 1e6 iterations.')

    # # Save final model
    # trainer.save()


    # Initialize wandb (一次就够)
    

    # Checkpoint 配置
    checkpoint_dir = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    checkpoint_file = os.path.join(checkpoint_dir, 'state.pt')

    # 加载 checkpoint（如果存在）
    start_step = 0
    if os.path.exists(checkpoint_file):
        print(f'✅ Loading checkpoint from {checkpoint_file}')
        state_dict = torch.load(checkpoint_file, map_location=Config.device)
        trainer.step = state_dict['step']
        trainer.model.load_state_dict(state_dict['model'])
        trainer.ema_model.load_state_dict(state_dict['ema'])
        start_step = trainer.step
    else:
        print(f'⚠️ No checkpoint found at {checkpoint_file}, starting from scratch.')

    # 设置训练参数
    total_train_steps = int(3e6)  # 训练总步数
    n_epochs = total_train_steps // n_steps_per_epoch

    # 计算起始 epoch（恢复时用）
    start_epoch = start_step // n_steps_per_epoch

    # 开始训练循环
    for i in range(start_epoch, n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=n_steps_per_epoch)
        trainer.save()  # 每轮都保存一次 checkpoint（可选）

    logger.print('✅ Training completed with 1e6 iterations.')