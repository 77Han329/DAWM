import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class MBConfig(ParamsProto):
    # misc
    seed = 20 # [20, 40, 60, 80, 100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/dss/dsshome1/0C/di97zuq/project/DAWM/weights/8'
    dataset = 'hopper-medium-v2'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianDiffusion'
    horizon = 8
    env = 'hopper-medium-v2'
    n_diffusion_steps = 5
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.MBDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 1000
    termination_penalty = -100
    RTG = 0.8  # hopper, walker 0.8 | halfcheetah 0.4
    returns_scale = 400.0 # 400 for Hopper, 550 for Walker, 1200 for Halfcheetah

    ## training
    # n_steps_per_epoch = 10000
    loss_type = 'L2'
    n_train_steps = 2e6
    batch_size = 64
    learning_rate = 1e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 100000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False

    ## mb agent training
    sim_horizon = 7
    mb_model = 'td3bc'
    mb_n_train_steps = 5e5
    mb_batch_size = 128
    num_eval_episodes = 30
    action_available_perc: 0.1
    action_available_threshold: 0.5
    need_pseudo_label: True
    

# class mb_agents_config(ParamsProto):

#     sim_horizon = 7
#     mb_model = 'td3bc'
#     mb_n_train_steps = 5e5
#     mb_batch_size = 128

#     state_dim = 11
#     act_dim = 3
#     discount=0.99,
#     tau=0.005,
#     policy_noise=0.2,
#     noise_clip=0.5,
#     policy_freq=2,
#     alpha=2.5,
    
