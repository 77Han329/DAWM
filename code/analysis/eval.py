if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import sys
    import jaynes
    from scripts.dwm_sample import dwm_sample
    from config.mb_config import MBConfig
    from params_proto.neo_hyper import Sweep

    config_file = sys.argv[1] if len(sys.argv) > 1 else "default_inv_sample.jsonl"
    sweep = Sweep(RUN, MBConfig).load(config_file)
    logger.diff = lambda *args, **kwargs: None  # <- Disable diffing

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(dwm_sample, **kwargs, logger_diff=False)  # ✅ 使用正确参数
        jaynes.run(thunk)

    #jaynes.listen()
    #jaynes.clear()