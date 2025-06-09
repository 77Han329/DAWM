if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep
    import sys
    import os
    #=============================================================================#
    # if you get error related to dataset
    # go diffuser.datasets/sequence.py 
    # comment line 91 & uncomment line 95
    #=============================================================================#
    config_file = sys.argv[1] if len(sys.argv) > 1 else "default_inv.jsonl"
    sweep = Sweep(RUN, Config).load(config_file)
    
    logger.diff = lambda *args, **kwargs: None


    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    # jaynes.listen()
    # jaynes.listen(timeout=10)  # Listen for feedback but timeout after 60 seconds
    # jaynes.clear() 