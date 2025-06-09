"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import csv
import pickle
from pathlib import Path
import wandb

import numpy as np


class Logger:
    def __init__(self, log_path: Path, cfg):

        self.log_path = log_path
        self.use_wandb = cfg.log_to_wandb  # ✅ fix name
        self.writer = None

        
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                entity=cfg.wandb_entity,
                config=dict(cfg)
            )

    def log_metrics(self, outputs, iter_num, csv_file_name, print_output):
        if print_output:
            for k, v in outputs.items():
                print(f"{k}: {v}")
        for k, v in outputs.items():
            if self.writer:
                self.writer.add_scalar(k, v, iter_num)
        if self.use_wandb:
            wandb.log(outputs, step=iter_num)

        csv_file_path = self.log_path / csv_file_name
        self.log_to_csv(outputs, csv_file_path)

    def log_to_csv(self, outputs, csv_file_path):
        should_write_header = not csv_file_path.is_file()

        with open(csv_file_path, "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=sorted(outputs.keys()))
            if should_write_header:
                csv_writer.writeheader()

            csv_writer.writerow(outputs)

    def log_to_file(self, fname: str, outputs: list):
        with open(self.log_path / fname, "w") as f:
            f.writelines(f"{o}\n" for o in outputs)

    def pickle_obj(self, fname: str, obj):
        with open(self.log_path / fname, "wb") as f:
            pickle.dump(obj, f)
