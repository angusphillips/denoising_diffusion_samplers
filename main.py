#!/usr/bin/env python3

import hydra
from dds.configs.config import set_task, get_config
from dds.train_dds import train_dds


@hydra.main(config_path="config", config_name="main")
def main(hydra_config):
    base_config = get_config()
    base_config.model.dt = hydra_config.tf / hydra_config.num_steps
    task_config = set_task(base_config, task=hydra_config.experiment)
    task_config.trainer.epochs = hydra_config.training_iters
    task_config.trainer.learning_rate = hydra_config.lr
    task_config.trainer.log_every_n_epochs = hydra_config.log_every_n_epochs
    task_config.seed = hydra_config.seed
    task_config.wandb.group = hydra_config.group
    task_config.wandb.name = hydra_config.run_name

    task_config.model.sigma = hydra_config.sigma
    task_config.model.alpha = hydra_config.alpha

    out_dict = train_dds(task_config)


if __name__ == "__main__":
    main()
