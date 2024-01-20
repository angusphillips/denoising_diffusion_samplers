#!/usr/bin/env python3

import hydra
from dds.configs.config import set_task, get_config
from dds.train_dds import train_dds


@hydra.main(config_path="config", config_name="main")
def main(hydra_config):
    base_config = get_config()

    base_config.model.tfinal = hydra_config.tf
    base_config.model.step_scheme = base_config.model.step_scheme_dict[
        hydra_config.step_scheme
    ]
    base_config.model.step_scheme_key = hydra_config.step_scheme
    base_config.model.dt = hydra_config.tf / hydra_config.num_steps

    task_config = set_task(base_config, task=hydra_config.task)

    task_config.seed = hydra_config.seed
    task_config.model.sigma = hydra_config.sigma
    task_config.model.alpha = hydra_config.alpha
    task_config.model.reference_process_key = hydra_config.reference_process_key
    task_config.use_vi_approx = hydra_config.use_vi_approx
    task_config.trainer.log_every_n_epochs = hydra_config.log_every_n_epochs
    task_config.trainer.learning_rate = hydra_config.lr
    task_config.trainer.lr_sch_base_dec = hydra_config.exp_decay
    task_config.trainer.random_seed = hydra_config.seed
    task_config.wandb.group = hydra_config.group
    task_config.wandb.name = hydra_config.run_name

    out_dict = train_dds(task_config)


if __name__ == "__main__":
    main()
