#!/usr/bin/env python3

import hydra
from dds.configs.config import set_task, get_config
from dds.train_dds import train_dds
from dds.configs.param_settings import param_settings


@hydra.main(config_path="config", config_name="main")
def main(hydra_config):
    base_config = get_config()

    base_config.model.tfinal = hydra_config.tf
    base_config.model.step_scheme = base_config.model.step_scheme_dict[
        hydra_config.step_scheme
    ]
    base_config.model.step_scheme_key = hydra_config.step_scheme

    num_steps = hydra_config.steps_mult * hydra_config.base_steps

    base_config.model.dt = hydra_config.tf / num_steps
    base_config.model.num_steps = num_steps
    base_config.model.steps_mult = hydra_config.steps_mult

    task_config = set_task(base_config, task=hydra_config.task)

    task_config.seed = hydra_config.seed

    task_config.model.sigma = hydra_config.sigma
    task_config.model.alpha = hydra_config.alpha
    # override sigma and alpha with optimised settings:
    if hydra_config.reference_process_key == "pisstl":
        try:
            task_config.model.sigma = param_settings[hydra_config.task]["pis"]["sigma"][
                hydra_config.steps_mult
            ]
        except:
            print("no dice")
            return None
    if hydra_config.reference_process_key == "oudstl":
        try:
            task_config.model.alpha = param_settings[hydra_config.task]["dds"]["alpha"][
                hydra_config.steps_mult
            ]
        except:
            print("no dice")
            return None

    task_config.model.reference_process_key = hydra_config.reference_process_key

    if hydra_config.reference_process_key == "pisstl":
        hydra_config.use_vi_approx = False
    task_config.use_vi_approx = hydra_config.use_vi_approx
    task_config.trainer.log_every_n_epochs = hydra_config.log_every_n_epochs
    task_config.trainer.learning_rate = hydra_config.lr
    task_config.trainer.lr_sch_base_dec = hydra_config.exp_decay
    task_config.trainer.random_seed = hydra_config.seed
    task_config.wandb.group = hydra_config.group
    task_config.wandb.name = hydra_config.run_name
    task_config.save_samples = hydra_config.save_samples

    out_dict = train_dds(task_config)


if __name__ == "__main__":
    main()
