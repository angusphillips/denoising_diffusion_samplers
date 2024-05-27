"""Experiment config file for SDE samplers.

This config acts as a super config which is properly innitailised by the sub
configs in the config directory.
"""
import distrax
import jax

import os
from jax import numpy as np

from ml_collections import config_dict as configdict
import optax

# from dds.configs import brownian_config
# from dds.configs import lgcp_config
# from dds.configs import log_reg_config
# from dds.configs import pretrained_nice_config
# from dds.configs import vae_config

from dds.discretisation_schemes import (
    cos_sq_fn_step_scheme,
    cos_sq_fn_step_scheme_unnorm,
)
from dds.discretisation_schemes import exp_fn_step_scheme
from dds.discretisation_schemes import linear_step_scheme
from dds.discretisation_schemes import linear_step_scheme_dds
from dds.discretisation_schemes import small_lst_step_scheme
from dds.discretisation_schemes import uniform_step_scheme
from dds.discretisation_schemes import uniform_step_scheme_dds

from dds.drift_nets import gelu
from dds.drift_nets import PISGRADNet
from dds.drift_nets import SimpleDriftNet

from dds.drift_nets_udp import UDPPISGRADNet
from dds.drift_nets_udp import UDPSimpleDriftNet

from dds.objectives import importance_weighted_partition_estimate
from dds.objectives import ou_terminal_loss
from dds.objectives import prob_flow_lnz
from dds.objectives import relative_kl_objective
from dds.objectives import controlled_ais_relative_kl_objective
from dds.objectives import controlled_ais_importance_weighted_partition_estimate_dds

from dds.stl_samplers import AugmentedBrownianFollmerSDESTL
from dds.stl_samplers import AugmentedOUDFollmerSDESTL
from dds.stl_samplers import AugmentedOUFollmerSDESTL
from dds.stl_samplers import AugmentedControlledAIS
from dds.stl_samplers import ULAAIS

from dds.targets import toy_targets
from dds.targets.distributions import (
    BayesianLogisticRegression,
    BrownianMissingMiddleScales,
    ChallengingTwoDimensionalMixture,
    FunnelDistribution,
    NormalDistributionWrapper,
    LogGaussianCoxPines,
    GaussianMixtureModel,
)

from dds.udp_samplers import AugmentedOUDFollmerSDEUDP


def get_config() -> configdict.ConfigDict:
    """Setup base config see /configs for more details."""

    config = configdict.ConfigDict()
    config.task = "no_name"
    config.use_vi_approx = True

    config.model = configdict.ConfigDict()
    config.dataset = configdict.ConfigDict()
    config.trainer = configdict.ConfigDict()
    config.vi_approx = configdict.ConfigDict()

    config.model.fully_connected_units = [64, 64]
    # config.model.fully_connected_units = [512, 512, 512]

    config.model.learn_betas = False
    config.trainer.timer = False

    config.model.batch_size = 300  # 128
    config.model.elbo_batch_size = 2000
    config.model.terminal_cost = ou_terminal_loss
    config.model.tfinal = 1.0
    config.model.dt = 0.01

    config.model.stl = False

    config.model.tpu = False

    config.model.network_key = "pis"
    config.model.network_dict = configdict.ConfigDict()
    config.model.network_dict.pis = PISGRADNet
    config.model.network_dict.pisudp = UDPPISGRADNet
    config.model.network_dict.vanilla = SimpleDriftNet
    config.model.network_dict.vanilla_udp = UDPSimpleDriftNet

    config.model.activation_key = "gelu"
    config.model.activation_dict = configdict.ConfigDict()
    config.model.activation_dict.gelu = gelu
    config.model.activation_dict.swish = jax.nn.swish
    config.model.activation_dict.relu = jax.nn.relu

    config.model.activation = config.model.activation_dict[config.model.activation_key]

    config.model.step_scheme_key = "cos_sq"
    config.model.step_scheme_dict = configdict.ConfigDict()
    config.model.step_scheme_dict.exp_dec = exp_fn_step_scheme
    config.model.step_scheme_dict.cos_sq = cos_sq_fn_step_scheme
    config.model.step_scheme_dict.cos_sq_unnorm = cos_sq_fn_step_scheme_unnorm
    config.model.step_scheme_dict.uniform = uniform_step_scheme
    config.model.step_scheme_dict.last_small = small_lst_step_scheme
    config.model.step_scheme_dict.linear_dds = linear_step_scheme_dds
    config.model.step_scheme_dict.linear = linear_step_scheme
    config.model.step_scheme_dict.uniform_dds = uniform_step_scheme_dds

    config.model.step_scheme = config.model.step_scheme_dict[
        config.model.step_scheme_key
    ]

    config.model.reference_process_key = "oudstl"
    config.model.reference_process_dict = configdict.ConfigDict()
    config.model.reference_process_dict.oustl = AugmentedOUFollmerSDESTL
    config.model.reference_process_dict.oudstl = AugmentedOUDFollmerSDESTL
    config.model.reference_process_dict.pisstl = AugmentedBrownianFollmerSDESTL
    config.model.reference_process_dict.oududp = AugmentedOUDFollmerSDEUDP
    config.model.reference_process_dict.cais = AugmentedControlledAIS
    config.model.reference_process_dict.ula = ULAAIS

    config.model.sigma = 1.0
    config.model.sigma_base = 0.25
    config.model.alpha = 1.0
    config.model.m = 1.0

    config.trainer.learning_rate = 0.0001

    config.trainer.epochs = 10000
    config.trainer.log_every_n_epochs = 1

    config.trainer.lr_sch_base_dec = 1.0  # 0.95 For funnel as per PIS repo
    config.model.stop_grad = True
    config.trainer.notebook = False
    config.trainer.simple_gaus_mean = 6.0

    config.trainer.objective = relative_kl_objective
    config.trainer.lnz_is_estimator = importance_weighted_partition_estimate
    config.trainer.lnz_pf_estimator = prob_flow_lnz
    config.model.detach_stl_drift = False
    config.model.detach_path = False
    config.model.log = False

    config.model.exp_dds = False  # ANGUS False = use ddpm_param

    config.trainer.random_seed = 0

    config.vi_approx.iters = 20_000
    config.vi_approx.batch_size = 512
    config.vi_approx.schedule = optax.constant_schedule(1e-3)

    config.eval = configdict.ConfigDict()
    config.eval.seeds = 100

    config.seed = 0

    config.wandb = configdict.ConfigDict()
    config.wandb.project = "dds"
    config.wandb.entity = "oxcsml"
    config.wandb.group = config.task
    config.wandb.name = config.task
    config.wandb.code_dir = os.getcwd()
    config.wandb.log = True
    return config


def set_task(
    config: configdict.ConfigDict, task: str = "lr_sonar"
) -> configdict.ConfigDict:
    """Sets up task specific attributes for config.

    Args:
      config:
      task:

    Raises:
      BaseException: raises exception if config class not implemented

    Returns:
      task processed config
    """
    config.task = task

    if task == "difficult_gaussian":
        config.model.input_dim = 1
        target_distribution = NormalDistributionWrapper(2.75, 0.25, 1, is_target=True)
        config.target_distribution = target_distribution

    elif task == "difficult_2d":
        config.model.input_dim = 2
        target_distribution = ChallengingTwoDimensionalMixture(dim=2, is_target=True)
        config.target_distribution = target_distribution

    elif task == "funnel":
        config.model.input_dim = 10
        target_distribution = FunnelDistribution(
            dim=config.model.input_dim, is_target=True
        )
        # log_prob_funn, _ = toy_targets.funnel(d=config.model.input_dim)
        config.target_distribution = target_distribution
        config.model.elbo_batch_size = 2000
        config.vi_approx.iters = 50_000

    elif task == "ion":
        config.model.input_dim = 35
        target_distribution = BayesianLogisticRegression(
            file_path="/vols/ziz/not-backed-up/anphilli/denoising_diffusion_samplers/data/ionosphere_full.pkl",
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "sonar":
        config.model.input_dim = 61
        target_distribution = BayesianLogisticRegression(
            file_path="/vols/ziz/not-backed-up/anphilli/denoising_diffusion_samplers/data/sonar_full.pkl",
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "brownian":
        config.model.input_dim = 32
        target_distribution = BrownianMissingMiddleScales(dim=32, is_target=True)
        config.target_distribution = target_distribution

    elif task == "lgcp":
        config.model.input_dim = 1600
        target_distribution = LogGaussianCoxPines(
            file_path="/vols/ziz/not-backed-up/anphilli/denoising_diffusion_samplers/data/fpines.csv",
            use_whitened=False,
            dim=1600,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm1":
        config.model.input_dim = 1
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm2":
        config.model.input_dim = 2
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm5":
        config.model.input_dim = 5
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm10":
        config.model.input_dim = 10
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm20":
        config.model.input_dim = 20
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm50":
        config.model.input_dim = 50
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    elif task == "gmm100":
        config.model.input_dim = 100
        target_distribution = GaussianMixtureModel(
            seed=42,
            dim=config.model.input_dim,
            n_mixes=40,
            loc_scaling=40,
            log_var_scaling=1.0,
            is_target=True,
        )
        config.target_distribution = target_distribution

    else:
        raise BaseException("Task config not implemented")

    tpu = config.model.tpu

    dtype = np.float32 if tpu else np.float64
    # import pdb; pdb.set_trace()
    step_scheme = config.model.step_scheme_dict[config.model.step_scheme_key]
    config.model.ts = step_scheme(0, config.model.tfinal, config.model.dt, dtype=dtype)

    config.model.stop_grad = True
    return config
