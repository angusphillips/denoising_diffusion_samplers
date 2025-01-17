"""Main training file.

For training diffusion based samplers (OU reversal SDE and Follmer SDE )
"""
import functools
import timeit
from typing import Any, List, Tuple, Optional
from absl import app, flags

from absl import logging
import haiku as hk
import distrax
import jax
import jax.numpy as jnp

from ml_collections import config_dict as configdict
from ml_collections import config_flags

import numpy as onp
import optax

from jaxline import utils
import tqdm

from dds.configs.config import set_task
from dds.data_paths import results_path
from dds.targets.distributions import (
    NormalDistributionWrapper,
    WhitenedDistributionWrapper,
)
from dds.utils import flatten_nested_dict
import wandb

from dds.vi import get_variational_approx

import warnings

warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS
Writer = Any

# lr_sonar, funnel, lgcp, ion, nice, vae, brownian
_TASK = flags.DEFINE_string("task", "lr_sonar", "Inference task name.")
config_flags.DEFINE_config_file(
    "config",
    "configs/config.py",
    lock_config=False,
    help_string="Path to ConfigDict file.",
)


def update_detached_params(
    trainable_params,
    non_trainable_params,
    attached_network_name="simple_drift_net",
    detached_network_name="stl_detach",
):
    """Auxiliary function updating detached params for STL.

    Args:
        trainable_params:
        non_trainable_params:
        attached_network_name:
        detached_network_name:
    Returns:
      Returns non trainable params
    """

    if len(trainable_params) != len(non_trainable_params):
        return non_trainable_params

    for key in trainable_params.keys():
        if attached_network_name in key:
            key_det = key.replace(attached_network_name, detached_network_name)
        else:
            key_det = key.replace("diffusion_network", detached_network_name + "_diff")
        non_trainable_params[key_det] = trainable_params[
            key
        ]  # pytype: disable=unsupported-operands

    return non_trainable_params


def train_dds(config: configdict.ConfigDict):
    # ) -> Tuple[hk.Params, hk.State, hk.TransformedWithState, jnp.ndarray,
    #            List[float]]:
    """Train Follmer SDE.

    Args:
      config : ConfigDict with model and training parameters.

    Returns:
      Tuple containing params, state, function that runs learned sde, and losses
    """
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "group": config.wandb.group,
        "name": config.wandb.name,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        key = jax.random.PRNGKey(config.seed)
        if config.use_vi_approx:
            logging.info("Learning VI approximation")
            key, key_ = jax.random.split(key)
            vi_params = get_variational_approx(
                dim=config.model.input_dim,
                lr_schedule=config.vi_approx.schedule,
                batch_size=config.vi_approx.batch_size,
                iters=config.vi_approx.iters,
                rng=key_,
                target_distribution=config.target_distribution,
            )
            target_distribution = WhitenedDistributionWrapper(
                config.target_distribution,
                vi_params["Variational"]["means"],
                vi_params["Variational"]["scales"],
            )
        else:
            target_distribution = config.target_distribution

        # train setup
        data_dim = config.model.input_dim
        device_no = jax.device_count()

        alpha = config.model.alpha
        sigma = config.model.sigma
        m = config.model.m

        # post setup model vars
        # config.model.source_obj = distrax.MultivariateNormalDiag(
        #     jnp.zeros(config.model.input_dim),
        #     config.model.sigma * jnp.ones(config.model.input_dim),
        # )
        config.model.source_obj = NormalDistributionWrapper(
            mean=0.0,
            scale=config.model.sigma,
            dim=config.model.input_dim,
            is_target=False,
        )
        config.model.source = lambda x: config.model.source_obj.evaluate_log_density(
            x, 0
        )[0]

        batch_size_ = int(config.model.batch_size / device_no)
        batch_size_elbo = int(config.model.elbo_batch_size / device_no)

        step_scheme = config.model.step_scheme_dict[config.model.step_scheme_key]

        dt = config.model.dt

        if config.model.reference_process_key == "oududp":
            key_conversion = {
                "pis": "pisudp",
                "vanilla": "vanilla_udp",
                "tmpis": "tmpis_udp",
            }
            # "pisudp"
            config.model.network_key = key_conversion[config.model.network_key]

        net_key = config.model.network_key
        network = config.model.network_dict[net_key]

        tpu = config.model.tpu

        detach_dif_path, detach_dritf_path = (
            config.model.detach_path,
            config.model.detach_path,
        )

        target = target_distribution.evaluate_log_density

        tfinal = config.model.tfinal
        lnpi = target_distribution.evaluate_log_density

        ref_proc_key = config.model.reference_process_key
        ref_proc = config.model.reference_process_dict[ref_proc_key]

        trim = (
            2 if "stl" in str(ref_proc).lower() or "udp" in str(ref_proc).lower() else 1
        )

        stl = config.model.stl

        brown = "brown" in str(ref_proc).lower()

        seed = config.trainer.random_seed if "random_seed" in config.trainer else 42

        # task directory (currently not in use)
        task = config.task
        method = config.model.reference_process_key
        task_path = results_path + f"/{task}" + f"/{ref_proc_key}" + f"/{net_key}"
        task_path += f"/{method}"

        # checkpoiting variables for wandb
        nsteps = config.model.ts.shape[0]
        keep_every_nth = int(config.trainer.epochs / 125)
        file_name = (
            f"/alpha_{alpha}_sigma_{sigma}_epochs_{config.trainer.epochs}"
            + f"_task_{task}_seed_{seed}_steps_{nsteps}_stl_{stl}_{method}"
            + f"_scheme_{config.model.step_scheme_key}_ddpm_test11_chk"
        )
        _ = task_path + file_name

        detach_stl_drift = (
            config.model.detach_stl_drift
            if "detach_stl_drift" in config.model
            else False
        )

        drift_network = lambda: network(config.model, data_dim, "simple_drift_net")

        ############## wandb logging  place holder ################
        data_id = "denoising_diffusion_samplers"  # Project name

        def _forward_fn(
            batch_size: int,
            density_state: int,
            training: bool = True,
            ode=False,
            exact=False,
            dt_=dt,
        ) -> jnp.ndarray:
            model_def = ref_proc(
                sigma,
                data_dim,
                drift_network,
                tfinal=tfinal,
                dt=dt_,
                step_scheme=step_scheme,
                alpha=alpha,
                target=target,
                tpu=tpu,
                detach_stl_drift=detach_stl_drift,
                diff_net=None,
                detach_dritf_path=detach_dritf_path,
                detach_dif_path=detach_dif_path,
                m=m,
                log=config.model.log,
                exp_bool=config.model.exp_dds,
                exact=exact,
            )

            return model_def(batch_size, density_state, training, ode=ode)

        forward_fn = hk.transform_with_state(_forward_fn)

        # opt and loss setup
        seq = hk.PRNGSequence(seed)
        rng_key = next(seq)
        # subkeys = jax.random.split(rng_key, device_no)
        subkeys = utils.bcast_local_devices(rng_key)

        p_init = jax.pmap(
            functools.partial(
                forward_fn.init, batch_size=batch_size_, density_state=0, training=True
            ),
            axis_name="num_devices",
        )

        params, model_state = p_init(subkeys)

        trainable_params, non_trainable_params = hk.data_structures.partition(
            lambda module, name, value: "stl_detach" not in module, params
        )

        clipper = optax.clip(1.0)
        base_dec = config.trainer.lr_sch_base_dec
        scale_by_adam = optax.scale_by_adam()
        # if base_dec == 0:
        #   scale_by_lr = optax.scale(-config.trainer.learning_rate)
        #   opt = optax.chain(clipper, scale_by_adam, scale_by_lr)
        # else:
        transition_steps = 50
        exp_lr = optax.exponential_decay(
            config.trainer.learning_rate, transition_steps, base_dec
        )
        scale_lr = optax.scale_by_schedule(exp_lr)
        opt = optax.chain(clipper, scale_by_adam, scale_lr, optax.scale(-1))

        # opt = optax.adam(learning_rate=config.trainer.learning_rate)
        opt_state = jax.pmap(opt.init)(trainable_params)

        @functools.partial(
            jax.pmap,
            axis_name="num_devices",
            static_broadcasted_argnums=(3, 5, 6, 7),
        )
        def forward_fn_jit(
            params,
            model_state: hk.State,
            subkeys: jnp.ndarray,
            batch_size: jnp.ndarray,
            density_state: int,
            ode=False,
            exact=False,
            dt_=dt,
        ):
            samps, _, density_state = forward_fn.apply(
                params,
                model_state,
                subkeys,
                int(batch_size / device_no),
                density_state=density_state,
                is_training=False,
                ode=ode,
                exact=exact,
                dt_=dt_,
            )
            samps = jax.device_get(samps)
            density_state = jax.device_get(density_state)

            augmented_trajectory, ts = samps
            return (augmented_trajectory, ts), _, density_state

        def forward_fn_wrap(
            params,
            model_state: hk.State,
            rng_key: jnp.ndarray,
            batch_size: jnp.ndarray,
            density_state: int,
            ode=False,
            exact=False,
            dt_=dt,
        ):
            subkeys = jax.random.split(rng_key, device_no)
            (augmented_trajectory, ts), _, density_state = forward_fn_jit(
                params, model_state, subkeys, batch_size, density_state, ode, exact, dt_
            )

            dv, ns, t, _ = augmented_trajectory.shape
            augmented_trajectory = augmented_trajectory.reshape(dv * ns, t, -1)
            return (augmented_trajectory, utils.get_first(ts)), _, density_state

        def full_objective(
            trainable_params,
            non_trainable_params,
            model_state: hk.State,
            rng_key: jnp.ndarray,
            batch_size: int,
            density_state: int,
            is_training: bool = True,
            ode: bool = False,
            stl: bool = False,
            exact: bool = False,
        ):
            params = hk.data_structures.merge(trainable_params, non_trainable_params)
            (augmented_trajectory, _, density_state), model_state = forward_fn.apply(
                params,
                model_state,
                rng_key,
                batch_size,
                density_state,
                True,
                ode,
                exact,
            )

            # import pdb; pdb.set_trace()
            gpartial = functools.partial(
                config.model.terminal_cost,
                lnpi=lnpi,
                source=config.model.source,
                tfinal=tfinal,
                brown=brown,
            )

            if is_training:
                loss, density_state = config.trainer.objective(
                    augmented_trajectory,
                    gpartial,
                    density_state,
                    stl=stl,
                    trim=trim,
                    dim=data_dim,
                )
            elif not ode:
                loss, density_state = config.trainer.lnz_is_estimator(
                    augmented_trajectory, gpartial, density_state, dim=data_dim
                )
            else:
                loss, density_state = config.trainer.lnz_pf_estimator(
                    augmented_trajectory,
                    config.model.source,
                    config.model.target,
                    density_state=density_state,
                )
            return loss, (model_state, density_state)

        @functools.partial(
            jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(5)
        )
        def update(
            trainable_params,
            non_trainable_params,
            model_state: hk.State,
            opt_state: Any,
            rng_key: jnp.ndarray,
            batch_size: jnp.ndarray,
            density_state: int,
        ) -> Tuple[Any, Any, Any, Any, int]:
            grads, (new_model_state, density_state) = jax.grad(
                full_objective, has_aux=True
            )(
                trainable_params,
                non_trainable_params,
                model_state,
                rng_key,
                batch_size,
                density_state,
                is_training=True,
                stl=stl,
            )
            grads = jax.lax.pmean(grads, axis_name="num_devices")

            updates, opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(trainable_params, updates)
            return new_params, opt_state, new_model_state, jax.device_get(density_state)

        @functools.partial(
            jax.pmap,
            axis_name="num_devices",
            static_broadcasted_argnums=(4, 6, 7, 8),
        )
        def jited_val_loss(
            trainable_params,
            non_trainable_params,
            model_state: hk.State,
            rng_key: jnp.ndarray,
            batch_size: jnp.ndarray,
            density_state: int,
            is_training: bool = True,
            ode: bool = False,
            exact: bool = False,
        ) -> Tuple[Any, hk.State, int]:
            loss, (new_model_state, density_state) = full_objective(
                trainable_params,
                non_trainable_params,
                model_state,
                rng_key,
                batch_size,
                density_state,
                is_training=is_training,
                ode=ode,
                stl=False,
                exact=exact,
            )

            loss = jax.lax.pmean(loss, axis_name="num_devices")
            return loss, new_model_state, density_state

        def eval_report(
            trainable_params,
            non_trainable_params,
            model_state: hk.State,
            rng_key: jnp.ndarray,
            batch_size: int,
            density_state: int,
            epoch: int,
            loss_list: List[float],
            is_training: bool = True,
            print_flag: bool = False,
            ode: bool = False,
            exact: bool = False,
            wandb_run=None,
            wandb_key: Optional[str] = None,
            progress_bar: Optional[Any] = None,
        ) -> None:
            if is_training:
                wandb_run.log({"density_calls": density_state}, step=epoch)
            loss, model_state, density_state = jited_val_loss(
                trainable_params,
                non_trainable_params,
                model_state,
                rng_key,
                batch_size,
                density_state,
                is_training,
                ode,
                exact,
            )
            loss = jax.device_get(loss)
            loss = onp.asarray(utils.get_first(loss).item()).item()

            if progress_bar:
                progress_bar.set_description(f"loss: {loss:.3f}")

            loss_list.append(loss)
            if wandb_run:
                wandb_run.log({f"{wandb_key}/loss": loss}, step=epoch)

            return loss, density_state

        loss_list = []
        loss_list_is = []
        loss_list_pf = []

        start = 0
        density_state_training = jnp.array([0], dtype=jnp.int32)
        times = []
        progress_bar = tqdm.tqdm(list(range(start, config.trainer.epochs)))
        for epoch in progress_bar:
            rng_key = next(seq)
            subkeys = jax.random.split(rng_key, device_no)

            trainable_params, opt_state, model_state, density_state_training = update(
                trainable_params,
                non_trainable_params,
                model_state,
                opt_state,
                subkeys,
                batch_size_,
                density_state_training,
            )

            if config.trainer.timer:

                def func():
                    return jax.block_until_ready(
                        update(
                            trainable_params,
                            non_trainable_params,
                            model_state,
                            opt_state,
                            subkeys,
                            batch_size_,
                            density_state_training,
                        )
                    )

                delta_time = timeit.timeit(func, number=1)
                times.append(delta_time)

            update_detached_params(
                trainable_params, non_trainable_params, "simple_drift_net", "stl_detach"
            )

            if epoch % config.trainer.log_every_n_epochs == 0:
                _, _ = eval_report(
                    trainable_params,
                    non_trainable_params,
                    model_state,
                    subkeys,
                    batch_size_elbo,
                    density_state_training,
                    epoch,
                    loss_list,
                    print_flag=True,
                    wandb_run=run,
                    wandb_key="elbo_results",
                    progress_bar=progress_bar,
                )

                _, _ = eval_report(
                    trainable_params,
                    non_trainable_params,
                    model_state,
                    subkeys,
                    batch_size_elbo,
                    density_state_training,
                    epoch,
                    loss_list_is,
                    is_training=False,
                    wandb_run=run,
                    wandb_key="is_results",
                )

                # _, _ = eval_report(
                #     trainable_params,
                #     non_trainable_params,
                #     model_state,
                #     subkeys,
                #     batch_size_elbo,
                #     0,
                #     epoch,
                #     loss_list_pf,
                #     is_training=False,
                #     ode=True,
                #     wandb_run=run,
                #     wandb_key="pf_results",
                # )

                lr = onp.asarray(exp_lr(epoch).item()).item()
                run.log({"lr/lr": lr}, step=epoch)

        loss_list_is_eval, loss_list_eval, loss_list_pf_eval = [], [], []
        for i in range(config.eval.seeds):
            rng_key = next(seq)
            subkeys = jax.random.split(rng_key, device_no)
            # _, _ = eval_report(
            #     trainable_params,
            #     non_trainable_params,
            #     model_state,
            #     subkeys,
            #     batch_size_elbo,
            #     0,
            #     i,
            #     loss_list_eval,
            #     print_flag=True,
            #     wandb_run=run,
            #     wandb_key="elbo_results_eval",
            # )

            _, sampling_density_calls = eval_report(
                trainable_params,
                non_trainable_params,
                model_state,
                subkeys,
                batch_size_elbo,
                jnp.array([0], dtype=jnp.int32),
                i,
                loss_list_is_eval,
                is_training=False,
                wandb_run=run,
                wandb_key="is_results_eval",
            )

            # _, _ = eval_report(
            #     trainable_params,
            #     non_trainable_params,
            #     model_state,
            #     subkeys,
            #     batch_size_elbo,
            #     0,
            #     i,
            #     loss_list_pf_eval,
            #     is_training=False,
            #     ode=True,
            #     exact=False,
            #     wandb_run=run,
            #     wandb_key="pf_results_eval",
            # )

        # params = hk.data_structures.merge(trainable_params, non_trainable_params)
        # if config.trainer.timer:
        #     print(times[1:])

        # samps = 2500
        # if method == "lgcp" and tfinal >= 12:
        #     samps = 100

        # (augmented_trajectory, _), _ = forward_fn_wrap(
        #     params, model_state, rng_key, samps
        # )

        # (augmented_trajectory_det, _), _ = forward_fn_wrap(
        #     params, model_state, rng_key, samps, True, False
        # )

        # (augmented_trajectory_det_ext, _), _ = forward_fn_wrap(
        #     params, model_state, rng_key, samps, True, True
        # )

        results_dict = {
            "elbo": loss_list,
            "is": loss_list_is,
            "pf": loss_list_pf,
            # "elbo_eval": loss_list_eval,
            "is_eval": loss_list_is_eval,
            # "pf_eval": loss_list_pf_eval,
            # "aug": augmented_trajectory,
            # "aug_ode": augmented_trajectory_det,
            # "aug_ode_ext": augmented_trajectory_det_ext,
        }

        log_z_mean = onp.mean(loss_list_is_eval)
        log_z_var = onp.var(loss_list_is_eval)
        run.log(
            {
                "final_log_Z": log_z_mean,
                "var_final_log_Z": log_z_var,
                "sampling_density_calls": sampling_density_calls,
            },
            step=config.trainer.epochs,
        )

        return params, model_state, forward_fn_wrap, rng_key, results_dict


def main(config):
    config_file = set_task(config, task=_TASK.value)
    logging.info(config_file)
    train_dds(config_file)


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
