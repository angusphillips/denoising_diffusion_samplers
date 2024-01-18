import jax.numpy as jnp
import jax
import optax
import tqdm
import equinox as eqx

import typing as tp
from hydra.utils import instantiate
from jaxtyping import PyTree, Array, PRNGKeyArray

Key = PRNGKeyArray

import haiku as hk

from dds.targets.distributions import Distribution, MeanFieldNormalDistribution


class TrainingState(eqx.Module):
    params: PyTree
    params_ema: PyTree
    opt_state: optax.OptState
    key: Array
    step: Array


# evaluate_log_density state is ignored during variational inference training


class VariationalLogDensity(hk.Module):
    def __init__(self, dim, dtype, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.dtype = dtype

    def __call__(self, x):
        means = hk.get_parameter(
            "means", shape=[self.dim], dtype=self.dtype, init=jnp.zeros
        )
        scales = hk.get_parameter(
            "scales", shape=[self.dim], dtype=self.dtype, init=jnp.ones
        )
        mean_field_dist = MeanFieldNormalDistribution(means, scales, self.dim)
        return mean_field_dist.evaluate_log_density(x, 0)[0]


class VariationalSampler(hk.Module):
    def __init__(self, dim, dtype, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.dtype = dtype

    def __call__(self, rng: Key, num_particles: int):
        means = hk.get_parameter(
            "means", shape=[self.dim], dtype=self.dtype, init=jnp.zeros
        )
        scales = hk.get_parameter(
            "scales", shape=[self.dim], dtype=self.dtype, init=jnp.ones
        )
        mean_field_dist = MeanFieldNormalDistribution(means, scales, self.dim)
        return mean_field_dist.sample(rng, num_particles)


def get_variational_approx(
    dim, lr_schedule, batch_size, iters, rng, target_distribution: Distribution
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    def variational_log_density(x):
        vld = VariationalLogDensity(dim=dim, dtype=jnp.float32, name="Variational")
        return vld(x)

    def variational_sampler(sampler_rng: Key, num_particles: int):
        v_sampler = VariationalSampler(dim=dim, dtype=jnp.float32, name="Variational")
        return v_sampler(sampler_rng, num_particles)

    var_log_density = hk.without_apply_rng(hk.transform(variational_log_density))
    var_sampler = hk.without_apply_rng(hk.transform(variational_sampler))

    def loss_fn(params, rng, n_particles=1000):
        X = var_sampler.apply(params=params, sampler_rng=rng, num_particles=n_particles)
        diff_log_pdf = (
            var_log_density.apply(params, X)
            - target_distribution.evaluate_log_density(X, 0)[0]
        )
        loss = jnp.mean(diff_log_pdf, axis=0)
        return loss

    def init(samples, key: Key) -> TrainingState:
        initial_params = var_log_density.init(None, samples)
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )

    @jax.jit
    def update_step(state: TrainingState) -> tp.Tuple[TrainingState, tp.Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss_value, grads = loss_and_grad_fn(state.params, loss_key, 1000)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * 0.999 + p * (1.0 - 0.999),
            state.params_ema,
            new_params,
        )
        new_state = TrainingState(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1,
        )
        metrics = {"loss": loss_value, "step": state.step}
        return new_state, metrics

    samples = jnp.ones((batch_size, dim))
    state = init(samples=samples, key=rng)

    progress_bar = tqdm.tqdm(
        list(range(1, iters + 1)), miniters=1, disable=False  # (not cfg.progress_bars),
    )
    for step in progress_bar:
        state, metrics = update_step(state)
        if jnp.isnan(metrics["loss"]).any():
            print("loss is nan")
            break
        metrics["lr"] = lr_schedule(step)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")

    return state.params
