"""Different stoch control objectives satisfying same HJB fixed points.
"""
import distrax
from jax import numpy as np
from jax import scipy as jscipy


def ou_terminal_loss(x_terminal, lnpi, sigma=1.0, tfinal=1.0, brown=False):
    """Terminal loss under OU reference prior at equilibrium.

    Can also be used for Brownian if you let sigma be the diff coef.

    Args:
        x_terminal: final time step samples from SDE
        lnpi: log target dist numerator
        sigma: stationary dist for OU dXt = -a* Xt * dt + sqrt(2a)*sigma*dW_t or
               diffusion coeficient for pinned brownian prior
        tfinal: terminal time value
        brown: flag for brownian reference process

    Returns:
        -(lnπ(X_T) - ln N(X_T; 0, sigma))
    """

    _, d = x_terminal.shape
    ln_target = lnpi(x_terminal)

    if brown:
        sigma = np.sqrt(tfinal) * sigma

    equi_normal = distrax.MultivariateNormalDiag(
        np.zeros(d), sigma * np.ones(d)
    )  # equilibrium distribution

    log_ou_equilibrium = equi_normal.log_prob(x_terminal)
    lrnd = -(ln_target - log_ou_equilibrium)

    return lrnd


def relative_kl_objective(augmented_trajectory, g, stl=False, trim=2, dim=2):
    """Vanilla relative KL control objective.

    Args:
        augmented_trajectory: X_{1:T} samples with ||u_t||^2/gamma as dim d+1
        g: terminal cost function typically - ln dπ/dp_1
        stl: boolean marking stl estimator usage
        trim: size of the augmented state space

    Returns:
        kl control loss
    """  # ANGUS this is the objective equation (10) from the paper.

    energy_cost_dt = augmented_trajectory[:, -1, -1]
    x_final_time = augmented_trajectory[:, -1, :dim]

    # import pdb; pdb.set_trace()

    stl = (
        augmented_trajectory[:, -1, dim] if stl else 0
    )  # ANGUS when should we use stl term?

    terminal_cost = g(x_final_time)
    return (energy_cost_dt + terminal_cost + stl).mean()


def prob_flow_lnz(  # ANGUS Probability flow estimate of log normalising constant, called pf results
    augmented_trajectory, eq_dist, target_dist, _=False, debug=False
):
    """Vanilla relative KL control objective.

    Args:
        augmented_trajectory: X_{1:T} samples with trace as final dim
        eq_dist: equilibriium distribution log prob
        target_dist: log target distribution (up to Z)

    Returns:
        kl control loss
    """

    trim = 2

    trace = augmented_trajectory[:, -1, -1]
    x_init_time = augmented_trajectory[:, 0, :-trim]
    x_final_time = augmented_trajectory[:, -1, :-trim]

    ln_gamma = target_dist(x_final_time)
    lnq_0 = eq_dist(x_init_time)
    lnq = lnq_0 - trace  # Instantaneous change of variables formula
    lns = ln_gamma - lnq

    ln_numsamp = np.log(lns.shape[0])
    lnz = jscipy.special.logsumexp(lns, axis=0) - ln_numsamp

    if debug:
        import pdb

        pdb.set_trace()
    return -lnz


def dds_kl_objective(augmented_trajectory, *_, **__):
    """DEPRECATED DO NOT USE.

    Mostly serves as a placeholder as this is computed in the solver.

    Args:
        augmented_trajectory: tuple with trajectory and loss
        *_: empty placeholder
        **__: empty placeholder

    Returns:
        kl control loss
    """
    (_, loss) = augmented_trajectory
    return (loss).mean()


def importance_weighted_partition_estimate(
    augmented_trajectory, g, dim=2
):  # ANGUS this is what's called 'is' results and what should be reported as the lnZ estimate
    """See TODO.

    Args:
        augmented_trajectory: X_{1:T} samples with ||u_t||^2/gamma as dim d+1, shape [batch_size, T, dim+(2)]
        g: terminal cost function typically - ln dπ/dp_1

    Returns:
        smoothed crosent control loss
    """

    energy_cost_dt = augmented_trajectory[:, -1, -1]

    x_final_time = augmented_trajectory[:, -1, :dim]

    stl = augmented_trajectory[:, -1, dim]

    terminal_cost = g(x_final_time)
    s_omega = -(energy_cost_dt + terminal_cost + stl)  # Equation (19)

    ln_numsamp = np.log(s_omega.shape[0])
    lnz = jscipy.special.logsumexp(s_omega, axis=0) - ln_numsamp  # ANGUS averaging
    return -lnz  # ANGUS why the extra - here?


def importance_weighted_partition_estimate_dds(augmented_trajectory, _):
    """Logsumexp IS estimator for dds.

    Args:
        augmented_trajectory:  tuple with trajectory and dds loss

    Returns:
        smoothed crosent control loss
    """

    _, loss = augmented_trajectory

    ln_numsamp = np.log(loss.shape[0])
    lnz = jscipy.special.logsumexp(-loss, axis=0) - ln_numsamp
    return -lnz


def controlled_ais_importance_weighted_partition_estimate_dds(
    augmented_trajectory, g, source=None, target=None, dim=2, *_, **__
):
    """Logsumexp IS estimator for dds.

    Args:
        augmented_trajectory:  tuple with trajectory and dds loss

    Returns:
        smoothed crosent control loss
    """

    l_cost_term = augmented_trajectory[:, -1, dim]
    z_cost_term = augmented_trajectory[:, -1, dim + 1]

    x_final_time = augmented_trajectory[:, -1, :dim]
    x_initial_time = augmented_trajectory[:, 0, :dim]

    terminal_cost = target(x_final_time)
    source_cost = source(x_initial_time)
    loss = l_cost_term - z_cost_term - terminal_cost + source_cost

    ln_numsamp = np.log(loss.shape[0])
    lnz = jscipy.special.logsumexp(-loss, axis=0) - ln_numsamp
    return -lnz


def controlled_ais_relative_kl_objective(
    augmented_trajectory,
    g,
    source=None,
    target=None,
    stl=False,
    trim=2,
    dim=2,
    *_,
    **__
):
    """Vanilla relative KL control objective.

    Args:
        augmented_trajectory: X_{1:T} samples with ||u_t||^2/gamma as dim d+1
        g: terminal cost function typically - ln dπ/dp_1
        stl: boolean marking stl estimator usage
        trim: size of the augmented state space

    Returns:
        kl control loss
    """

    l_cost_term = augmented_trajectory[:, -1, dim]
    z_cost_term = augmented_trajectory[:, -1, dim + 1]

    x_final_time = augmented_trajectory[:, -1, :dim]
    x_initial_time = augmented_trajectory[:, 0, :dim]

    source_cost = source(x_initial_time)

    terminal_cost = target(x_final_time)
    #   return (augmented_trajectory[:, 1, :dim][0] ).mean()
    return (l_cost_term - z_cost_term - terminal_cost + source_cost).mean()
