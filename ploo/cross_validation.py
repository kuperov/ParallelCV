"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module defines a model class that users can extend to implement
arbitrary likelihood models.
"""
from typing import Tuple, Union

import arviz as az
import chex
import matplotlib.pyplot as plt
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import random, vmap
from scipy import stats as sst

from .hmc import CrossValidationState, CVHMCState, InfParams, cv_kernel, new_cv_state
from .schemes import CrossValidationScheme, cv_factory
from .util import Timer, to_posterior_dict


class CrossValidation:  # pylint: disable=too-many-instance-attributes
    """Model cross-validated

    This class contains summary statistics, and optionally draws, for all CV
    posteriors.

    :param post: full-data posterior
    :param scheme: cross-validation scheme. Can be either a class name
        or an instance of CrossValidationScheme
    :param retain_draws: if True, keep full traces for all MCMC draws
    :param rng_key: random number generator state
    :param kwargs: arguments to pass to scheme constructor
    """

    def __init__(
        self,
        post,  # : _Posterior
        scheme: Union[CrossValidationScheme, str],
        retain_draws: bool,
        rng_key: chex.ArrayDevice = None,
        **kwargs,
    ) -> None:
        rng_key = rng_key or post.rng_key
        self.post = post
        timer = Timer()
        _, example_ll = self.model.log_prior_likelihood(self.model.initial_value())
        cv_shape = example_ll.shape
        if isinstance(scheme, str):
            scheme = cv_factory(scheme)(shape=cv_shape, **kwargs)
        self.scheme = scheme
        cv_chains = self.chains * scheme.folds
        title = f"Brute-force {scheme.name}: {self.model.name}"
        print(title)
        print("=" * len(title))
        print(
            f"Fitting posteriors for {scheme.folds:,} folds "
            f"using {cv_chains:,} chains..."
        )
        masks = scheme.mask_array()
        fold_preds = scheme.pred_indexes()

        def potential(inf_params: InfParams, cv_fold: int) -> chex.ArrayDevice:
            return self.model.cv_potential(inf_params, masks[cv_fold])

        def log_cond_pred(
            inf_params: InfParams, coords: chex.ArrayDevice, mask: chex.ArrayDevice
        ) -> chex.ArrayDevice:
            model_params, _ = self.model.inverse_transform_log_det(inf_params)
            log_lik = self.model.log_cond_pred(model_params, coords)  # covariates?
            return jnp.mean(log_lik * mask)

        fold_initial_state = vmap(new_cv_state, (0, None, None))  # map over chains
        cv_initial_states = vmap(fold_initial_state, (None, None, 0))(  # map over folds
            post.warmup_res.starting_values, potential, jnp.arange(scheme.folds)
        )

        cv_initial_accumulator = CrossValidationState(
            divergence_count=jnp.zeros((scheme.folds, self.chains)),
            accepted_count=jnp.zeros((scheme.folds, self.chains)),
            sum_log_pred_dens=jnp.zeros((scheme.folds, self.chains)),
            hmc_state=cv_initial_states,
        )
        kernel = cv_kernel(
            potential,
            post.warmup_res.step_size,
            post.warmup_res.mass_matrix,
            post.warmup_res.int_steps,
        )

        def do_cv():
            # each step operates vector of states (representing a cross-section
            # across chains) and vector of rng keys, one per draw
            def one_step(
                cv_state: CrossValidationState, rng_subkey: chex.ArrayDevice
            ) -> Tuple[CrossValidationState, CVHMCState]:
                # only need as many random keys as chains; different folds don't
                # need different random keys
                rng_keys = random.split(rng_subkey, self.chains)
                # markov kernel function for a single chain number, across all folds
                kernel_c = vmap(kernel, in_axes=[None, 0])
                # markov kernel function for all chain-folds
                kernel_cf = vmap(kernel_c, in_axes=[0, 1])
                hmc_state, hmc_info = kernel_cf(rng_keys, cv_state.hmc_state)
                # conditional predictive function for a single chain number, all folds
                pred_c = vmap(log_cond_pred, in_axes=[0, 0, 0])
                # conditional predictive for all chain-folds
                pred_cf = vmap(pred_c, in_axes=(1, None, None))
                pred = pred_cf(hmc_state.position, fold_preds.coords, fold_preds.masks)
                # update online estimators
                div_count = cv_state.divergence_count + 1.0 * hmc_info.is_divergent
                accept_count = cv_state.accepted_count + 1.0 * hmc_info.is_accepted
                log_pred_cumsum = cv_state.sum_log_pred_dens + pred
                # state to pass to next iteration
                updated_state = CrossValidationState(
                    divergence_count=div_count,
                    accepted_count=accept_count,
                    sum_log_pred_dens=log_pred_cumsum,
                    hmc_state=hmc_state,
                )
                return updated_state, hmc_state.position

            # this version of one_step used if retain_draws==False
            def one_step_nodraws(
                cv_state: CrossValidationState, rng_subkey: chex.ArrayDevice
            ) -> Tuple[CrossValidationState, CVHMCState]:
                updated_state, _ = one_step(cv_state=cv_state, rng_subkey=rng_subkey)
                return updated_state, None

            step_function = one_step if retain_draws else one_step_nodraws

            draw_keys = random.split(rng_key, self.draws)
            accumulator, positions = lax.scan(
                step_function, cv_initial_accumulator, draw_keys
            )
            return accumulator, positions

        j_do_cv = do_cv  # jax.jit(do_cv)
        accumulator, states = j_do_cv()

        accumulator.divergence_count.block_until_ready()  # for accurate iter rate
        divergent_chains = jnp.sum(accumulator.divergence_count > 0)
        if divergent_chains > 0:
            print(f"      WARNING: {divergent_chains} divergent chain(s).")
        print(
            f"      {cv_chains*self.draws:,} HMC draws took {timer}"
            f" ({cv_chains*self.draws/timer.sec:,.0f} iter/sec)."
        )

        def inverse_transform(param):
            model_param, _ = self.model.inverse_transform_log_det(param)
            return model_param

        if retain_draws:
            # map positions back to model coordinates
            # NB: if we can evaluate objective online, this will not be necessary
            position_model = vmap(inverse_transform)(states)
            # want axes to be (chain, draws, ... <variable dims> ...)
            rearranged_draws = {
                var: jnp.swapaxes(draws, axis1=0, axis2=2)
                for (var, draws) in position_model.items()
            }
        else:
            rearranged_draws = None

        self.accumulator = accumulator
        self.states = rearranged_draws
        self.fold_elpds = (
            np.mean(self.accumulator.sum_log_pred_dens, axis=1) / self.draws
        )

    @property
    def elpd(self):
        """Mean elpd across all folds"""
        return float(jnp.mean(self.accumulator.sum_log_pred_dens / self.draws))

    @property
    def elpd_se(self):
        """s.e. of elpd estimates - makes flawed assumption of fold independence"""
        return float(jnp.std(self.fold_elpds) / jnp.sqrt(self.folds))

    @property
    def model(self):
        """Model being cross-validated"""
        return self.post.model

    @property
    def draws(self) -> int:
        """Number of draws per chain"""
        return self.post.draws

    @property
    def chains(self) -> int:
        """Number of chains per CV fold"""
        return self.post.chains

    @property
    def folds(self) -> int:
        """Number of CV folds in this CV scheme"""
        return self.scheme.folds

    def __lt__(self, cv):
        return self.elpd.__gt__(cv.elpd)  # note change of sign, want largest first

    def arviz(self, cv_fold: int) -> az.InferenceData:
        """Retrieves ArviZ :class:`az.InferenceData` object for a CV fold

        :param cv_fold: index of CV fold corresponding to desired posterior
        :raise Exception: if draws were not retained, we can't analyze them
        :return: an ArviZ object
        :rtype: arviz.InferenceData
        """
        if not self.states:
            raise Exception("States not retained. Cannot construct ArviZ object.")
        fold_indexes = jnp.arange(self.scheme.folds)
        chain_folds = jnp.repeat(fold_indexes, self.chains)
        chain_i = chain_folds == cv_fold
        if not jnp.sum(chain_i):
            raise Exception("No chains match CV fold {self.cv_fold}")
        draw_subset = {
            var: np.compress(chain_i, drw, axis=0) for (var, drw) in self.states.items()
        }
        return az.InferenceData(posterior=to_posterior_dict(draw_subset))

    @property
    def divergences(self):
        """Divergence count for each chain"""
        return self.accumulator.divergence_count

    @property
    def num_divergent_chains(self):
        """Total number of chains with nonzero divergences"""
        return int(jnp.sum(self.accumulator.divergence_count > 0))

    @property
    def acceptance_rates(self):
        """Acceptance rate for all chains, as fraction of 1"""
        return self.accumulator.accepted_count / self.draws

    def __repr__(self) -> str:
        avg_accept = float(jnp.mean(self.acceptance_rates))
        min_accept = float(jnp.min(self.acceptance_rates))
        max_accept = float(jnp.max(self.acceptance_rates))
        title = f"{self.scheme} Summary: {self.model.name}"
        return "\n".join(
            [
                title,
                "=" * len(title),
                "",
                f"    elpd = {self.elpd:.4f} (se {self.elpd_se:.4f})",
                "",
                f"Calculated from {self.folds:,} folds "
                f"({self.chains:,} chains per fold, "
                f"{self.chains*self.folds:,} total)",
                "",
                f"Average acceptance rate {avg_accept*100:.1f}% "
                f"(min {min_accept*100:.1f}%, max {max_accept*100:.1f}%)",
                "",
                f"Divergent chain count: {self.num_divergent_chains:,}",
            ]
        )

    def block_until_ready(self) -> None:
        """Block the thread until results are back from the GPU."""
        self.accumulator.divergence_count.block_until_ready()

    def densities(self, par, combine=False, ncols=4, figsize=(40, 80)):
        """Small-multiple kernel densities for cross-validation posteriors."""
        rows = int(jnp.ceil(self.model.cv_folds() / ncols))
        _, axes = plt.subplots(nrows=rows, ncols=ncols, figsize=figsize)
        for fold, axis in zip(range(self.model.cv_folds()), axes.ravel()):
            chain_indexes = jnp.arange(fold * self.chains, (fold + 1) * self.chains)
            all_draws = self.states[par][chain_indexes, :]
            if combine:
                all_draws = jnp.expand_dims(jnp.reshape(all_draws, (-1,)), axis=1)
            x_coords = jnp.linspace(jnp.min(all_draws), jnp.max(all_draws), 1_000)
            for i in range(self.chains):
                draws = all_draws[i, :]
                try:
                    axis.plot(x_coords, sst.gaussian_kde(draws)(x_coords))
                except np.linalg.LinAlgError:
                    print(f"Error evaluating kde for fold {fold}, chain {i}")

    def trace_plots(self, par, ncols=4, figsize=(40, 80)) -> None:
        """Plot trace plots for every single cross validation fold."""
        rows = int(jnp.ceil(self.model.cv_folds() / ncols))
        _, axes = plt.subplots(nrows=rows, ncols=ncols, figsize=figsize)
        for fold, axis in zip(range(self.model.cv_folds()), axes.ravel()):
            chain_indexes = jnp.arange(fold * self.chains, (fold + 1) * self.chains)
            axis.plot(self.states[par][chain_indexes, :].T)
