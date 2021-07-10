"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module defines a model class that users can extend to implement
arbitrary likelihood models.
"""
from typing import Any, Dict, Iterable, Tuple, Union

import arviz as az
import chex
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jax import numpy as jnp
from jax import random, vmap
from scipy import stats as sst
from tabulate import tabulate

from .cv import CrossValidationScheme, cv_factory
from .hmc import (
    CrossValidationState,
    InfParams,
    WarmupResults,
    cross_validate,
    full_data_inference,
    warmup,
)
from .statistics import ess, split_rhat
from .util import Timer

# model parameters are in a constrained coordinate space
ModelParams = Dict[str, chex.ArrayDevice]

# CV fold is either 1D or 2D integer index
CVFold = Union[int, Tuple[int, int]]

_ARVIZ_PLOT = [name for name in dir(az) if name.startswith("plot_")]
_ARVIZ_OTHER = ["summary", "ess", "loo"]
_ARVIZ_METHODS = _ARVIZ_PLOT + _ARVIZ_OTHER


def _print_devices():
    """Print summary of available devices to console."""
    device_list = [f"{d.device_kind} ({d.platform}{d.id})" for d in jax.devices()]
    if len(device_list) > 0:
        print(f'Detected devices: {", ".join(device_list)}')
    else:
        print("Only CPU is available. Check cuda/cudnn library versions.")


def _to_posterior_dict(post_draws):
    """Construct xarrays for ArviZ

    Converts all objects to in-memory numpy arrays. This involves a lot of copying,
    of course, but ArviZ chokes if given jax.numpy arrays.

    :param post_draws: dict of posterior draws, keyed by parameter name
    :returns: xarray dataset suitable for passing to az.InferenceData
    """
    first_param = next(iter(post_draws))
    chains = post_draws[first_param].shape[0]
    draws = post_draws[first_param].shape[1]
    post_draw_map = {}
    coords = {  # gets updated for
        "chain": (["chain"], np.arange(chains)),
        "draw": (["draw"], np.arange(draws)),
    }
    for var, drws in post_draws.items():
        # dimensions are chain, draw number, variable dim 0, variable dim 1, ...
        extra_dims = [(f"{var}{i}", length) for i, length in enumerate(drws.shape[2:])]
        keys = ["chain", "draw"] + [n for n, len in extra_dims]
        post_draw_map[var] = (keys, np.asarray(drws))
        for dimname, length in extra_dims:
            coords[dimname] = ([dimname], np.arange(length))

    posterior = xr.Dataset(post_draw_map, coords=coords)
    return posterior


class _Posterior(az.InferenceData):
    """ploo posterior: captures full-data and loo results

    This is an ArviZ :class:`az.InferenceData` object, so you can use the full
    range of ArviZ posterior exploration features directly on this object.

    Members:
        model:       Model instance this was created from
        post_draws:  map of posterior draw arrays, keyed by variable, with axes
                     (chain, draw, variable_axis0, ...)
        warmup_res:  results from warmup
        accumulator: accumulator state object from inference routine
        rng_key:     random number generator state
    """

    def __init__(
        self,
        model: "Model",
        post_draws: Dict[str, chex.ArrayDevice],
        warmup_res: WarmupResults,
        accumulator: CrossValidationState,
        rng_key: chex.ArrayDevice,
    ) -> None:
        self.model = model
        self.post_draws = post_draws
        first_param = next(iter(post_draws))
        self.chains = post_draws[first_param].shape[0]
        self.draws = post_draws[first_param].shape[1]
        self.warmup_res = warmup_res
        self.rng_key = rng_key
        self.accumulator = accumulator
        posterior = _to_posterior_dict(self.post_draws)
        super().__init__(posterior=posterior)

    @property
    def chain_divergences(self) -> int:
        """Divergence count by chain"""
        return self.accumulator.divergence_count

    @property
    def total_divergences(self) -> int:
        """Total number of divergences, summed across chains"""
        return jnp.sum(self.accumulator.divergence_count)

    @property
    def avg_acceptance_rate(self) -> float:
        """Average acceptance rate across all chains, as proportion in [0,1]"""
        return float(
            jnp.sum(self.accumulator.accepted_count) / self.chains / self.draws
        )

    @property
    def acceptance_rates(self) -> float:
        """Acceptance rate by chains, as proportion in [0,1]"""
        return self.accumulator.accepted_count / self.draws

    def __str__(self) -> str:
        title = f"{self.model.name} inference summary"
        arg0 = next(iter(self.post_draws))
        chains, iters = self.post_draws[arg0].shape[:2]
        desc_rows = [
            title,
            "=" * len(title),
            "",
            f"{iters*chains:,} draws from {iters:,} iterations on {chains:,} chains",
            f"{self.total_divergences} divergences, "
            f"{self.avg_acceptance_rate*100}% acceptance rate",
        ] + [self._post_table()]
        return "\n".join(desc_rows)

    def _post_table(self) -> str:
        """Construct a summary table for posterior draws"""
        table_headers = [
            "Parameter",
            "Mean",
            "MCSE",
            "1%",
            "5%",
            "25%",
            "Med",
            "75%",
            "95%",
            "99%",
            "R̂ᵇᵘˡᵏ",
            "Sᵉᶠᶠ",
        ]
        table_quantiles = jnp.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

        def table_row(par, draws):
            eff_sample = ess(draws)
            mcse = jnp.std(draws) / jnp.sqrt(eff_sample)
            cols1 = [par, f"{jnp.mean(draws):4.2f}", f"({mcse:.2f})"]
            cols2 = [f"{q:.02f}" for q in jnp.quantile(draws, table_quantiles)]
            cols3 = [split_rhat(draws), round(eff_sample)]
            return cols1 + cols2 + cols3

        table_rows = [table_row(par, draws) for par, draws in self.post_draws.items()]
        return tabulate(table_rows, headers=table_headers)

    def cross_validate(
        self,
        cv_scheme: Union[str, CrossValidationScheme] = "LOO",
        retain_draws=False,
        rng_key: chex.ArrayDevice = None,
        **kwargs,
    ) -> "CrossValidation":
        """Run cross-validation for this posterior.

        Number of chains and draws per chain are the same as the original inference
        procedure. Only enable the `retain_draws` flag if you are sure you have enough
        memory on your GPU. Even moderately-sized problems can exhaust a GPU's memory
        quite quickly.

        Args:
            cv_scheme:    name of cross-validation scheme to apply
            retain_draws: if true, retain MCMC draws
            rng_key:      random generator state
            kwargs:       arguments to pass to cross-validation scheme constructor

        Returns:
            CrossValidation object containing all CV posteriors
        """
        rng_key = rng_key or self.rng_key
        timer = Timer()
        # shape from a likelihood evaluation: wasteful but reduces mistakes
        _, example_ll = self.model.log_prior_likelihood(self.model.initial_value())
        cv_shape = example_ll.shape
        if isinstance(cv_scheme, str):
            cv_scheme = cv_factory(cv_scheme)(shape=cv_shape, **kwargs)
        cv_chains = self.chains * cv_scheme.cv_folds()
        title = f"Brute-force {cv_scheme.name}: {self.model.name}"
        print(title)
        print("=" * len(title))
        print(
            f"Fitting posteriors for {cv_scheme.cv_folds():,} folds "
            f"using {cv_chains:,} chains..."
        )
        masks = cv_scheme.mask_array()
        pred_indexes = cv_scheme.pred_index_array()

        def potential(inf_params: InfParams, cv_fold: int) -> chex.ArrayDevice:
            return self.model.cv_potential(inf_params, masks[cv_fold])

        def log_cond_pred(inf_params: InfParams, cv_fold: int) -> chex.ArrayDevice:
            model_params, _ = self.model.inverse_transform_log_det(inf_params)
            return self.model.log_cond_pred(model_params, pred_indexes[cv_fold])

        accumulator, states = cross_validate(
            potential,
            log_cond_pred,
            self.warmup_res,
            cv_scheme.cv_folds(),
            self.draws,
            self.chains,
            rng_key,
            retain_draws,
        )
        accumulator.divergence_count.block_until_ready()
        divergent_chains = jnp.sum(accumulator.divergence_count > 0)
        if divergent_chains > 0:
            print(f"      WARNING: {divergent_chains} divergent chain(s).")
        print(
            f"      {cv_chains*self.draws:,} HMC draws took {timer}"
            f" ({cv_chains*self.draws/timer.sec:,.0f} iter/sec)."
        )

        if retain_draws:
            # map positions back to model coordinates
            # NB: if we can evaluate objective online, this will not be necessary
            position_model = vmap(self.model.to_model_params)(states)
            # want axes to be (chain, draws, ... <variable dims> ...)
            rearranged_draws = {
                var: jnp.swapaxes(draws, axis1=0, axis2=1)
                for (var, draws) in position_model.items()
            }
        else:
            rearranged_draws = None

        return CrossValidation(self, accumulator, rearranged_draws, scheme=cv_scheme)

    def __getattribute__(self, name: str) -> Any:
        """If the user invokes a plot_* function, delegate to ArviZ."""
        if name in _ARVIZ_METHODS:
            delegate_to = getattr(az, name)

            def plot_function(*args, **kwargs):
                return delegate_to.__call__(self, *args, **kwargs)

            # return copy of ArviZ docstring with some tweaks
            plot_function.__doc__ = (
                (
                    f"**Note: function delegated to ArviZ. See: help(arviz.{name})**"
                    f"\n\n{delegate_to.__doc__}"
                )
                .replace(f"{name}(data:", f"{name}(self:")
                .replace("data: obj", "self:")
            )
            return plot_function
        # not an ArviZ method; use standard attr resolution
        return super().__getattribute__(name)

    def __dir__(self) -> Iterable[str]:
        parent_dir = super().__dir__()
        return parent_dir + _ARVIZ_METHODS


class CrossValidation:  # pylint: disable=too-many-instance-attributes
    """Model cross-validated

    This class contains draws for all the CV posteriors.
    """

    def __init__(
        self,
        post: _Posterior,
        accumulator: CrossValidationState,
        states: Dict[str, chex.ArrayDevice],
        scheme: CrossValidationScheme,
    ) -> None:
        """Create a new CrossValidation instance

        :param post: full-data posterior
        :param accumulator: state accumulator used during inference step
        :param states: MCMC states, as dict keyed by parameter
        :param folds: number of cross-validation folds
        :param fold_indexes: indexes of cross-validation folds
        """
        self.post = post
        self.accumulator = accumulator
        self.states = states
        self.scheme = scheme
        self.elpd = float(jnp.mean(self.accumulator.sum_log_pred_dens / self.draws))
        self.elpd_se = float(
            jnp.std(self.accumulator.sum_log_pred_dens / self.draws)
            / jnp.sqrt(self.draws)
        )

    @property
    def model(self) -> "Model":
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
        return self.scheme.cv_folds()

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
        fold_indexes = jnp.arange(self.scheme.cv_folds())
        chain_folds = jnp.repeat(fold_indexes, self.chains)
        chain_i = chain_folds == cv_fold
        if not jnp.sum(chain_i):
            raise Exception("No chains match CV fold {self.cv_fold}")
        draw_subset = {
            var: np.compress(chain_i, drw, axis=0) for (var, drw) in self.states.items()
        }
        return az.InferenceData(posterior=_to_posterior_dict(draw_subset))

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
        return "\n".join(
            [
                "Cross-validation summary",
                "========================",
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


class Model:
    """A Bayesian model. Encapsulates both data and specification.

    There are two sets of parameters referenced in this class, both of
    which are expressed as dicts keyed by variable name:

      * Model parameters (type annotation ModelParams) are in the
        coordinate system used by the model, which may be constrained
        (e.g. variance parameters might take values on positive half-line)

      * Inference parameters (type annotation InfParams) are in an
        unconstrained real coordinate system, i.e. Θ = ℝᵈ for some d ∈ ℕ.

    Transformations are up to the user, but should probably be done using
    instances of the Transform class.
    """

    name = "Unnamed model"

    def log_prior_likelihood(
        self, model_params: ModelParams
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        """Log likelihood log p(y|θ) and prior log p(θ)

        .. note::
            JAX needs to be able to trace this function. See JAX docs for what that
            means.

        :param model_params: dict of model parameters, in constrained (model)
                             parameter space
        :return: log likelihood at the given parameters
        """
        raise NotImplementedError()

    def log_cond_pred(
        self, model_params: ModelParams, coords: chex.ArrayDevice
    ) -> chex.ArrayDevice:
        """Computes log conditional predictive ordinate, log p(yᵢ|θˢ).

        :param model_params: a dict of parameters (constrained (model) parameter
            space) that potentially contains vectors
        :param coords: points at which to evaluate predictive density, expressed
            as coordinates that correspond to the shape of the log likelihood
            contributions array
        """
        raise NotImplementedError()

    def initial_value(self) -> ModelParams:
        """A deterministic starting value in model parameter space."""
        raise NotImplementedError()

    def initial_value_unconstrained(self) -> InfParams:
        """Deterministic starting value, transformed to unconstrained inference space"""
        inf_params, _ = self.inverse_transform_log_det(self.initial_value())
        return inf_params

    def cv_potential(
        self, inf_params: InfParams, likelihood_mask: chex.ArrayDevice
    ) -> chex.ArrayDevice:
        """Potential for a CV fold.

        :param inf_params: model parameters in inference (unconstrained) space
        :param likelihood_mask: mask to apply to likelihood contributions by
            elementwise multiplication
        :return: potential of model, adjusted for CV fold
        """
        model_params, ldet = self.inverse_transform_log_det(inf_params)
        lprior, llik = self.log_prior_likelihood(model_params=model_params)
        llik_subset = jnp.sum(llik * likelihood_mask)
        return -(llik_subset + lprior + ldet)

    def parameters(self):
        """Names of parameters"""
        return list(self.initial_value().keys())

    # pylint: disable=no-self-use
    def forward_transform(self, model_params: ModelParams) -> InfParams:
        """Convert constrained (model) params to unconstrained (sampler) space

        The argument model_params is expressed in constrained (model) coordinate
        space.

        :param model_params: dictionary of parameters in constrained (model)
            parameter space, keyed by name
        :return: dictionary of parameters with same structure as params, but in
            unconstrained (sampler) parameter space.
        """
        # by default just do identity transform
        return model_params

    def inverse_transform_log_det(
        self, inf_params: InfParams
    ) -> Tuple[ModelParams, chex.ArrayDevice]:
        """Map unconstrained to constrained parameters, with log determinant

        Maps unconstrained (inference) params in `inf_params` to corresponding
        parameters in constrained (model) parameter space. We do both at once so it's
        harder to forget to implement the log determinant.

        :param inf_params: dictionary of parameters in unconstrained (sampler)
            parameter space, keyed by name
        :return: Tuple of model parameter dict and log determinant. The parameter dict
            has the same structure as inf_params, but in constrained (model) parameter
            space.
        """
        # by default just do identity transform
        return inf_params, jnp.array(0.0)

    def inference(
        self,
        draws: int = 2_000,
        warmup_steps: int = 500,
        chains: int = 8,
        seed: int = 42,
        warmup_results: WarmupResults = None,
    ) -> _Posterior:
        """Run HMC with full dataset, tuned by Stan+NUTS warmup

        :param draws: number of draws per chain
        :param warmup_steps: number of Stan warmup steps to run
        :param chains: number of chains for main inference step
        :param seed: random seed
        :parm warmup_results: use this instead of running warmup
        :return: posterior object
        """
        rng_key = random.PRNGKey(seed)
        warmup_key, inference_key, post_key = random.split(rng_key, 3)
        draws, warmup_steps = int(draws), int(warmup_steps)
        title = f"Full-data posterior inference: {self.name}"
        print(title)
        print("=" * len(title))
        _print_devices()
        if warmup_results:
            print("Skipping warmup")
        else:
            print("Starting Stan warmup using NUTS...")
            timer = Timer()

            def warmup_potential(params):
                return self.cv_potential(params, jnp.array(1.0))

            warmup_results = warmup(
                warmup_potential,
                self.initial_value(),
                warmup_steps,
                chains,
                warmup_key,
            )
            print(
                f"      {warmup_steps:,} warmup draws took {timer}"
                f" ({warmup_steps/timer.sec:.1f} iter/sec)."
            )
            print(
                f"      Step size {warmup_results.step_size:.4f}, "
                f"integration length {warmup_results.int_steps} steps."
            )

        print(
            f"HMC for {draws*chains:,} full-data draws "
            f"({chains} chains, {draws:,} draws per chain)..."
        )
        timer = Timer()

        def inference_potential(params, _dummy_fold):
            return self.cv_potential(params, jnp.array(1.0))

        accum, states = full_data_inference(
            inference_potential, warmup_results, draws, chains, inference_key
        )
        divergent_chains = jnp.sum(accum.divergence_count > 0)
        if divergent_chains > 0:
            print(f"      WARNING: {divergent_chains} divergent chain(s).")
        accept_rate_pc = float(100 * jnp.mean(accum.accepted_count) / draws)
        print(f"      Average HMC acceptance rate {accept_rate_pc:.1f}%.")

        # map positions back to model coordinates (drop log determinant)
        def inverse_tfm(params):
            mparam, _ = self.inverse_transform_log_det(params)
            return mparam

        position_model = vmap(inverse_tfm)(states.position)
        # want axes to be (chain, draws, ... <variable dims> ...)
        rearranged_positions = {
            var: jnp.swapaxes(draws, axis1=0, axis2=1)
            for (var, draws) in position_model.items()
        }
        print(
            f"      {chains*draws:,} HMC draws took {timer}"
            f" ({chains*draws/timer.sec:,.0f} iter/sec)."
        )

        return _Posterior(
            self,
            post_draws=rearranged_positions,
            warmup_res=warmup_results,
            accumulator=accum,
            rng_key=post_key,
        )
