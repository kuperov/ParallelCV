"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module defines a model class that users can extend to implement
arbitrary likelihood models.
"""
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import arviz as az
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
from .util import Progress, Timer

# model parameters are in a constrained coordinate space
ModelParams = Dict[str, jnp.DeviceArray]

# CV fold is either 1D or 2D integer index
CVFold = Union[int, Tuple[int, int]]

_ARVIZ_PLOT = [name for name in dir(az) if name.startswith("plot_")]
_ARVIZ_OTHER = ["summary", "ess", "loo"]
_ARVIZ_METHODS = _ARVIZ_PLOT + _ARVIZ_OTHER


class _Posterior(az.InferenceData):
    """ploo posterior: captures full-data and loo results

    This is an ArviZ :class:`az.InferenceData` object, so you can use the full
    range of ArviZ posterior exploration features directly on this object.

    Members:
        model:      Model instance this was created from
        post_draws: map of posterior draw arrays, keyed by variable, with axes
                    (chain, draw, variable_axis0, ...)
        cv_draws:   cross-validation draws
        seed:       seed used when invoking inference
        chains:     number of chains per CV posterior
        warmup_res: results from warmup
        rng_key:    random number generator state
        write:      function for writing output to the console
    """

    def __init__(
        self,
        model: "Model",
        post_draws: Dict[str, jnp.DeviceArray],
        seed: int,
        chains: int,
        draws: int,
        warmup_res: WarmupResults,
        rng_key: jnp.DeviceArray,
        write: Callable,
    ) -> None:
        self.model = model
        self.post_draws = post_draws
        self.seed = seed
        self.chains = chains
        self.draws = draws
        self.warmup_res = warmup_res
        self.rng_key = rng_key
        self.write = write
        # construct xarrays for ArviZ
        # FIXME: this incorrectly assumes univariate parameters
        posterior = xr.Dataset(
            {var: (["chain", "draw"], drws) for var, drws in self.post_draws.items()},
            coords={
                "chain": (["chain"], jnp.arange(self.chains)),
                "draw": (["draw"], jnp.arange(self.draws)),
            },
        )
        super().__init__(posterior=posterior)

    def __str__(self) -> str:
        title = f"{self.model.name} inference summary"
        arg0 = next(iter(self.post_draws))
        chains, iters = self.post_draws[arg0].shape[:2]
        desc_rows = [
            title,
            "=" * len(title),
            "",
            f"{iters*chains:,} draws from {iters:,} iterations on {chains:,} "
            f"chains with seed {self.seed}",
            "",
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
        cv_type: Union[str, CrossValidationScheme] = "LOO",
        rng_key: jnp.DeviceArray = None,
        **kwargs,
    ) -> "CrossValidation":
        """Run cross-validation for this posterior.

        Number of chains and draws per chain are the same as the original inference
        procedure.

        Keyword arguments:
            cv_scheme: name of cross-validation scheme to apply
            rng_key:   random generator state
            kwargs:    arguments to pass to cross-validation scheme constructor

        Returns:
            CrossValidation object containing all CV posteriors
        """
        rng_key = rng_key or self.rng_key
        timer = Timer()
        # shape from a likelihood evaluation: wasteful but prevents mistakes
        cv_shape = self.model.log_likelihood(self.model.initial_value()).shape
        cv_scheme = cv_factory(cv_type)(shape=cv_shape, **kwargs)
        cv_chains = self.chains * cv_scheme.cv_folds()
        self.write(
            f"Cross-validation with {cv_scheme.cv_folds():,} folds "
            f"using {cv_chains:,} chains..."
        )
        masks = cv_scheme.mask_array()
        pred_indexes = cv_scheme.pred_index_array()

        def potential(inf_params: InfParams, cv_fold: int) -> jnp.DeviceArray:
            return self.model.cv_potential(inf_params, masks[cv_fold])

        def log_cond_pred(inf_params: InfParams, cv_fold: int) -> jnp.DeviceArray:
            model_params = self.model.to_model_params(inf_params)
            return self.model.log_cond_pred(model_params, pred_indexes[cv_fold])

        accumulator, states = cross_validate(
            potential,
            log_cond_pred,
            self.warmup_res,
            cv_scheme.cv_folds(),
            self.draws,
            self.chains,
            rng_key,
        )
        self.write(
            f"      {cv_chains*self.draws:,} HMC draws took {timer}"
            f" ({cv_chains*self.draws/timer.sec:,.0f} iter/sec)."
        )

        # map positions back to model coordinates
        # NB: if we can evaluate objective online, this will not be necessary
        position_model = vmap(self.model.to_model_params)(states.position)
        # want axes to be (chain, draws, ... <variable dims> ...)
        rearranged_draws = {
            var: jnp.swapaxes(draws, axis1=0, axis2=1)
            for (var, draws) in position_model.items()
        }

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
        states: Dict[str, jnp.DeviceArray],
        scheme: CrossValidationScheme,
    ) -> None:
        """Create a new CrossValidation instance

        Keyword arguments:
            post:         full-data posterior
            accumulator:  state accumulator used during inference step
            states:       MCMC states, as dict keyed by parameter
            folds:        number of cross-validation folds
            fold_indexes: indexes of cross-validation folds
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
        return self.post.draws

    @property
    def chains(self) -> int:
        return self.post.chains

    def __lt__(self, cv):
        return self.elpd.__gt__(cv.elpd)  # note change of sign, want largest first

    def arviz(self, cv_fold):
        """Retrieves ArviZ :class:`az.InferenceData` object for a CV fold

        Keyword arguments
            cv_fold: index of CV fold corresponding to desired posterior
        """
        fold_indexes = jnp.arange(self.scheme.cv_folds())
        chain_folds = jnp.repeat(fold_indexes, self.chains)
        chain_i = chain_folds == cv_fold
        if not jnp.sum(chain_i):
            raise Exception("No chains match CV fold {self.cv_fold}")
        posterior = xr.Dataset(
            {
                var: (["chain", "draw"], jnp.compress(chain_i, drws, axis=0))
                for var, drws in self.states.items()
            },
            coords={
                "chain": (["chain"], jnp.arange(self.chains)),
                "draw": (["draw"], jnp.arange(self.draws)),
            },
        )
        return az.InferenceData(posterior=posterior)

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

    def log_likelihood(self, model_params: ModelParams) -> jnp.DeviceArray:
        """Log likelihood

        JAX needs to be able to trace this function.

        Note: future versions of this function won't take cv_fold as a
        parameter. But we haven't yet built the CV abstraction. Future
        version will return log likelihood contributions contributions
        { log p(yᵢ|θ): i=1,2,⋯,n }, as a 1- or 2-dimensional array.
        The shape of the array corresponds to the shape of the model's
        dependence structure, or a 1-dimensional array if data are iid.

        Keyword args:
            params:  dict of model parameters, in constrained (model)
                     parameter space

        Returns:
            log likelihood at the given parameters for the given CV fold
        """
        raise NotImplementedError()

    def log_prior(self, model_params: ModelParams) -> jnp.DeviceArray:
        """Compute log prior log p(θ)

        JAX needs to be able to trace this function.

        Keyword args:
            params: dict of model parameters in constrained (model)
                    parameter space
        """
        raise NotImplementedError()

    def log_cond_pred(
        self, model_params: ModelParams, coords: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Computes log conditional predictive ordinate, log p(ỹ|θˢ).

        FIXME: needs some kind of index to identify the conditioning values

        Keyword arguments:
            params:  a dict of parameters (constrained (model) parameter
                     space) that potentially contains vectors
            coords:  points at which to evaluate predictive density, expressed
                     as coordinates that correspond to the shape of the log
                     likelihood contributions
        """
        raise NotImplementedError()

    def initial_value(self) -> ModelParams:
        """A deterministic starting value in model parameter space."""
        raise NotImplementedError()

    def initial_value_unconstrained(self) -> InfParams:
        """Deterministic starting value, transformed to unconstrained inference space"""
        return self.to_inference_params(self.initial_value())

    def cv_potential(
        self, inf_params: InfParams, likelihood_mask: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Potential for a CV fold.

        Keyword arguments:
            inf_params: model parameters in inference (unconstrained) space
            lhood_mask: mask to apply to likelihood contributions by elementwise
                        multiplication

        Returns
            joint potential of model, adjusted for CV fold
        """
        model_params = self.to_model_params(inf_params)
        llik = self.log_likelihood(model_params=model_params) * likelihood_mask
        lprior = self.log_prior(model_params=model_params)
        ldet = self.log_det(model_params=model_params)
        return -jnp.sum(llik) - lprior - ldet

    def potential(self, inf_params: InfParams, cv_fold: int = -1) -> jnp.DeviceArray:
        """Potential for a CV fold.

        Keyword arguments:
            inf_params: model parameters in inference (unconstrained) space
            cv_fold:    NOT USED - dummy parameter so we can reuse HMC routines

        Returns
            joint potential of model, adjusted for CV fold
        """
        model_params = self.to_model_params(inf_params)
        llik = self.log_likelihood(model_params=model_params)
        lprior = self.log_prior(model_params=model_params)
        ldet = self.log_det(model_params=model_params)
        return -jnp.sum(llik) - lprior - ldet

    def parameters(self):
        """Names of parameters"""
        return list(self.initial_value().keys())

    @classmethod
    def generate(
        cls, random_key: jnp.DeviceArray, model_params: ModelParams
    ) -> jnp.DeviceArray:
        """Generate a dataset corresponding to the specified random key."""
        raise NotImplementedError()

    # pylint: disable=no-self-use
    def to_inference_params(self, model_params: ModelParams) -> InfParams:
        """Convert constrained (model) params to unconstrained (sampler) space

        The argument model_params is expressed in constrained (model) coordinate
        space.

        Keyword arguments:
            model_params: dictionary of parameters in constrained (model)
                          parameter space, keyed by name

        Returns:
            dictionary of parameters with same structure as params, but
            in unconstrained (sampler) parameter space.
        """
        return model_params

    def to_model_params(self, inf_params: InfParams) -> ModelParams:
        """Convert unconstrained to constrained parameters space

        Maps unconstrained (inference) params in `inf_params` to corresponding
        parameters in constrained (model) parameter space.

        Keyword arguments:
            inf_params: dictionary of parameters in unconstrained (sampler) parameter
                        space, keyed by name

        Returns:
            dictionary of parameters with same structure as inf_params, but
            in constrained (model) parameter space.
        """
        return inf_params

    # pylint: disable=unused-argument
    def log_det(self, model_params: ModelParams) -> jnp.DeviceArray:
        """Return total log determinant of transformation to constrained parameters

        Keyword arguments:
            model_params: dic of parameters in constrained (model) parameter space

        Returns:
            dictionary of parameters with same structure as model_params
        """
        return 0.0

    def inference(
        self,
        draws: int = 2_000,
        warmup_steps: int = 500,
        chains: int = 8,
        seed: int = 42,
        out: Progress = None,
        warmup_results=None,
    ) -> _Posterior:
        """Run HMC with full dataset, tuned by Stan+NUTS warmup

        Keyword arguments:
            draws:          number of draws per chain
            warmup_steps:   number of Stan warmup steps to run
            chains:         number of chains for main inference step
            seed:           random seed
            out:            progress indicator
            warmup_results: use this instead of running warmup
        """
        write = (out or Progress()).print
        rng_key = random.PRNGKey(seed)
        warmup_key, inference_key, post_key = random.split(rng_key, 3)

        if warmup_results:
            write("Skipping warmup")
        else:
            write("Starting Stan warmup using NUTS...")
            timer = Timer()
            warmup_results = warmup(
                self.potential,
                self.initial_value(),
                warmup_steps,
                chains,
                warmup_key,
            )
            write(
                f"      {warmup_steps} warmup draws took {timer}"
                f" ({warmup_steps/timer.sec:.1f} iter/sec)."
            )

        write(
            f"Obtaining {draws*chains:,} full-data posterior draws "
            f"({chains} chains, {draws:,} draws per chain)..."
        )
        timer = Timer()
        states = full_data_inference(
            self.potential, warmup_results, draws, chains, inference_key
        )
        write(
            f"      {chains*draws:,} HMC draws took {timer}"
            f" ({chains*draws/timer.sec:,.0f} iter/sec)."
        )

        # map positions back to model coordinates
        position_model = vmap(self.to_model_params)(states.position)
        # want axes to be (chain, draws, ... <variable dims> ...)
        rearranged_positions = {
            var: jnp.swapaxes(draws, axis1=0, axis2=1)
            for (var, draws) in position_model.items()
        }

        return _Posterior(
            self,
            post_draws=rearranged_positions,
            seed=seed,
            chains=chains,
            draws=draws,
            warmup_res=warmup_results,
            rng_key=post_key,
            write=write,
        )
