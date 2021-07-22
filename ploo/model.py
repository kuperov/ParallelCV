"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module defines a model class that users can extend to implement
arbitrary likelihood models.
"""
from typing import Any, Dict, Iterable, Tuple, Union

import arviz as az
import chex
from jax import numpy as jnp
from jax import random, vmap
from tabulate import tabulate

from .cross_validation import CrossValidation
from .hmc import (
    CrossValidationState,
    InfParams,
    WarmupResults,
    full_data_inference,
    warmup,
)
from .schemes import CrossValidationScheme
from .statistics import ess, split_rhat
from .util import Timer, print_devices, to_posterior_dict

# model parameters are in a constrained coordinate space
ModelParams = Dict[str, chex.ArrayDevice]

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
        posterior = to_posterior_dict(self.post_draws)
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
        scheme: Union[str, CrossValidationScheme] = "LOO",
        thin: int = 100,
        rng_key: chex.ArrayDevice = None,
        **kwargs,
    ) -> "CrossValidation":
        """Run cross-validation for this posterior.

        Number of chains and draws per chain are the same as the original inference
        procedure. Only decreate `thin` if you are sure you have enough memory on your
        GPU. Even moderately-sized problems can exhaust a GPU's memory quite quickly.

        :param scheme: name of cross-validation scheme to apply
        :param thin: thin MCMC draws
        :param rng_key: random generator state
        :param kwargs: arguments to pass to cross-validation scheme constructor

        :return: CrossValidation object containing all CV posteriors
        """
        return CrossValidation(
            self, scheme=scheme, thin=thin, rng_key=rng_key, **kwargs
        )

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

    def initial_value_transformed(self) -> InfParams:
        """Deterministic starting value, transformed to unconstrained inference space"""
        inf_params = self.forward_transform(self.initial_value())
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
        :param warmup_results: use this instead of running warmup
        :return: posterior object
        """
        rng_key = random.PRNGKey(seed)
        warmup_key, inference_key, post_key = random.split(rng_key, 3)
        draws, warmup_steps = int(draws), int(warmup_steps)
        title = f"Full-data posterior inference: {self.name}"
        print(title)
        print("=" * len(title))
        print_devices()
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

        # map positions back to model coordinates
        def inverse_tfm(params):
            mparam, _ = self.inverse_transform_log_det(params)  # ignore log det
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
