from typing import Callable, Dict, Tuple, Union

from jax import random, vmap, numpy as jnp
from scipy import stats as sst

import matplotlib.pyplot as plt
from tabulate import tabulate

from .util import Progress, Timer
from .hmc import (
    CrossValidationState,
    InfParams,
    CVHMCState,
    WarmupResults,
    warmup,
    full_data_inference,
    cross_validate,
)


# model parameters are in a constrained coordinate space
ModelParams = Dict[str, jnp.DeviceArray]

# CV fold is either 1D or 2D integer index
CVFold = Union[int, Tuple[int, int]]


class Posterior(object):
    """ploo posterior: captures full-data and loo results

    Members:
        model: Model instance this was created from
        post_draws: posterior draw array
        cv_draws: cross-validation draws
        seed: seed used when invoking inference
    """

    def __init__(
        self,
        model: "Model",
        post_draws: jnp.DeviceArray,
        seed: int,
        chains: int,
        draws: int,
        warmup: WarmupResults,
        rng_key: jnp.DeviceArray,
        print: Callable,
    ) -> None:
        self.model = model
        self.post_draws = post_draws
        self.seed = seed
        self.chains = chains
        self.draws = draws
        self.warmup = warmup
        self.rng_key = rng_key
        self.print = print

    def __repr__(self) -> str:
        title = f"{self.model.name} inference summary"
        arg0 = next(iter(self.post_draws.position))
        it, ch = self.post_draws.position[arg0].shape
        desc_rows = [
            title,
            "=" * len(title),
            "",
            f"{it*ch:,} draws from {it:,} iterations on {ch:,} chains with seed {self.seed}",
            "",
        ] + [self.post_table()]
        return "\n".join(desc_rows)

    def post_table(self) -> str:
        """Construct a summary table for posterior draws"""
        table_headers = [
            "Parameter",
            "Mean",
            "(SE)",
            "1%",
            "5%",
            "25%",
            "Median",
            "75%",
            "95%",
            "99%",
        ]
        table_quantiles = jnp.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        table_rows = [
            [
                par,
                f"{jnp.mean(draws):4.2f}",
                f"({jnp.std(draws):.2f})",
            ]
            + [f"{q:.02f}" for q in jnp.quantile(draws, table_quantiles)]
            for par, draws in self.post_draws.position.items()
        ]
        return tabulate(table_rows, headers=table_headers)

    def cross_validate(self, draws=None, chains=None, rng_key=None):
        """Run cross-validation for this posterior.

        Keyword arguments:
            draws:   number of draws per chain
            chains:  number of chains per CV posterior
            rng_key: random generator state
        """
        chains = chains or self.chains
        rng_key = rng_key or self.rng_key
        draws = draws or self.draws
        timer = Timer()
        cv_chains = self.chains * self.model.cv_folds()
        self.print(
            f"Cross-validation with {self.model.cv_folds():,} folds "
            f"using {cv_chains:,} chains..."
        )

        def cond_pred(cv_fold: CVFold, inf_params: InfParams) -> jnp.DeviceArray:
            model_params = self.model.to_model_params(inf_params)
            self.model.log_cond_pred(cv_fold, model_params)

        accumulator, states = cross_validate(
            self.model.cv_potential,
            self.model.log_cond_pred,
            self.warmup,
            self.model.cv_folds(),
            draws,
            chains,
            rng_key,
        )
        self.print(
            f"      {cv_chains*draws:,} HMC draws took {timer}"
            f" ({cv_chains*draws/timer.sec:,.0f} iter/sec)."
        )

        # map positions back to model coordinates
        # FIXME: if we can evaluate objective online, this will not be necessary
        position_model = vmap(self.model.to_model_params)(states.position)
        states = CVHMCState(
            position_model,
            states.potential_energy,
            states.potential_energy_grad,
            states.cv_fold,
        )

        return CrossValidation(
            self,
            accumulator,
            states,
            draws=draws,
            chains=chains,
            folds=self.model.cv_folds(),
        )

    def trace_plot(self, par, figsize=(16, 8)) -> None:
        """Plot trace plots for posterior draws"""
        plt.plot(self.post_draws.position[par][:, :])

    def density(self, par, combine=False):
        """Kernel densities for full-data posteriors"""
        all_draws = self.post_draws.position[par]
        if combine:
            all_draws = jnp.expand_dims(jnp.reshape(all_draws, (-1,)), axis=1)
        for i in range(all_draws.shape[1]):
            draws = all_draws[:, i]
            kde = sst.gaussian_kde(draws)
            xs = jnp.linspace(min(draws), max(draws), 1_000)
            plt.plot(xs, kde(xs))
        plt.title(f"{par} posterior density")


class CrossValidation(object):
    """Cross-validated model

    This class contains draws for all the CV posteriors.
    """

    def __init__(
        self,
        post: Posterior,
        cv_accumulator: CrossValidationState,
        cv_states: CVHMCState,
        draws: int,
        chains: int,
        folds: int,
    ) -> None:
        self.post = post
        self.accumulator = cv_accumulator
        self.states = cv_states
        self.draws = draws
        self.chains = chains
        self.folds = folds

    @property
    def model(self) -> "Model":
        return self.post.model

    @property
    def elpd(self) -> float:
        return float(jnp.mean(self.accumulator.sum_log_pred_dens / self.draws))

    @property
    def divergences(self):
        return self.accumulator.divergence_count

    @property
    def num_divergent_chains(self):
        return int(jnp.sum(self.accumulator.divergence_count > 0))

    @property
    def acceptance_rates(self):
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
                f"    elpd = {self.elpd:.4f}",
                "",
                f"Calculated from {self.folds:,} folds ({self.chains:,} per fold, "
                f"{self.chains*self.folds} total chains)",
                "",
                f"Average acceptance rate {avg_accept*100:.1f}% (min {min_accept*100:.1f}%, "
                f"max {max_accept*100:.1f}%)",
                "",
                f"Divergent chain count: {self.num_divergent_chains}",
            ]
        )

    def block_until_ready(self) -> None:
        """Block the thread until results are back from the GPU."""
        self.cv_accumulator.divergence_count.block_until_ready()

    def densities(self, par, combine=False, ncols=4, figsize=(40, 80)):
        """Small-multiple kernel densities for cross-validation posteriors."""
        rows = int(jnp.ceil(self.model.cv_folds() / ncols))
        fig, axes = plt.subplots(nrows=rows, ncols=ncols, figsize=figsize)
        for fold, ax in zip(range(self.model.cv_folds()), axes.ravel()):
            chain_indexes = jnp.arange(fold * self.chains, (fold + 1) * self.chains)
            all_draws = self.states.position[par][:, chain_indexes]
            if combine:
                all_draws = jnp.expand_dims(jnp.reshape(all_draws, (-1,)), axis=1)
            for i in range(self.chains):
                draws = all_draws[:, i]
                try:
                    kde = sst.gaussian_kde(draws)
                    xs = jnp.linspace(min(draws), max(draws), 1_000)
                    ax.plot(xs, kde(xs))
                except Exception:
                    print(f"Error evaluating kde for fold {fold}, chain {i}")

    def trace_plots(self, par, ncols=4, figsize=(40, 80)) -> None:
        """Plot trace plots for every single cross validation fold."""
        rows = int(jnp.ceil(self.model.cv_folds() / ncols))
        fig, axes = plt.subplots(nrows=rows, ncols=ncols, figsize=figsize)
        for fold, ax in zip(range(self.model.cv_folds()), axes.ravel()):
            chain_indexes = jnp.arange(fold * self.chains, (fold + 1) * self.chains)
            ax.plot(self.states.position[par][:, chain_indexes])


class Model(object):
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

    def log_likelihood(
        self, model_params: ModelParams, cv_fold: CVFold = -1
    ) -> jnp.DeviceArray:
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
            cv_fold: cross-validation fold to evaluate

        Returns:
            log likelihood at the given parameters for the given CV fold
        """
        raise NotImplementedError()

    def log_prior(self, model_params: ModelParams):
        """Compute log prior log p(θ)

        JAX needs to be able to trace this function.

        Keyword args:
            params: dict of model parameters in constrained (model)
                    parameter space
        """
        return 0.0

    def log_cond_pred(self, model_params: ModelParams, cv_fold: CVFold):
        """Computes log conditional predictive ordinate, log p(ỹ|θˢ).

        FIXME: needs some kind of index to identify the conditioning values

        Keyword arguments:
            params:  a dict of parameters (constrained (model) parameter
                     space) that potentially contains vectors
            cv_fold: index of point at which to evaluate predictive density
        """
        raise NotImplementedError()

    def initial_value(self) -> ModelParams:
        """A deterministic starting value in model parameter space.

        FIXME: Deal with multiple independent chains.
               Make this a function of the chain number?
               Supply vectors of starting positions?
        """
        raise NotImplementedError()

    def initial_value_unconstrained(self) -> InfParams:
        """Deterministic starting value, transformed to unconstrained inference space"""
        return self.to_inference_params(self.initial_value())

    def cv_potential(self, inf_params: InfParams, cv_fold: CVFold) -> jnp.DeviceArray:
        """Potential for the given CV fold.

        We use index=-1 to indicate full-data likelihood (ie no CV folds dropped). If
        index >= 0, then the potential function should leave out a likelihood
        contribution identified by the value of cv_fold.

        Keyword arguments:
            inf_params: model parameters in inference (unconstrained) space
            cv_fold:    an integer corresponding to a CV fold
        """
        model_params = self.to_model_params(inf_params)
        llik = self.log_likelihood(model_params=model_params, cv_fold=cv_fold)
        lprior = self.log_prior(model_params=model_params)
        ldet = self.log_det(model_params=model_params)
        return -llik - lprior - ldet

    def cv_folds(self):
        """Number of cross-validation folds."""
        raise NotImplementedError()

    def parameters(self):
        """Names of parameters"""
        return list(self.initial_value().keys())

    @classmethod
    def generate(
        cls, random_key: jnp.DeviceArray, model_params: ModelParams
    ) -> jnp.DeviceArray:
        """Generate a dataset corresponding to the specified random key."""
        raise NotImplementedError()

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
        """Convert unconstrained (inference) params to constrained (model) parameter space

        Keyword arguments:
            inf_params: dictionary of parameters in unconstrained (sampler) parameter
                        space, keyed by name

        Returns:
            dictionary of parameters with same structure as inf_params, but
            in constrained (model) parameter space.
        """
        return inf_params

    def log_det(self, model_params: ModelParams) -> jnp.DeviceArray:
        """Return total log determinant of transformation to constrained parameters

        Keyword arguments:
            model_params: dictionary of parameters in constrained (model) parameter space

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
    ) -> Posterior:
        """Run HMC with full dataset, tuned by Stan+NUTS warmup

        Keyword arguments:
            draws:          number of draws per chain
            warmup_steps:   number of Stan warmup steps to run
            chains:         number of chains for main inference step
            seed:           random seed
            out:            progress indicator
            warmup_results: use this instead of running warmup
        """
        print = (out or Progress()).print
        rng_key = random.PRNGKey(seed)
        warmup_key, inference_key, post_key = random.split(rng_key, 3)

        print("Thor's Cross-Validatory Hammer")
        print("==============================\n")

        if warmup_results:
            print("Skipping warmup")
        else:
            print("Starting Stan warmup using NUTS...")
            timer = Timer()
            warmup_results = warmup(
                self.cv_potential,
                self.initial_value(),
                warmup_steps,
                chains,
                warmup_key,
            )
            print(
                f"      {warmup_steps} warmup draws took {timer}"
                f" ({warmup_steps/timer.sec:.1f} iter/sec)."
            )

        print(f"Running full-data inference with {chains} chains...")
        timer = Timer()
        states = full_data_inference(
            self.cv_potential, warmup_results, draws, chains, inference_key
        )
        print(
            f"      {chains*draws:,} HMC draws took {timer}"
            f" ({chains*draws/timer.sec:,.0f} iter/sec)."
        )

        # map positions back to model coordinates
        position_model = vmap(self.to_model_params)(states.position)
        states = CVHMCState(
            position_model,
            states.potential_energy,
            states.potential_energy_grad,
            states.cv_fold,
        )

        return Posterior(
            self,
            post_draws=states,
            seed=seed,
            chains=chains,
            draws=draws,
            warmup=warmup_results,
            rng_key=post_key,
            print=print,
        )
