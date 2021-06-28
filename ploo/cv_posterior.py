from typing import List
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tabulate import tabulate

from .model import CVModel


class CVPosterior(object):
    """ploo posterior: captures full-data and loo results

    Members:
        model: CVModel instance this was created from
        post_draws: posterior draw array
        cv_draws: cross-validation draws
        seed: seed used when invoking inference
    """

    def __init__(self, model: CVModel, post_draws, cv_draws, seed) -> None:
        self.model = model
        self.post_draws = post_draws
        self.cv_draws = cv_draws
        self.seed = seed

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

    def cv_trace_plots(self, par, ncols=4, figsize=(40, 80)) -> None:
        """Plot trace plots for every single cross validation fold."""
        rows = int(jnp.ceil(self.model.cv_folds / ncols))
        fig, axes = plt.subplots(nrows=rows, ncols=ncols, figsize=figsize)
        for fold, ax in zip(range(self.model.cv_folds), axes.ravel()):
            ax.plot(
                self.cv_draws.position[par][:, jnp.arange(fold * 4, (fold + 1) * 4)]
            )

    def trace_plot(self, par, figsize=(16, 8)) -> None:
        """Plot trace plots for posterior draws"""
        plt.plot(self.post_draws.position["sigma"][:, :])
