"""ploo: package for parallel cross-validation

This module compares models using cross-validation output
"""
from typing import Dict, List, Tuple

from tabulate import tabulate

from .model import CrossValidation


class ModelComparison:
    def __init__(
        self, ordered_cvs: List[Tuple[str, CrossValidation]], cv_type: str
    ) -> None:
        self.ordered_cvs = ordered_cvs
        self.cv_type = cv_type

    def __getitem(self, key):
        if isinstance(key, str):
            return dict(self.ordered_cvs)[key]
        return self.ordered_cvs[key]

    def __repr__(self) -> str:
        title = f"{self.cv_type} Cross Validation Comparison"
        best_elpd = self.ordered_cvs[0][1].elpd
        headers = ["Model", "elpd", "elpd diff", "elpd se"]
        table = [
            [name, cv.elpd, best_elpd - cv.elpd, cv.elpd_se]
            for name, cv in self.ordered_cvs
        ]
        output = [
            title,
            "=" * len(title),
            tabulate(table, headers=headers),
        ]
        return "\n".join(output)

    def names(self):
        return [n for n, _ in self.ordered_cvs]


def compare(*args: List[CrossValidation], **kwargs: Dict[str, CrossValidation]):
    """Compare cross-validations

    Compare cross-validation results and provide model rank ordering.

    For now just uses elpd.
    """
    # name arguments as model0, model1, ...
    arg_models = {
        f"model{i}": m for i, m in enumerate(args) if isinstance(m, CrossValidation)
    }
    kwarg_models = {n: m for (n, m) in kwargs.items() if isinstance(m, CrossValidation)}
    dupes = set(arg_models).intersection(kwarg_models)
    if dupes:
        raise Exception(f'Model names must be unique: {", ".join(dupes)}')
    arg_models.update(kwarg_models)
    ordering = sorted(arg_models.items(), key=lambda m1: m1[1])
    cv_types = [cv.cv_type for (_, cv) in ordering]
    if any(cv_types[0] != t for t in cv_types):
        raise Exception(f'CV types must be the same. Got: {", ".join(cv_types)}.')
    return ModelComparison(ordering, cv_type=cv_types[0])
