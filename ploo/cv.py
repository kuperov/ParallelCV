"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module compares models using cross-validation output
"""

from typing import Iterable, Iterator, Sequence, Tuple, Union

import numpy as np
from jax import numpy as jnp
from jax import random

CVFold = Union[int, Tuple[int, int], Tuple[int, int, int]]


class CrossValidation(Iterable):
    """Abstract class representing a structured cross-validation.

    Each instance is instantiated with a numpy-style `shape` that
    denotes the dimensions of the likelihood contribution array.

    Iterating over a cross-validation yields an iterable of indicator
    mask arrays (one per fold) that should be multiplied elementwise
    with the likelihood contribution arrays to produce the likelihood
    for the model in each fold.
    """

    def __init__(self) -> None:
        """Abstract class: not implemented"""
        raise NotImplementedError()

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.name} cross-validation"

    def summary_matrix(self) -> jnp.DeviceArray:
        """Generate summary matrix for 1D CV schemes.

        Basically stacks masks on top of each other, row by row.

        Returns
            jnp.DeviceArray
        """
        return NotImplementedError()


class LOO(CrossValidation):
    """Leave-one-out cross validation.

    Each fold removes just one likelihood contribution.
    """

    name = "LOO"

    def __init__(self, shape) -> None:
        """Create a new LOO CrossValidation.

        Keyword args:
            shape: numpy shape of likelihood contribution array
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        assert len(self.shape) >= 1 and len(self.shape) <= 3
        self.folds = np.prod(self.shape)

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        return jnp.ones(shape=self.shape).at[fold].set(0.0)

    def __iter__(self) -> Iterator[CVFold]:
        if len(self.shape) == 1:
            # integer indexes
            return iter(range(self.shape[0]))
        else:
            # tuple indexes
            return np.ndindex(self.shape)


class LFO(CrossValidation):
    """Leave-future-out cross validation.

    Each fold removes future observations, leaving a margin of v
    observations at the start.
    """

    name = "LFO"

    def __init__(self, shape, margin: int) -> None:
        """Create a new leave-future-out (LFO) CrossValidation.

        This currently only works with one-dimensional datasets.

        Keyword args:
            shape:  length of 1D likelihood contribution array
            margin: number of observations to always include at start of sequence
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        self.margin = margin
        assert len(self.shape) == 1
        self.folds = self.shape[0] - self.margin

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        return jnp.ones(shape=self.shape).at[fold + self.margin].set(0.0)

    def __iter__(self) -> Iterator[CVFold]:
        # Fold indexes still start at zero, but don't correspond to
        # the missing contribution. We could have gone either way here
        # but this approach helps check we aren't breaking the ADT elsewhere
        return iter(range(self.shape[0] - self.margin))


class KFold(CrossValidation):
    """Random K-Fold cross validation

    Each fold removes N/K likelihood contributions
    """

    def __init__(self, shape, k, rng_key) -> None:
        """Create new KFold object

        Keyword arguments
            shape:   numpy shape of likelihood contributions
            k:       number of folds
            rng_key: random number generator key
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        self.k = k
        self.name = f"Random {k}-fold"
        # randomly allocate coords to folds
        prototype_masks = {index: np.ones(shape=shape) for index in range(self.k)}
        # I know we're mixing numpy and jnp here but we want to make sure
        # we use the jax random state
        coord_list = list(np.ndindex(*self.shape))
        all_coords = random.permutation(key=rng_key, x=jnp.array(coord_list))
        # this obviously can't be traced but that shouldn't be an issue because
        # it only happens once at the start of the CV procedure
        for i, coord in enumerate(all_coords):
            prototype_masks[i % k][tuple(coord)] = 0.0
        self.masks = {fold: jnp.array(mask) for fold, mask in prototype_masks.items()}
        self.folds = self.k

    def __iter__(self) -> Iterator[CVFold]:
        return iter(range(self.k))

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        return self.masks[fold]
