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

Coordinate = Union[int, Tuple[int, int], Tuple[int, int, int]]


class CrossValidationScheme(Iterable):
    """Abstract class representing a structured cross-validation.

    Each instance is instantiated with a numpy-style `shape` that
    denotes the dimensions of the likelihood contribution array.

    Iterating over a cross-validation yields an iterable of fold
    identifiers. These fold identifiers are keys that identify:

      * indicator mask arrays (one per fold) that should be multiplied
        elementwise with the likelihood contribution arrays to produce
        the likelihood for the model in each fold; and

      * lists of coordinates for the same arrays to be used for
        evaluating conditional predictives.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        """Array to multiply elementwise with likelihood contributions

        Keyword arguments:
            fold: a cross-validation fold

        Returns:
            jnp.DeviceArray of shape `self.shape`
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.name} cross-validation"

    def coordinates_for(self, fold: CVFold) -> Iterable[Coordinate]:
        """Returns CV coordinates for the given fold.

        The coordinates refer to the shape of the likelihood contribution array.

        Keyword arguments:
            fold: fold identifier (integers or tuples)

        Returns:
            iterator over coordinates (integers or tuples)
        """
        raise NotImplementedError()

    def summary_array(self) -> jnp.DeviceArray:
        """Generate summary matrix for 1D CV schemes.

        Basically stacks masks on top of each other, row by row.
        For 1D data, output is a 2D array, etc.

        Returns
            jnp.DeviceArray
        """
        arrays = [self.mask_for(fold) for fold in self]
        return np.stack(arrays)


class LOO(CrossValidationScheme):
    """Leave-one-out cross validation.

    Each fold removes just one likelihood contribution.
    """

    def __init__(self, shape) -> None:
        """Create a new LOO CrossValidation.

        Keyword args:
            shape: numpy shape of likelihood contribution array
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        assert len(self.shape) >= 1 and len(self.shape) <= 3
        self.folds = np.prod(self.shape)
        super().__init__("LOO")

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        return jnp.ones(shape=self.shape).at[fold].set(0.0)

    def coordinates_for(self, fold: CVFold) -> Iterable[Coordinate]:
        return [fold]

    def __iter__(self) -> Iterator[CVFold]:
        if len(self.shape) == 1:
            # integer indexes
            return iter(range(self.shape[0]))
        # tuple indexes
        return np.ndindex(self.shape)


class LFO(CrossValidationScheme):
    """Leave-future-out cross validation.

    Each fold removes future observations, leaving a margin of v
    observations at the start.
    """

    def __init__(self, shape, margin: int) -> None:
        """Create a new leave-future-out (LFO) CrossValidation.

        This currently only works with one-dimensional dependence structures
        (like univariate time series).

        Keyword args:
            shape:  length of 1D likelihood contribution array
            margin: number of observations to always include at start of sequence
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        self.margin = margin
        assert len(self.shape) == 1
        self.folds = self.shape[0] - self.margin
        super().__init__("LFO")

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        # fold is coordinate of dropped observation
        return jnp.ones(shape=self.shape).at[fold].set(0.0)

    def coordinates_for(self, fold: CVFold) -> Iterable[Coordinate]:
        return [fold]

    def __iter__(self) -> Iterator[CVFold]:
        return iter(range(self.margin, self.shape[0]))


class KFold(CrossValidationScheme):
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
        # randomly allocate coords to folds
        prototype_masks = {index: np.ones(shape=shape) for index in range(self.k)}
        self.coords = {index: [] for index in range(self.k)}
        # I know we're mixing numpy and jnp here but we want to make sure
        # we use the jax random state
        shuffled = random.permutation(
            key=rng_key, x=jnp.array(list(np.ndindex(*self.shape)))
        )
        # this obviously can't be traced but that shouldn't be an issue because
        # it only happens once at the start of the CV procedure
        for i, coord in enumerate(shuffled):
            prototype_masks[i % k][tuple(coord)] = 0.0
            self.coords[i % k].append(coord)
        self.masks = {fold: jnp.array(mask) for fold, mask in prototype_masks.items()}
        self.folds = self.k
        super().__init__(f"Random {k}-fold")

    def __iter__(self) -> Iterator[CVFold]:
        return iter(range(self.k))

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        return self.masks[fold]

    def coordinates_for(self, fold: CVFold) -> Iterable[Coordinate]:
        return self.coords[fold]
