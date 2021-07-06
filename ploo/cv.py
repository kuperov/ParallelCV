"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module compares models using cross-validation output
"""

from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from jax import numpy as jnp
from jax import random

CVFold = int


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

    def coordinates_for(self, fold: CVFold) -> jnp.DeviceArray:
        """Returns CV coordinates for the given fold.

        The coordinates refer to the shape of the likelihood contribution array.

        Keyword arguments:
            fold: integer fold identifier

        Returns:
            coordinates as an integer array
        """
        raise NotImplementedError()

    def cv_folds(self) -> int:
        """Number of CV folds in this scheme."""
        raise NotImplementedError()

    def pred_index_array(self) -> jnp.DeviceArray:
        """Generate array of prediction indexes

        The resulting array has one more dimension than the dependence structure,
        with axis 0 as the fold.
        """
        return jnp.stack([np.atleast_1d(fold_coord) for fold_coord in self])

    def mask_array(self) -> jnp.DeviceArray:
        """Generate array of likelihood contribution masks

        Basically stacks masks on top of each other, row by row. For 1D data, output
        is a 2D array, etc.

        Returns
            jnp.DeviceArray
        """
        return jnp.stack([self.mask_for(fold) for fold in self])

    def summary_array(self) -> np.ndarray:
        """Array for visualizing this CV scheme.

        Interpretation:
          * deleted = 0.0
          * training set = 1.0
          * test set = 2.0

        Can plot this in a notebook with `matplotlib.pyplot.matshow()`

        NB: this is really only useful for 1D schemes, where the output is 2D.
        """

        def array_for_fold(fold_i):
            # use mutable numpy arrays
            mask = np.array(self.mask_for(fold_i))
            for pred_i in self.coordinates_for(fold_i):
                mask[pred_i] = -1.0
            return mask

        return np.stack([array_for_fold(i) for i in range(self.cv_folds())])


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

    def coordinates_for(self, fold: CVFold) -> jnp.DeviceArray:
        return jnp.array([fold])

    def cv_folds(self) -> int:
        return np.prod(self.shape)

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
        return jnp.concatenate(
            [
                jnp.ones(shape=(self.margin + fold,)),
                jnp.zeros(shape=(self.shape[0] - fold - self.margin,)),
            ]
        )

    def coordinates_for(self, fold: CVFold) -> jnp.DeviceArray:
        return jnp.array([fold + self.margin])

    def cv_folds(self) -> int:
        return self.shape[0] - self.margin

    def __iter__(self) -> Iterator[CVFold]:
        return iter(range(self.shape[0] - self.margin))


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

    def coordinates_for(self, fold: CVFold) -> jnp.DeviceArray:
        return self.coords[fold]

    def cv_folds(self) -> int:
        return self.k


def cv_factory(name: str) -> Callable:
    """Returns constructor for CV scheme identified by name"""
    cls = {"LOO": LOO, "LFO": LFO, "KFold": KFold}[name]
    return cls
