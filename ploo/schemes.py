"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module compares models using cross-validation output
"""

from typing import Callable, DefaultDict, Iterable, Iterator, Sequence, Tuple

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

        Args:
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

        Args:
            fold: integer fold identifier

        Returns:
            coordinates as an integer array
        """
        raise NotImplementedError()

    def cv_folds(self) -> int:
        """Number of CV folds in this scheme."""
        raise NotImplementedError()

    def pred_indexes(self) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        """Generate list of prediction indexes

        The resulting list has one more dimension than the dependence structure,
        with "axis 0" as the fold. Numpy and jax don't support jagged arrays, so
        we also return an array of coordinate counts; users should take the first
        <count> elements of the coordinate array.
        """
        coord_list = [np.atleast_1d(self.coordinates_for(fold_id)) for fold_id in self]
        coord_counts = jnp.array([len(cs) for cs in coord_list], dtype=jnp.int32)
        max_coord_count = max(coord_counts)
        coord_shape = coord_list[0].shape[1:]
        index_shape = (self.cv_folds(), max_coord_count) + coord_shape
        index = np.zeros(shape=index_shape, dtype=np.int32)
        for i, coords in enumerate(coord_list):
            index[i, : len(coords), ...] = coord_list[i]
        return jnp.array(index), coord_counts

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

        Args:
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

        Args:
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


class LGO(CrossValidationScheme):
    """Leave-group-out (LGO) cross-validation scheme

    Leaves a single group out of the posterior each fold. Groups are identified
    by an array of group numbers.

    .. note::
        At this time, only a 1D array of likelihood contributions is supported,
        corresponding to the bottom of the model hierarchy.
    """

    def __init__(self, shape: Tuple, group_ids: Iterable[int]) -> None:
        """Create LGO instance

        :param shape: shape of (1D) log likelihood contribution array
        :param group_ids: group identifiers, of same shape as log likelihood contributions
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        assert len(self.shape) == 1, "Only 1D lower level supported"
        self.ids = list(group_ids)
        groups = DefaultDict(list)
        for i, x in enumerate(self.ids):
            groups[x].append(i)
        self.num_folds = len(groups)
        mutable_masks = np.ones(shape=(self.num_folds, self.shape[0]))
        # define fresh indexes, contiguous and starting at zero
        group_indexes = dict(enumerate(groups))
        self.coords = {
            i: jnp.array(groups[group_id]) for i, group_id in group_indexes.items()
        }
        for i in range(self.num_folds):
            mutable_masks[i, self.coords[i]] = 0.0
        self.masks = jnp.array(mutable_masks)
        name = f"Leave-group-out (LGO) CV with {self.num_folds} groups/folds"
        super().__init__(name)

    def cv_folds(self) -> int:
        return self.num_folds

    def coordinates_for(self, fold: CVFold) -> jnp.DeviceArray:
        return self.coords[fold]

    def mask_for(self, fold: CVFold) -> jnp.DeviceArray:
        return self.masks[fold]

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.num_folds))


def cv_factory(name: str) -> Callable:
    """Returns constructor for CV scheme identified by name"""
    cls = {"LOO": LOO, "LFO": LFO, "KFold": KFold}[name]
    return cls
