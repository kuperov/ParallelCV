"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module compares models using cross-validation output
"""

from typing import Callable, DefaultDict, Iterable, Iterator, List, Sequence, Tuple

import chex
import numpy as np
from jax import numpy as jnp
from jax import random

CVFold = int


@chex.dataclass
class PredCoords:
    """Set of (possibly unbalanced) prediction coordinates

    This class contains a set of prediction coordinates as a "jagged array" implemented
    using a jax array and an array of masks to apply to the predictions, with 1 meaning
    include and 0 meaning exclude. We took this approach because Alex couldn't work out
    how to dynamically slice the coordinate array within the inference loop, so we just
    perform a static number of predictions for each group, and filter out the ones that
    aren't needed.

    The coords array axes are:

      * axis0: fold number
      * axis1: deleted cases
      * axis2: case coordinates, wrt to log likelihood contribution array

    The masks array has shape coords.shape[:2], and is defined by code equivalent to:

        masks[i] = jnp.array(1.0 * jnp.arange(coords.shape[1]) <= lengths[i])
    """

    coords: chex.ArrayDevice
    lengths: chex.ArrayDevice
    masks: chex.ArrayDevice


class CrossValidationScheme(Iterable):
    """Abstract class representing a cross-validation (CV) scheme.

    A CV scheme is a pattern of case deletions and predictive coordinates
    used to construct an estimate of a model's predictive ability.

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

    def __init__(self, name: str, num_folds: int, **_kwargs) -> None:
        self.name = name
        self.num_folds = num_folds

    def mask_for(self, fold: CVFold) -> chex.ArrayDevice:
        """Array to multiply elementwise with likelihood contributions

        :param fold: a cross-validation fold
        :return: array of shape `self.shape`
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.name} cross-validation"

    def coordinates_for(self, fold: CVFold) -> chex.ArrayDevice:
        """Returns CV coordinates for the given fold.

        The coordinates refer to the shape of the likelihood contribution array.

        :param fold: integer fold identifier
        :return: coordinates as an integer array
        """
        raise NotImplementedError()

    @property
    def folds(self) -> int:
        """Number of CV folds in this scheme."""
        return self.num_folds

    def pred_indexes(self) -> PredCoords:
        """Generate list of prediction indexes.
        The resulting list has 3 dimensions than the dependence structure,
        with axis 0 as the fold, axis 1 as the deleted cases, and axis 2 as the
        case coordinates. Numpy and jax don't support jagged arrays, so the
        returned object also contains an array of coordinate counts; users should
        take the first <count> elements of the coordinate array.
        """
        coord_list = [np.atleast_1d(self.coordinates_for(fold_id)) for fold_id in self]
        coord_counts = jnp.array([len(cs) for cs in coord_list], dtype=jnp.int32)
        max_coord_count = max(coord_counts)
        coord_shape = coord_list[0].shape[1:]
        index_shape = (self.folds, max_coord_count) + coord_shape
        index = np.zeros(shape=index_shape, dtype=np.int32)
        for i, coords in enumerate(coord_list):
            index[i, : len(coords), ...] = coord_list[i]
        masks = jnp.vstack(
            jnp.array(1.0 * (jnp.arange(index.shape[1]) <= coord_counts[i]))
            for i in range(self.folds)
        )
        return PredCoords(coords=jnp.array(index), lengths=coord_counts, masks=masks)

    def mask_array(self) -> chex.ArrayDevice:
        """Generate array of likelihood contribution masks.
        Basically stacks masks on top of each other, row by row. For 1D data, output
        is a 2D array, etc.

        :return: array of arrays of same shape as likelihood contribution matrix
        """
        return jnp.stack([self.mask_for(fold) for fold in self])

    def summary_array(self) -> np.ndarray:
        """Array for visualizing this CV scheme.

        Interpretation:
          * deleted = 0.0
          * training set = 1.0
          * test set = 2.0

        Can plot this in a notebook with `matplotlib.pyplot.matshow()`

        NB: this is really only useful for 1D schemes, where the output is 2D. In
        higher dimensions, you probably want to take slices of the returned array.
        """

        def array_for_fold(fold_i):
            # use mutable numpy arrays
            mask = np.array(self.mask_for(fold_i))
            for pred_i in self.coordinates_for(fold_i):
                mask[pred_i] = -1.0
            return mask

        return np.stack([array_for_fold(i) for i in range(self.folds)])

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.folds))


class LOO(CrossValidationScheme):
    """Leave-one-out cross validation.

    Each fold removes just one likelihood contribution.
    """

    def __init__(self, shape: Tuple, **_kwargs) -> None:
        """Create a new LOO CrossValidation.

        :param shape: numpy shape of likelihood contribution array
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        assert len(self.shape) >= 1 and len(self.shape) <= 3
        super().__init__("LOO", np.prod(self.shape))

    def mask_for(self, fold: CVFold) -> chex.ArrayDevice:
        mask = np.ones(shape=self.shape)
        for coord in self.coordinates_for(fold):
            mask[coord] = 0.0
        return jnp.array(mask)

    def coordinates_for(self, fold: CVFold) -> List[Tuple]:
        if len(self.shape) == 1:
            return [(fold,)]
        if len(self.shape) == 2:
            coord_0 = fold % self.shape[0]
            fold //= self.shape[0]
            coord_1 = fold % self.shape[1]
            return [(coord_0, coord_1)]
        if len(self.shape) == 3:
            coord_0 = fold % self.shape[0]
            fold //= self.shape[0]
            coord_1 = fold % self.shape[1]
            fold //= self.shape[1]
            coord_2 = fold % self.shape[2]
            return [(coord_0, coord_1, coord_2)]
        raise Exception("Invalid shape")


class LFO(CrossValidationScheme):
    """Leave-future-out cross validation.

    Each fold removes future observations, leaving a margin of v
    observations at the start.
    """

    def __init__(self, shape, margin: int, **_kwargs) -> None:
        """Create a new leave-future-out (LFO) CrossValidation.

        This currently only works with one-dimensional dependence structures
        (like univariate time series).

        :param shape: length of 1D likelihood contribution array
        :param margin: number of observations to always include at start of sequence
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        self.margin = margin
        assert len(self.shape) == 1
        super().__init__("LFO", self.shape[0] - self.margin)

    def mask_for(self, fold: CVFold) -> chex.ArrayDevice:
        return jnp.concatenate(
            [
                jnp.ones(shape=(self.margin + fold,)),
                jnp.zeros(shape=(self.shape[0] - fold - self.margin,)),
            ]
        )

    def coordinates_for(self, fold: CVFold) -> chex.ArrayDevice:
        return jnp.array([[fold + self.margin]])


class KFold(CrossValidationScheme):
    """Random K-Fold cross validation

    Each fold removes N/K likelihood contributions
    """

    def __init__(self, shape, k, rng_key, **_kwargs) -> None:
        """Create new KFold object

        :param shape:   numpy shape of likelihood contributions
        :param k:       number of folds
        :param rng_key: random number generator key
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
        super().__init__(f"Random {k}-fold", self.k)

    def mask_for(self, fold: CVFold) -> chex.ArrayDevice:
        return self.masks[fold]

    def coordinates_for(self, fold: CVFold) -> chex.ArrayDevice:
        return self.coords[fold]


class LGO(CrossValidationScheme):
    """Leave-group-out (LGO) cross-validation scheme

    Leaves a single group out of the posterior each fold. Groups are identified
    by an array of group numbers.

    .. note::
        At this time, only a 1D array of likelihood contributions is supported,
        corresponding to the bottom of the model hierarchy.
    """

    def __init__(self, shape: Tuple, group_ids: Iterable[int], **_kwargs) -> None:
        """Create LGO instance

        :param shape: shape of (1D) log likelihood contribution array
        :param group_ids: group identifiers, of same shape as log likelihood
            contributions
        """
        self.shape = shape if isinstance(shape, Sequence) else (shape,)
        assert len(self.shape) == 1, "Only 1D lower level supported"
        self.obs_ids = list(group_ids)
        groups = DefaultDict(list)
        for obs_id, x in enumerate(self.obs_ids):
            groups[x].append([obs_id])
        num_folds = len(groups)
        mutable_masks = np.ones(shape=(num_folds, self.shape[0]))
        # define fresh indexes, contiguous and starting at zero
        fold_indexes = dict(enumerate(groups))
        self.coords = {
            i: jnp.array(groups[group_id]) for i, group_id in fold_indexes.items()
        }
        for obs_id in range(num_folds):
            mutable_masks[obs_id, self.coords[obs_id]] = 0.0
        self.masks = jnp.array(mutable_masks)
        name = f"Leave-group-out (LGO) CV with {num_folds} groups/folds"
        super().__init__(name, num_folds)

    def coordinates_for(self, fold: CVFold) -> chex.ArrayDevice:
        return self.coords[fold]

    def mask_for(self, fold: CVFold) -> chex.ArrayDevice:
        return self.masks[fold]


def cv_factory(name: str) -> Callable:
    """Returns constructor for CV scheme identified by name"""
    cls = {"LOO": LOO, "LFO": LFO, "KFold": KFold}[name]
    return cls
