import functools
import numpy as np
from hj_reachability import Grid


class Grid(Grid):
    """Numpy implementation of interpolate function of hj.Grid. Delivers 5x speedup."""

    def interpolate(self, values, state):
        """Interpolates `values` (possibly multidimensional per node) defined over the grid at the given `state`."""
        position = (state - self.domain.lo) / np.array(self.spacings)
        index_lo = np.floor(position).astype(np.int32)
        index_hi = index_lo + 1
        weight_hi = position - index_lo
        weight_lo = 1 - weight_hi
        index_lo, index_hi = tuple(
            np.where(
                self._is_periodic_dim,
                index % np.array(self.shape),
                np.clip(index, 0, np.array(self.shape)),
            )
            for index in (index_lo, index_hi)
        )
        weight = functools.reduce(lambda x, y: x * y, np.ix_(*np.stack([weight_lo, weight_hi], -1)))
        # TODO: Update based on resolved TODO in `hj_reachability`
        return np.sum(
            weight[(...,) + (np.newaxis,) * (values.ndim - self.ndim)]
            * values[np.ix_(*np.stack([index_lo, index_hi], -1))],
            tuple(range(self.ndim)),
        )
