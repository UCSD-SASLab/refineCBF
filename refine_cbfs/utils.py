import functools
import numpy as np
import jax.numpy as jnp
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


def build_sdf(boundary, obstacles):
    """
    Args:
        boundary: [n x 2] matrix indicating upper and lower boundaries of safe space
        obstacles: list of [n x 2] matrices indicating obstacles in the state space
    Returns:
        Function that can be queried for unbatched state vector
    """
    def sdf(x):
        sdf = jnp.min(jnp.array([x - boundary[:,0], boundary[:,1] - x]))
        for obstacle in obstacles:
            max_dist_per_dim = jnp.max(jnp.array([obstacle[:,0] - x, x - obstacle[:,1]]), axis=0)
            def outside_obstacle(_):
                return jnp.linalg.norm(jnp.maximum(max_dist_per_dim, 0))
            def inside_obstacle(_):
                return jnp.max(max_dist_per_dim)
            obstacle_sdf = lax.cond(jnp.all(max_dist_per_dim) < 0.0, inside_obstacle, outside_obstacle, operand=None)
            sdf = jnp.min(jnp.array([sdf, obstacle_sdf]))
        return sdf
    return sdf


def hj_solve(solver_settings, dyn_hjr, grid, times, init_values):
    """
    Extension of hj_solve for parametric uncertainty.
    Args:
        dynamics: hj_reachability.Dynamics object
        grid: hj_reachability.Grid object
        target_values: [n x n x n x n] array of target values
        sdf: signed distance function
    Returns:
        [n x n x n x n] array of value function values
    """
    import hj_reachability as hj
    extrema = dyn_hjr.dynamics.get_param_combinations(type="extrema")
    values = []
    for i, extremum in enumerate(extrema):
        import copy
        dyn_hjr_alt = copy.deepcopy(dyn_hjr)
        dyn_hjr_alt.dynamics.params = extremum
        values_i = hj.solve(solver_settings, dyn_hjr_alt, grid, times, init_values)
        values.append(copy.deepcopy(values_i))
    
    return jnp.min(jnp.array(values), axis=0)
        
