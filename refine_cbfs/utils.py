import functools
import numpy as np
import jax.numpy as jnp
from jax import lax
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
        
    


# def animate_figure(ax, static_objects, dynamic_objects):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)


#     ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#             tabular_cbf.vf_table[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[0], colors='grey', linewidths=4)  
#     cont3 = ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#             sdf_values[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[-10, 0], colors='red')
#     cont2 = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#             sdf_values[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[0], colors='k', linewidths=4)
#     cont = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#             target_values_full[0][:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[0], colors='green', linewidths=2)
#     ax.set_xlabel('$y$ (Horizontal)')
#     ax.set_ylabel('$z$ (Vertical)')
#     tx = ax.set_title(f'$v_y=0, v_z=0$, HJR time $t=0$')

# # Update function to draw contours for a given idi value
# def update(idi):
#     global cont, cont2, cont3
#     arr = target_values_full[idi][:, :, grid.shape[2] // 2, grid.shape[3] // 2].T
#     arr_sdf = sdf_full[idi][:, :, grid.shape[2] // 2, grid.shape[3] // 2].T
#     vmax = np.abs(arr).max()
#     # ax.clear()
#     cont.collections[0].remove()
#     cont = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#                arr, levels=[0], colors='green')
#     cont2.collections[0].remove()
#     cont2 = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#                           arr_sdf, levels=[0], colors='k', linewidths=4)
#     cont3.collections[0].remove()
#     cont3 = ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#            arr_sdf, levels=[-10, 0], colors='red')
#     tx.set_text('$v_y=0, v_z=0$, HJR time t={:.2f}'.format(np.abs(times[idi].item())))


# # Animate with idi values from 0 to 11
# ani = FuncAnimation(fig, update, frames=range(len(times)));
# plt.close()
# HTML(ani.to_jshtml())



# #################################
# def find_closest_time(df, time_ind):
#     return df.t.iloc[df.t.sub(time_ind).abs().idxmin()]


# def visualize(fig, timestamps, contours = None, dataframe = None, **kwargs):
    


# ss_exp = StateSpaceExperiment('quad', x_indices=[0, 1], start_x=x0)
# # Set up the figure and axis
# fig = plt.figure(figsize=(20,10))
# ax = fig.add_subplot(111)
# alphas = [0.1, 0.5, 0.5, 1.0]
# nbr_controllers = len(results_df.controller.unique())

# plt.legend(results_df.controller.unique())
# ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#            tabular_cbf.vf_table[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[0], colors='grey', linewidths=4)  
# cont3 = ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#            sdf_values[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[-10, 0], colors='red')
# cont2 = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#            sdf_values[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[0], colors='k', linewidths=4)
# cont = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#            target_values_f(0.0)[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T, levels=[0], colors='green', linewidths=2)
# ax.set_xlabel('$y$ (Horizontal)', fontsize=20)
# ax.set_ylabel('$z$ (Vertical)', fontsize=20)
# tx = ax.set_title('$v_y=0, v_z=0$, HJR time $t=0$')
# ss_exp.plot(batched_dyn, results_df, ax=ax, add_direction=False, max_time=0.0, alpha=alphas)
# ax.legend(ax.lines[::len(ax.lines) // nbr_controllers], results_df.controller.unique(), loc="upper center", ncol=4, fontsize=20)

# # Update function to draw contours for a given idi value
# def update(time):
#     global cont, cont2, cont3
#     arr = target_values_f(time)[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T
#     sdf_arr = sdf_values_f(time)[:, :, grid.shape[2] // 2, grid.shape[3] // 2].T
#     # ax.clear()
#     for line in ax.lines:
#         line.remove()
#     for patch in ax.patches:
#         patch.remove()
#     ax.set_prop_cycle(None)
#     cont.collections[0].remove()
#     cont = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#                arr, levels=[0], colors='green')
#     cont2.collections[0].remove()
#     cont2 = ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#                             sdf_arr, levels=[0], colors='k', linewidths=4)
#     cont3.collections[0].remove()
#     cont3 = ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1],
#                         sdf_arr, levels=[-10, 0], colors='red')
#     ss_exp.plot(batched_dyn, results_df, ax=ax, add_direction=False, max_time=time, alpha=alphas)
#     closest_time = find_closest_time(results_df, time)
#     curr_vals = results_df[(results_df.t == closest_time) & (results_df.measurement.isin(["y", "z", "tan(phi)"]))].value.values.reshape(nbr_controllers, -1)
#     colors = []
#     for line in ax.lines[::len(ax.lines) // nbr_controllers]:
#         colors.append(line.get_color())
#     for i, curr_val in enumerate(curr_vals):
#         # get color from prop_cycle 
#         get_drone(ax, curr_val[0], curr_val[1], np.arctan(-curr_val[2]), rel_size=0.3, height_scale=0.9, alpha=alphas[i], color=colors[i])
#     tx.set_text('$v_y=0, v_z=0$, Simulation time t={:.2f}'.format(np.abs(time)))
#     fig.tight_layout()

# ani = FuncAnimation(fig, update, frames=np.linspace(0,20,100))
# plt.close()
# HTML(ani.to_jshtml())