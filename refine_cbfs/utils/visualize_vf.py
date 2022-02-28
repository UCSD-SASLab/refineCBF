import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# class ValueFunctionVisualization:
#     def __init__(self, grid, constraint_set, dims_to_display, **kwargs):
#         self.grid = grid
#         self.grid_display_dims = [self.grid.coordinate_vectors[dim] for dim in dims_to_display]
#         self.constraint_set = constraint_set
#         pass

#     def contour_plot_fig(self, target_values, filled=False, **kwargs):
#         fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 10)))

#         if target_values.ndim != 2:
#             dims_to_slice = kwargs['dims_to_slice']
#             indices_of_sliced = kwargs['indices_of_sliced']
#             target_values = np.take(target_values, indices_of_sliced, axis=dims_to_slice)
#             constraint_set = np.take(self.constraint_set, indices_of_sliced, axis=dims_to_slice)
            

#         if filled:
#             vmin, vmax = target_values.min(), target_values.max()
#             cmap = sns.diverging_palette(vmin=vmin, vmax=vmax, as_cmap=True)
#             ax.contourf(*self.grid_display_dims, target_values, cmap=cmap)
#         else:
#             level_sets = kwargs.get("level_sets", [0])
#             cont=ax.contour(*self.grid_display_dims, target_values, levels=level_sets)

#     def Threedim_contour_plot(self, **kwargs):
#         raise NotImplementedError("To be implemented")
#         # TODO: Use go.Figure (plotly.graph_objects) to create a 3D contour plot

#     def contour_plot_video(self, target_values, filled=False, **kwargs):
#         fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 10)))



#         if filled:
#             vmin, vmax = target_values.min(), target_values.max()
#             cmap = sns.diverging_palette(vmin=vmin, vmax=vmax, as_cmap=True)
#             ax.contourf(*self.grid_display_dims, target_values, cmap=cmap)
#         else:
#             level_sets = kwargs.get("level_sets", [0])
#             ax.,

#         def _update_frame(idx):



"""
Animations for various dynamical systems using `matplotlib`.
Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation


def animate_planar_quad(t, x, y, θ, title_string=None, display_in_notebook=False):
    """Animate the planar quadrotor system from given position data.
    All arguments are assumed to be 1-D NumPy arrays, where `x`, `y`, and `θ`
    are the degrees of freedom of the planar quadrotor over time `t`.
    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_planar_quad
        fig, ani = animate_planar_quad(t, x, θ)
        ani.save('planar_quad.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    rod_width = 2.
    rod_height = 0.15
    axle_height = 0.2
    axle_width = 0.05
    prop_width = 0.5 * rod_width
    prop_height = 1.5 * rod_height
    hub_width = 0.3 * rod_width
    hub_height = 2.5 * rod_height

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x), np.max(x)
    x_pad = (rod_width + prop_width) / 2 + 0.1 * (x_max - x_min)
    y_min, y_max = np.min(y), np.max(y)
    y_pad = (rod_width + prop_width) / 2 + 0.1 * (y_max - y_min)
    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([y_min - y_pad, y_max + y_pad])
    ax.set_aspect(1.)
    if title_string is not None:
        plt.title(title_string)

    # Artists
    rod = mpatches.Rectangle((-rod_width / 2, -rod_height / 2),
                             rod_width,
                             rod_height,
                             facecolor='tab:blue',
                             edgecolor='k')
    hub = mpatches.FancyBboxPatch((-hub_width / 2, -hub_height / 2),
                                  hub_width,
                                  hub_height,
                                  facecolor='tab:blue',
                                  edgecolor='k',
                                  boxstyle='Round,pad=0.,rounding_size=0.05')
    axle_left = mpatches.Rectangle((-rod_width / 2, rod_height / 2),
                                   axle_width,
                                   axle_height,
                                   facecolor='tab:blue',
                                   edgecolor='k')
    axle_right = mpatches.Rectangle((rod_width / 2 - axle_width, rod_height / 2),
                                    axle_width,
                                    axle_height,
                                    facecolor='tab:blue',
                                    edgecolor='k')
    prop_left = mpatches.Ellipse(((axle_width - rod_width) / 2, rod_height / 2 + axle_height),
                                 prop_width,
                                 prop_height,
                                 facecolor='tab:gray',
                                 edgecolor='k',
                                 alpha=0.7)
    prop_right = mpatches.Ellipse(((rod_width - axle_width) / 2, rod_height / 2 + axle_height),
                                  prop_width,
                                  prop_height,
                                  facecolor='tab:gray',
                                  edgecolor='k',
                                  alpha=0.7)
    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right)
    for patch in patches:
        ax.add_patch(patch)
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    def animate(k, t, x, y, θ):
        transform = mtransforms.Affine2D().rotate_around(0., 0., θ[k])
        transform += mtransforms.Affine2D().translate(x[k], y[k])
        transform += ax.transData
        for patch in patches:
            patch.set_transform(transform)
        trace.set_data(x[:k + 1], y[:k + 1])
        timestamp.set_text('t = {:.1f} s'.format(t[k]))
        artists = patches + (trace, timestamp)
        return artists

    dt = t[1] - t[0]
    step = max(int(np.floor((1 / 30) / dt)), 1)  # max out at 30Hz for faster rendering
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  t[::step].size,
                                  fargs=(t[::step], x[::step], y[::step], θ[::step]),
                                  interval=step * dt * 1000,
                                  blit=True)
    if display_in_notebook:
        try:
            get_ipython()
            from IPython.display import HTML
            ani = HTML(ani.to_html5_video())
        except (NameError, ImportError):
            raise RuntimeError("`display_in_notebook = True` requires this code to be run in jupyter/colab.")
    return fig, ani