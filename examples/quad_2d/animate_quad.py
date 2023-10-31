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


def get_drone(ax, x, y, θ, rel_size=1.0, height_scale=1.0, color="tab:blue", alpha=1.0):
    rod_width = 2.0 * rel_size
    rod_height = 0.15 * rel_size * height_scale
    axle_height = 0.2 * rel_size * height_scale
    axle_width = 0.05 * rel_size
    prop_width = 0.5 * rod_width
    prop_height = 1.5 * rod_height
    hub_width = 0.3 * rod_width
    hub_height = 2.5 * rod_height

    # Figure and axis
    x_min, x_max = np.min(x), np.max(x)
    x_pad = (rod_width + prop_width) / 2 + 0.1
    y_min, y_max = np.min(y), np.max(y)
    y_pad = (rod_width + prop_width) / 2 + 0.1
    # ax.set_xlim([x_min - x_pad, x_max + x_pad])
    # ax.set_ylim([y_min - y_pad, y_max + y_pad])
    # ax.set_aspect(1.0)

    # Artists
    rod = mpatches.Rectangle(
        (-rod_width / 2, -rod_height / 2),
        rod_width,
        rod_height,
        facecolor=color,
        edgecolor="k",
        alpha=alpha,
    )
    hub = mpatches.FancyBboxPatch(
        (-hub_width / 2, -hub_height / 2),
        hub_width,
        hub_height,
        facecolor=color,
        edgecolor="k",
        boxstyle="Round,pad=0.,rounding_size=0.05",
        alpha=alpha,
    )
    axle_left = mpatches.Rectangle(
        (-rod_width / 2, rod_height / 2),
        axle_width,
        axle_height,
        facecolor=color,
        edgecolor="k",
        alpha=alpha,
    )
    axle_right = mpatches.Rectangle(
        (rod_width / 2 - axle_width, rod_height / 2),
        axle_width,
        axle_height,
        facecolor=color,
        edgecolor="k",
        alpha=alpha,
    )
    prop_left = mpatches.Ellipse(
        ((axle_width - rod_width) / 2, rod_height / 2 + axle_height),
        prop_width,
        prop_height,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7 * alpha,
    )
    prop_right = mpatches.Ellipse(
        ((rod_width - axle_width) / 2, rod_height / 2 + axle_height),
        prop_width,
        prop_height,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7 * alpha,
    )
    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right)
    for patch in patches:
        ax.add_patch(patch)

    transform = mtransforms.Affine2D().rotate_around(0.0, 0.0, θ)
    transform += mtransforms.Affine2D().translate(x, y)
    transform += ax.transData
    for patch in patches:
        patch.set_transform(transform)

    artists = patches
    return artists


def animate_planar_quad(t, x, y, θ, fig=None, title_string=None):
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
    rod_width = 2.0
    rod_height = 0.15
    axle_height = 0.2
    axle_width = 0.05
    prop_width = 0.5 * rod_width
    prop_height = 1.5 * rod_height
    hub_width = 0.3 * rod_width
    hub_height = 2.5 * rod_height

    # Figure and axis
    if fig is None:
        fig, ax = plt.subplots(dpi=100)
        x_min, x_max = np.min(x), np.max(x)
        x_pad = (rod_width + prop_width) / 2 + 0.1 * (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        y_pad = (rod_width + prop_width) / 2 + 0.1 * (y_max - y_min)
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_min - y_pad, y_max + y_pad])
    else:
        ax = fig.axes[0]

    ax.set_aspect(1.0)
    if title_string is not None:
        plt.title(title_string)

    # Artists
    rod = mpatches.Rectangle(
        (-rod_width / 2, -rod_height / 2),
        rod_width,
        rod_height,
        facecolor="tab:blue",
        edgecolor="k",
    )
    hub = mpatches.FancyBboxPatch(
        (-hub_width / 2, -hub_height / 2),
        hub_width,
        hub_height,
        facecolor="tab:blue",
        edgecolor="k",
        boxstyle="Round,pad=0.,rounding_size=0.05",
    )
    axle_left = mpatches.Rectangle(
        (-rod_width / 2, rod_height / 2),
        axle_width,
        axle_height,
        facecolor="tab:blue",
        edgecolor="k",
    )
    axle_right = mpatches.Rectangle(
        (rod_width / 2 - axle_width, rod_height / 2),
        axle_width,
        axle_height,
        facecolor="tab:blue",
        edgecolor="k",
    )
    prop_left = mpatches.Ellipse(
        ((axle_width - rod_width) / 2, rod_height / 2 + axle_height),
        prop_width,
        prop_height,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7,
    )
    prop_right = mpatches.Ellipse(
        ((rod_width - axle_width) / 2, rod_height / 2 + axle_height),
        prop_width,
        prop_height,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7,
    )
    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right)
    for patch in patches:
        ax.add_patch(patch)
    trace = ax.plot([], [], "--", linewidth=2, color="tab:orange")[0]
    timestamp = ax.text(0.1, 0.9, "", transform=ax.transAxes)

    def animate(k, t, x, y, θ):
        transform = mtransforms.Affine2D().rotate_around(0.0, 0.0, θ[k])
        transform += mtransforms.Affine2D().translate(x[k], y[k])
        transform += ax.transData
        for patch in patches:
            patch.set_transform(transform)
        trace.set_data(x[: k + 1], y[: k + 1])
        timestamp.set_text("t = {:.1f} s".format(t[k]))
        artists = patches + (trace, timestamp)
        return artists

    dt = t[1] - t[0]
    step = max(int(np.floor((1 / 30) / dt)), 1)  # max out at 30Hz for faster rendering
    ani = animation.FuncAnimation(
        fig,
        animate,
        t[::step].size,
        fargs=(t[::step], x[::step], y[::step], θ[::step]),
        interval=step * dt * 1000,
        blit=True,
    )

    return fig, ani


def animate_multi_planar_quad(
    t, x, y, θ, fig=None, alphas=None, colors=None, title_string=None):
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
    rod_width = 2.0
    rod_height = 0.15
    axle_height = 0.2
    axle_width = 0.05
    prop_width = 0.5 * rod_width
    prop_height = 1.5 * rod_height
    hub_width = 0.3 * rod_width
    hub_height = 2.5 * rod_height

    # Figure and axis
    if fig is None:
        fig, ax = plt.subplots(dpi=100)
        x_min, x_max = np.min(x), np.max(x)
        x_pad = (rod_width + prop_width) / 2 + 0.1 * (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        y_pad = (rod_width + prop_width) / 2 + 0.1 * (y_max - y_min)
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_min - y_pad, y_max + y_pad])
    else:
        ax = fig.axes[0]

    ax.set_aspect(1.0)
    if title_string is not None:
        plt.title(title_string)

    # Artists
    rod = mpatches.Rectangle(
        (-rod_width / 2, -rod_height / 2),
        rod_width,
        rod_height,
        facecolor="tab:blue",
        edgecolor="k",
    )
    hub = mpatches.FancyBboxPatch(
        (-hub_width / 2, -hub_height / 2),
        hub_width,
        hub_height,
        facecolor="tab:blue",
        edgecolor="k",
        boxstyle="Round,pad=0.,rounding_size=0.05",
    )
    axle_left = mpatches.Rectangle(
        (-rod_width / 2, rod_height / 2),
        axle_width,
        axle_height,
        facecolor="tab:blue",
        edgecolor="k",
    )
    axle_right = mpatches.Rectangle(
        (rod_width / 2 - axle_width, rod_height / 2),
        axle_width,
        axle_height,
        facecolor="tab:blue",
        edgecolor="k",
    )
    prop_left = mpatches.Ellipse(
        ((axle_width - rod_width) / 2, rod_height / 2 + axle_height),
        prop_width,
        prop_height,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7,
    )
    prop_right = mpatches.Ellipse(
        ((rod_width - axle_width) / 2, rod_height / 2 + axle_height),
        prop_width,
        prop_height,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7,
    )
    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right)
    from copy import deepcopy

    num_quads = x.shape[0]
    patches_full = [patches, deepcopy(patches)]
    traces = []
    lines = []
    for quad in range(num_quads):
        for patch in patches_full[quad]:
            patch.set(alpha=alphas[quad])
            ax.add_patch(patch)
            # ax.set(alpha=alphas[quad])
        l = ax.plot([], [], "--", linewidth=2, color=colors[quad])[0]
        traces.append(l)
        lines.append(l)
    timestamp = ax.text(0.1, 0.9, "", transform=ax.transAxes)

    def animate(k, t, x, y, θ):
        artists = None
        for quad in range(num_quads):
            transform = mtransforms.Affine2D().rotate_around(0.0, 0.0, θ[quad][k])
            transform += mtransforms.Affine2D().translate(x[quad][k], y[quad][k])
            transform += ax.transData
            for patch in patches_full[quad]:
                patch.set_transform(transform)
            trace = traces[quad]
            trace.set_data(x[quad][: k + 1], y[quad][: k + 1])
            if artists is None:
                artists = patches
            else:
                artists += patches
            artists += (trace,)
        timestamp.set_text("t = {:.1f} s".format(t[k]))
        artists += (timestamp,)
        return artists

    dt = t[1] - t[0]
    step = max(int(np.floor((1 / 30) / dt)), 1)  # max out at 30Hz for faster rendering
    ani = animation.FuncAnimation(
        fig,
        animate,
        t[::step].size,
        fargs=(t[::step], x[:, ::step], y[:, ::step], θ[:, ::step]),
        interval=step * dt * 1000,
        blit=True,
    )
    return fig, ani
