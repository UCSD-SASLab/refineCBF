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


def get_car(ax, x, y, θ, rel_size=1.0, color="tab:blue", alpha=1.0):
    radius = rel_size

    # Figure and axis
    x_min, x_max = np.min(x), np.max(x)
    x_pad = radius + 0.1
    y_min, y_max = np.min(y), np.max(y)
    y_pad = radius + 0.1
    # ax.set_xlim([x_min - x_pad, x_max + x_pad])
    # ax.set_ylim([y_min - y_pad, y_max + y_pad])
    # ax.set_aspect(1.0)

    # Artists
    rod = mpatches.Rectangle(
        (0, -radius/20),
        radius,
        radius/10,
        facecolor=color,
        edgecolor="k",
        alpha=alpha,
    )
    body = mpatches.Ellipse(
        (0 , 0),
        2*radius,
        2*radius,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7 * alpha,
    )
    patches = (rod, body)
    for patch in patches:
        ax.add_patch(patch)

    transform = mtransforms.Affine2D().rotate_around(0.0, 0.0, θ)
    transform += mtransforms.Affine2D().translate(x, y)
    transform += ax.transData
    for patch in patches:
        patch.set_transform(transform)

    artists = patches
    return artists


def animate_planar_car(t, x, y, θ, fig=None, title_string=None):
    """Animate the car system from given position data.
    All arguments are assumed to be 1-D NumPy arrays, where `x`, `y`, and `θ`
    are the degrees of freedom of the car over time `t`.
    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_planar_car
        fig, ani = animate_planar_car(t, x, θ)
        ani.save('planar_car.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    radius = 1

    # Figure and axis
    if fig is None:
        fig, ax = plt.subplots(dpi=100)
        x_min, x_max = np.min(x), np.max(x)
        x_pad = radius + 0.1 * (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        y_pad = radius + 0.1 * (y_max - y_min)
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_min - y_pad, y_max + y_pad])
    else:
        ax = fig.axes[0]

    ax.set_aspect(1.0)
    if title_string is not None:
        plt.title(title_string)

    # Artists
    rod = mpatches.Rectangle(
        (radius/ 2, 0),
        radius,
        radius/10,
        facecolor="tab:blue",
        edgecolor="k",
    )
    body = mpatches.Ellipse(
        (0 , 0),
        radius,
        radius,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7,
    )
    patches = (rod, body)
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


def animate_multi_planar_car(
    t, x, y, θ, fig=None, alphas=None, colors=None, title_string=None):
    """Animate the planar car system from given position data.
    All arguments are assumed to be 1-D NumPy arrays, where `x`, `y`, and `θ`
    are the degrees of freedom of the planar car over time `t`.
    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_planar_car
        fig, ani = animate_planar_car(t, x, θ)
        ani.save('planar_car.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    radius = 1

    # Figure and axis
    if fig is None:
        fig, ax = plt.subplots(dpi=100)
        x_min, x_max = np.min(x), np.max(x)
        x_pad = radius + 0.1 * (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        y_pad = radius + 0.1 * (y_max - y_min)
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_min - y_pad, y_max + y_pad])
    else:
        ax = fig.axes[0]

    ax.set_aspect(1.0)
    if title_string is not None:
        plt.title(title_string)

    # Artists
    rod = mpatches.Rectangle(
        (radius/ 2, 0),
        radius,
        radius/10,
        facecolor="tab:blue",
        edgecolor="k",
    )
    body = mpatches.Ellipse(
        (0 , 0),
        radius,
        radius,
        facecolor="tab:gray",
        edgecolor="k",
        alpha=0.7,
    )
    patches = (rod, body)
    from copy import deepcopy

    num_cars = x.shape[0]
    patches_full = [patches, deepcopy(patches)]
    traces = []
    lines = []
    for car in range(num_cars):
        for patch in patches_full[car]:
            patch.set(alpha=alphas[car])
            ax.add_patch(patch)
            # ax.set(alpha=alphas[car])
        l = ax.plot([], [], "--", linewidth=2, color=colors[car])[0]
        traces.append(l)
        lines.append(l)
    timestamp = ax.text(0.1, 0.9, "", transform=ax.transAxes)

    def animate(k, t, x, y, θ):
        artists = None
        for car in range(num_cars):
            transform = mtransforms.Affine2D().rotate_around(0.0, 0.0, θ[car][k])
            transform += mtransforms.Affine2D().translate(x[car][k], y[car][k])
            transform += ax.transData
            for patch in patches_full[car]:
                patch.set_transform(transform)
            trace = traces[car]
            trace.set_data(x[car][: k + 1], y[car][: k + 1])
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
