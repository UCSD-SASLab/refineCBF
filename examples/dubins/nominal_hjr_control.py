import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


class NominalControlHJ:
    def __init__(self, dyn, grid, **kwargs):
        self.dyn = dyn
        self.grid = grid
        self.final_time = kwargs.get("final_time", -50.0)
        self.time_intervals = kwargs.get("time_intervals", 201)
        self.times = jnp.linspace(0, self.final_time, self.time_intervals)
        self.value_pp = kwargs.get("value_pp", lambda target: lambda t, x: jnp.maximum(x, target))
        self.solver_accuracy = kwargs.get("solver_accuracy", "medium")
        self.pad = kwargs.get("padding", 0.1 * jnp.ones(self.grid.ndim))
        self.tv_vf = None

    def _get_target_function(self, target):
        return lambda x: jnp.min(
            jnp.array(
                [
                    x[0] - target[0] + self.pad[0],
                    target[0] + self.pad[0] - x[0],
                    x[1] - target[1] + self.pad[1],
                    target[1] + self.pad[1] - x[1],
                ]
            )
        )
        # return lambda x: jnp.min(jnp.array([x - target + self.pad, target + self.pad - x]))

    def solve(self, target):
        target_f = self._get_target_function(target)
        init_values = hj.utils.multivmap(target_f, jnp.arange(self.grid.ndim))(self.grid.states)
        solver_settings = hj.SolverSettings.with_accuracy(
            self.solver_accuracy, value_postprocessor=self.value_pp(init_values)
        )
        self.tv_vf = hj.solve(solver_settings, self.dyn, self.grid, self.times, init_values)

    def get_reachable_set(self, time):
        idx = (jnp.abs(self.times - time)).argmin()
        return self.tv_vf[idx]

    def _line_search(self, x):
        upper = self.time_intervals
        lower = 0

        while upper > lower:
            mid = (upper + lower) // 2
            val = self.grid.interpolate(self.tv_vf[mid], x)

            lower, upper = jnp.where(
                val > 1e-4, jnp.array([lower, mid]), jnp.array([mid + 1, upper])
            )
        return mid

    def get_nominal_control(self, x):
        idx = self._line_search(x)
        grad_vals = self.grid.grad_values(self.tv_vf[idx])
        grad_val = self.grid.interpolate(grad_vals, x)
        return self.dyn.optimal_control_state(x, 0.0, grad_val)

    def get_nominal_control_table(self):
        return hj.utils.multivmap(self.get_nominal_control, jnp.arange(self.grid.ndim))(
            self.grid.states
        )

    def get_nominal_controller(self, target):
        self.solve(target)
        table = self.get_nominal_control_table()
        return lambda x, t: self.grid.interpolate(table, x)


class NominalControlHJNP:
    def __init__(self, dyn, grid, **kwargs):
        self.dyn = dyn
        self.grid = grid
        self.final_time = kwargs.get("final_time", -50.0)
        self.time_intervals = kwargs.get("time_intervals", 201)
        self.times = jnp.linspace(0, self.final_time, self.time_intervals)
        self.value_pp = kwargs.get("value_pp", lambda target: lambda t, x: jnp.maximum(x, target))
        self.solver_accuracy = kwargs.get("solver_accuracy", "medium")
        self.pad = kwargs.get("padding", 0.1 * jnp.ones(self.grid.ndim))
        self.tv_vf = None
        self.target = kwargs.get("target")

    def _get_target_function(self, target):
        return lambda x: jnp.min(
            jnp.array(
                [
                    x[0] - target[0] + self.pad[0],
                    target[0] + self.pad[0] - x[0],
                    x[1] - target[1] + self.pad[1],
                    target[1] + self.pad[1] - x[1],
                ]
            )
        )

    def solve(self, target = None):
        self.target = target if target is not None else self.target
        assert self.target.shape[0] == self.grid.ndim, "Target has to match dimension of grid + dynamics"
        target_f = self._get_target_function(self.target)
        init_values = hj.utils.multivmap(target_f, jnp.arange(self.grid.ndim))(self.grid.states)
        solver_settings = hj.SolverSettings.with_accuracy(
            self.solver_accuracy, value_postprocessor=self.value_pp(init_values)
        )
        self.tv_vf = hj.solve(solver_settings, self.dyn, self.grid, self.times, init_values)

    def get_reachable_set(self, time):
        idx = (jnp.abs(self.times - time)).argmin()
        return self.tv_vf[idx]

    def _line_search(self, x):
        upper = self.time_intervals
        lower = 0

        while upper > lower:
            mid = (upper + lower) // 2
            val = self.grid.interpolate(self.tv_vf[mid], x)

            lower, upper = jnp.where(
                val > 1e-4, jnp.array([lower, mid]), jnp.array([mid + 1, upper])
            )
        return mid

    def get_nominal_control(self, x, t):
        if x.ndim == 1:
            x = x[None, :]
        opt_ctrl = np.zeros((x.shape[0], self.dyn.dynamics.control_dims))
        for i in range(x.shape[0]):
            idx = self._line_search(x[i])
            grad_vals = self.grid.grad_values(self.tv_vf[idx])
            grad_val = self.grid.interpolate(grad_vals, x[i])
            opt_ctrl[i] = self.dyn.optimal_control_state(x[i], t, grad_val)
        return opt_ctrl

    def get_nominal_control_table(self):
        table = np.zeros_like(self.grid.states)[..., np.newaxis]
        self.grid_shape = self.grid.shape
        self.grid_states_np = np.array(self.grid.states)
        for i in tqdm(range(self.grid_shape[0])):
            if self.grid.ndim == 1:
                table[i] = self.get_nominal_control(self.grid_states_np[i])
            else:
                for j in range(self.grid_shape[1]):
                    if self.grid.ndim == 2:
                        table[i, j] = self.get_nominal_control(self.grid_states_np[i, j])
                    else:
                        for k in range(self.grid_shape[2]):
                            if self.grid.ndim == 3:
                                table[i, j, k] = self.get_nominal_control(
                                    self.grid_states_np[i, j, k]
                                )
        # return hj.utils.multivmap(self.get_nominal_control, jnp.arange(self.grid.ndim))(
        #     self.grid.states
        # )
        return table

    def get_nominal_controller(self, target):
        self.solve(target)
        table = self.get_nominal_control_table()
        return lambda x, t: self.grid.interpolate(table, x)


class NominalPolicy:
    """Decorator for HJ nominal controller to interface with experiment_wrapper."""

    def __init__(self, opt_ctrl):
        self.opt_ctrl = opt_ctrl

    def __call__(self, x, t):
        return self.opt_ctrl.get_nominal_control(x, t)

    def save_measurements(self, state, control, time):
        return {"dist_to_goal": np.linalg.norm(state[..., :2] - self.opt_ctrl.target[:2], axis=-1)}
