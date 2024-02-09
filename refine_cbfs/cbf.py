import hj_reachability as hj
import numpy as np
import jax.numpy as jnp
from cbf_opt.cbf import CBF, ControlAffineCBF
from cbf_opt.dynamics import Dynamics, ControlAffineDynamics
from tqdm import tqdm
from scipy.interpolate import interp1d


class TabularCBF(CBF):
    """
    Provides a tabularized implementation of a CBF to interface with a spatially
    -discretized value function (e.g. from hj_reachability).
    Interfaces with `cbf_opt` and can be used to spatially discretize
    an existing val fct.
    """

    def __init__(self, dynamics: Dynamics, **kwargs) -> None:
        """Initialize a TabularCBF.

        Args:
            dynamics (Dynamics): cbf_opt dynamics object
            grid (hj.grid.Grid): Grid of shape (n1, n2, ..., nk), with k the state dimension.
            params (dict, optional): Parameters for the CBF, e.g. discount rate. Defaults to dict().
        """
        self.grid = kwargs.get("grid")
        assert isinstance(
            self.grid, hj.Grid
        )  # FIXME: Pass in grid as argument, without having weird things happen in super().__init__ --> see below
        self.grid_states_np = np.array(self.grid.states)
        self.grid_shape = self.grid.shape
        self._vf_table = None
        self.orig_cbf = None
        self._grad_vf_table = None
        super().__init__(dynamics, **kwargs)

    def clip_state(self, state):
        return np.clip(state, np.array(self.grid.domain.lo) + 0.01, np.array(self.grid.domain.hi) - 0.01)

    def vf(self, state: np.ndarray, time: float):
        """_summary_

        Args:
            state (np.ndarray): current state (n,) # TODO: Make batched work too
            time (float): current simulation timestamp

        Returns:
            float: Value function at current state, time # TODO: check float or (1,)
        """
        assert self.vf_table is not None, "Requires instantiation of vf_table"
        if state.ndim == 1:
            state_i = self.clip_state(state)
            return self.grid.interpolate(self.vf_table, state_i)
        else:
            vf = np.zeros(state.shape[0])
            for i in range(state.shape[0]):
                state_i = self.clip_state(state[i])
                vf[i] = self.grid.interpolate(self.vf_table, state_i)
            return vf

    def _grad_vf(self, state, time):
        if state.ndim == 1:
            state_i = self.clip_state(state)
            return self.grid.interpolate(self._grad_vf_table, state_i)
        else:
            grad_vf = np.zeros_like(state)
            for i in range(state.shape[0]):
                state_i = self.clip_state(state[i])
                grad_vf[i] = self.grid.interpolate(self._grad_vf_table, state_i)
        return grad_vf

    @property
    def vf_table(self):
        return self._vf_table

    @vf_table.setter
    def vf_table(self, value):
        self._vf_table = value
        self._grad_vf_table = np.array(self.grid.grad_values(self._vf_table))

    def tabularize_cbf(self, orig_cbf: CBF, time=0.0):
        """
        Tabularizes a control-affine CBF.
        """
        self.orig_cbf = orig_cbf
        assert isinstance(orig_cbf, CBF)
        assert self.orig_cbf.dynamics == self.dynamics
        self.vf_table = np.array(self.orig_cbf.vf(self.grid.states, time))

    def get_cbf_cond_table(self, gamma):
        dV = hj.utils.multivmap(
            lambda state, value, grad_value: self.dynamics.hamiltonian(state, 0.0, value, grad_value),
            np.arange(self.grid.ndim),
        )(self.grid.states, self.vf_table, self._grad_vf_table)

        return dV + gamma * self.vf_table


class TabularControlAffineCBF(ControlAffineCBF, TabularCBF):
    def __init__(self, dynamics: ControlAffineDynamics, **kwargs) -> None:
        super().__init__(dynamics, **kwargs)


class TabularTVControlAffineCBF(TabularControlAffineCBF):
    from scipy.interpolate import interp1d

    @property
    def vf_table(self):
        return self._vf_table

    def set_vf_table(self, times, values):
        times_mod = np.concatenate([times, [times[-1] + (times[-1] - times[-2])]])
        if isinstance(values, interp1d):
            self._vf_table = values
            grad_vfs = []
            for time in tqdm(times):
                grad_vfs.append(np.array(self.grid.grad_values(self.vf_table(time))))
            grad_vfs.append(
                np.array(self.grid.grad_values(self.vf_table(times[-1])))
            )  # Twice for last timestep to control extrapolation
        else:
            values_mod = np.concatenate(
                [values, values[-1][None, :]], axis=0
            )  # Twice for last timestep to control extrapolation
            self._vf_table = interp1d(times_mod, values_mod, axis=0, fill_value="extrapolate")
            grad_vfs = []
            for value in tqdm(values_mod):
                grad_vfs.append(np.array(self.grid.grad_values(value)))
        grad_vfs = np.array(grad_vfs)
        self._grad_vf_table = interp1d(times_mod, grad_vfs, axis=0, fill_value="extrapolate")

    def vf(self, state, time):
        assert self.vf_table is not None and isinstance(self.vf_table, interp1d)
        if state.ndim == 1:
            time = time.item() if hasattr(time, "item") else time  # If time is an array, convert to float
            state_i = self.clip_state(state)
            return self.grid.interpolate(self.vf_table(time), state_i)
        else:
            time = jnp.array(time) if isinstance(time, float) else time  # If time is a float, convert to array
            vf = np.zeros(state.shape[0])
            for i in range(state.shape[0]):
                state_i = self.clip_state(state[i])
                vf[i] = self.grid.interpolate(self.vf_table(time[i]), state_i)
            return vf

    def _grad_vf(self, state, time):
        if state.ndim == 1:
            time = time.item() if hasattr(time, "item") else time  # If time is an array, convert to float
            state_i = self.clip_state(state)
            return self.grid.interpolate(self._grad_vf_table(time), state_i)
        else:
            time = jnp.array(time) if isinstance(time, float) else time  # If time is a float, convert to array
            grad_vf = np.zeros_like(state)
            for i in range(state.shape[0]):
                state_i = self.clip_state(state[i])
                grad_vf[i] = self.grid.interpolate(self._grad_vf_table(time[i]), state_i)
            return grad_vf
