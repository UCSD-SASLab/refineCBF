import hj_reachability as hj
import numpy as np
from cbf_opt.cbf import CBF, ControlAffineCBF
from cbf_opt.dynamics import Dynamics, ControlAffineDynamics


class TabularCBF(CBF):
    """
    Provides a tabularized implementation of a CBF to interface with a spatially
    -discretized value function (e.g. from hj_reachability).
    Interfaces with `cbf_opt` and can be used to spatially discretize
    an existing val fct.
    """

    def __init__(self, dynamics: Dynamics, params: dict = dict(), test: bool = False, **kwargs) -> None:
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
        super().__init__(dynamics, params, test, **kwargs)

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
            return self.grid.interpolate(self.vf_table, state)
        else:
            vf = np.zeros(state.shape[0])
            for i in range(state.shape[0]):
                vf[i] = self.grid.interpolate(self.vf_table, state[i])
            return vf

    def _grad_vf(self, state, time):
        if state.ndim == 1:
            return self.grid.interpolate(self._grad_vf_table, state)
        else:
            grad_vf = np.zeros_like(state)
            for i in range(state.shape[0]):
                state_i = np.clip(state[i], np.array(self.grid.domain.lo) + 0.01, np.array(self.grid.domain.hi) - 0.01)
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
    def __init__(self, dynamics: ControlAffineDynamics, params: dict = dict(), test: bool = False, **kwargs) -> None:
        super().__init__(dynamics, params, test, **kwargs)
