import hj_reachability as hj
import numpy as np
from cbf_opt.cbf import CBF, ControlAffineCBF
from cbf_opt.dynamics import Dynamics, ControlAffineDynamics


class TabularCBF(CBF):
    """
    Provides a tabularized implementation of a CBF to interface with a spatially-discretized value function
    (e.g. from hj_reachability). Interfaces with `cbf_opt` and can be used to spatially discretize an existing val fct.
    """

    def __init__(
        self, dynamics: Dynamics, params: dict = dict(), test: bool = False, **kwargs
    ) -> None:
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
        self.vf_table = None
        self.orig_cbf = None
        self.grad_vf_table = None
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
        if self.grad_vf_table is None:
            self.grad_vf_table = self.grid.grad_values(self.vf_table)

        if state.ndim == 1:
            return self.grid.interpolate(self.grad_vf_table, state)
        else:
            grad_vf = np.zeros_like(state)
            for i in range(state.shape[0]):
                grad_vf[i] = self.grid.interpolate(self.grad_vf_table, state[i])
        return grad_vf

    # def is_unsafe(self, state: np.ndarray, time: float):
    #     """_summary_

    #     Args:
    #         state (np.ndarray): current state (n,)

    #     """
    #     is_unsafe_array = np.zeros(state.shape[0], dtype=bool)
    #     for i in range(state.shape[0]):
    #         is_unsafe_array[i] = super().is_unsafe(state[i], time)
    #     return is_unsafe_array

    def tabularize_cbf(self, orig_cbf: CBF, time=0.0):
        """
        Tabularizes a control-affine CBF.
        """
        self.orig_cbf = orig_cbf
        assert isinstance(orig_cbf, CBF)
        assert self.orig_cbf.dynamics == self.dynamics
        self.vf_table = np.array(self.orig_cbf.vf(self.grid.states, time))


class TabularControlAffineCBF(ControlAffineCBF, TabularCBF):
    def __init__(
        self, dynamics: ControlAffineDynamics, params: dict = dict(), test: bool = False, **kwargs
    ) -> None:
        super().__init__(dynamics, params, test, **kwargs)
