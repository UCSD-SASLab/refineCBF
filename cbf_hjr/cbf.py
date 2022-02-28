import hj_reachability as hj
import numpy as np
from tqdm import tqdm
from cbf_opt.cbf import CBF, ControlAffineCBF
from cbf_opt.dynamics import Dynamics, ControlAffineDynamics


class TabularCBF(CBF):
    """
    Provides a tabularized implementation of a CBF to interface with a spatially-discretized value function (e.g. from hj_reachability). Interfaces with `cbf_opt` and can be used to spatially discretize an existing value function.
    """

    def __init__(self, dynamics: Dynamics, params: dict = dict(), **kwargs) -> None:
        """Initialize a TabularCBF.

        Args:
            dynamics (Dynamics): cbf_opt dynamics object
            grid (hj.grid.Grid): Grid of shape (n1, n2, ..., nk), with k the state dimension.
            params (dict, optional): Parameters for the CBF, e.g. discount rate. Defaults to dict().
        """
        super().__init__(dynamics, params, **kwargs)
        self.grid = kwargs.get("grid")
        assert (
            self.grid is not None
        ), "Requires grid"  # FIXME: assert isinstance(self.grid, hj.grid.Grid)
        self.grid_states_np = np.array(self.grid.states)
        self.grid_shape = self.grid.shape
        self.vf_table = None
        self.orig_cbf = None
        self.grad_vf_table = None

    def vf(self, state: np.ndarray, time: float):
        """_summary_

        Args:
            state (np.ndarray): current state (n,) # TODO: Make batched work too
            time (float): current simulation timestamp

        Returns:
            float: Value function at current state, time # TODO: check float or (1,)
        """
        assert self.vf_table is not None, "Requires instantiation of vf_table"
        return np.array(self.grid.interpolate(self.vf_table, state))

    def _grad_vf(self, state, time):
        if self.grad_vf_table is None:
            self.grad_vf_table = self.grid.grad_values(self.vf_table)

        return np.array(self.grid.interpolate(self.grad_vf_table, state))

    # TOTEST
    def tabularize_cbf(self, orig_cbf: CBF, time=0.0):
        """
        Tabularizes a control-affine CBF.
        """
        # TODO: test alternative
        # self.orig_cbf = orig_cbf
        # assert isinstance(orig_cbf, CBF)
        # assert self.orig_cbf.dynamics == self.dynamics
        # self.vf_table = np.array(self.orig_cbf.vf(jnp.moveaxis(self.grid.states, -1, 0), time))

        self.orig_cbf = orig_cbf
        assert isinstance(self.orig_cbf, CBF)
        assert self.orig_cbf.dynamics == self.dynamics

        self.vf_table = np.zeros(self.grid.shape)

        for i in tqdm(range(self.grid_shape[0])):
            if self.grid.ndim == 1:
                self.vf_table[i] = self.orig_cbf.vf(self.grid_states_np[i], time)
            else:
                for j in range(self.grid_shape[1]):
                    if self.grid.ndim == 2:
                        self.vf_table[i, j] = self.orig_cbf.vf(self.grid_states_np[i, j], time)
                    else:
                        for k in range(self.grid_shape[2]):
                            if self.grid.ndim == 3:
                                self.vf_table[i, j, k] = self.orig_cbf.vf(
                                    self.grid_states_np[i, j, k], time
                                )
                            else:
                                for l in range(self.grid_shape[3]):
                                    if self.grid.ndim == 4:
                                        self.vf_table[i, j, k, l] = self.orig_cbf.vf(
                                            self.grid_states_np[i, j, k, l], time
                                        )
                                    else:
                                        for m in range(self.grid_shape[4]):
                                            if self.grid.ndim == 5:
                                                self.vf_table[i, j, k, l, m] = self.orig_cbf.vf(
                                                    self.grid_states_np[i, j, k, l, m], time
                                                )
                                            else:
                                                for n in range(self.grid_shape[5]):
                                                    if self.grid.ndim == 6:
                                                        self.vf_table[
                                                            i, j, k, l, m, n
                                                        ] = self.orig_cbf.vf(
                                                            self.grid_states_np[i, j, k, l, m, n],
                                                            time,
                                                        )
                                                    else:
                                                        raise NotImplementedError(
                                                            "Only up to 6 dimensions supported"
                                                        )


class TabularControlAffineCBF(ControlAffineCBF, TabularCBF):
    def __init__(self, dynamics: ControlAffineDynamics, params: dict = dict(), **kwargs) -> None:
        super().__init__(dynamics, params, **kwargs)
