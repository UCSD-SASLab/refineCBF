from cbf_opt.dynamics import ControlAffineDynamics
import jax.numpy as jnp
import hj_reachability as hj


class HJControlAffineDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """Provides portability between cbf_opt and hj_reachability Dynamics definitions"""

    def __init__(self, dynamics, **kwargs):
        assert isinstance(dynamics, ControlAffineDynamics)

        self.dynamics = dynamics
        control_mode = kwargs.get("control_mode", "max")
        disturbance_mode = kwargs.get("disturbance_mode", "min")
        control_space = kwargs.get(
            "control_space",
            hj.sets.Box(
                jnp.atleast_1d(-jnp.ones(self.dynamics.control_dims)),
                jnp.atleast_1d(jnp.ones(self.dynamics.control_dims)),
            ),
        )
        disturbance_space = kwargs.get(
            "disturbance_space",
            hj.sets.Box(
                jnp.atleast_1d(-jnp.zeros(self.dynamics.disturbance_dims)),
                jnp.atleast_1d(jnp.zeros(self.dynamics.disturbance_dims)),
            ),
        )
        super().__init__(
            control_mode=control_mode,
            disturbance_mode=disturbance_mode,
            control_space=control_space,
            disturbance_space=disturbance_space,
        )

    def open_loop_dynamics(self, state, time=None):
        return jnp.array(self.dynamics.open_loop_dynamics(state, time))

    def control_jacobian(self, state, time=None):
        return jnp.array(self.dynamics.control_matrix(state, time))

    def disturbance_jacobian(self, state, time=None):
        return jnp.array(self.dynamics.disturbance_jacobian(state, time))

    def optimal_control_state(self, state, time, grad_value):
        return self.optimal_control_and_disturbance(state, time, grad_value)[0]
