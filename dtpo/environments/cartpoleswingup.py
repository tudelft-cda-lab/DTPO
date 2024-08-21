"""
discrete Cart pole swing-up:
Adapted from:
hardmaru - https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py
Changes:
* Discrete number of actions. Each action gives provides a force in a certain direction of fixed magnitude, as in
  CartPole.
* The reward function has been adapted slightly, to provide 0 reward when the pole is below horizontal

Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py
hardmaru's changes:
More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""

import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

max_float = jnp.finfo(jnp.float32).max


@struct.dataclass
class EnvState:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time: int


@struct.dataclass
class EnvParams:
    g: float = 9.82  # gravity
    m_c: float = 0.5  # cart mass
    m_p: float = 0.5  # pendulum mass
    l: float = 0.6  # pole's length
    force_mag: float = 10.0
    dt: float = 0.01  # seconds between state updates
    b: float = 0.1  # friction coefficient
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 1000


class CartPoleSwingUp(environment.Environment):
    """
    JAX Compatible version of CartPoleSwingUp. Adapted from:
    https://github.com/TTitcombe/CartPoleSwingUp/tree/master
    """

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def __init__(self):
        super().__init__()

        self.feature_names = [
            "cart position",
            "cart velocity",
            "cos(pole angle)",
            "sin(pole angle)",
            "pole angular velocity",
        ]
        self.action_names = ["left", "right"]

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""

        force = jnp.where(action == 1, params.force_mag, -params.force_mag)

        x = state.x
        x_dot = state.x_dot
        theta = state.theta
        theta_dot = state.theta_dot
        t = state.time

        s = jnp.sin(theta)
        c = jnp.cos(theta)

        total_m = params.m_p + params.m_c
        m_p_l = params.m_p * params.l

        xdot_update = (
            -2 * m_p_l * (theta_dot**2) * s
            + 3 * params.m_p * params.g * s * c
            + 4 * force
            - 4 * params.b * x_dot
        ) / (4 * total_m - 3 * params.m_p * c**2)
        thetadot_update = (
            -3 * m_p_l * (theta_dot**2) * s * c
            + 6 * total_m * params.g * s
            + 6 * (force - params.b * x_dot) * c
        ) / (4 * params.l * total_m - 3 * m_p_l * c**2)
        x = x + x_dot * params.dt
        theta = theta + theta_dot * params.dt
        x_dot = x_dot + xdot_update * params.dt
        theta_dot = theta_dot + thetadot_update * params.dt
        t = t + 1

        new_state = EnvState(x, x_dot, theta, theta_dot, t)

        terminated = jnp.logical_or(x < -params.x_threshold, (x > params.x_threshold))
        truncated = t >= params.max_steps_in_episode

        # Reward_theta is 1 when theta is 0, 0 if between 90 and 270
        reward_theta = jnp.maximum(0, jnp.cos(theta))

        # Reward_x is 0 when cart is at the edge of the screen, 1 when it's in the centre
        reward_x = jnp.cos((x / params.x_threshold) * (jnp.pi / 2.0))

        # [0, 1]
        reward = reward_theta * reward_x

        done = jnp.logical_or(terminated, truncated)

        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, dict]:
        """Reset environment state by sampling initial position."""
        scale = 0.2
        state_vals = (
            jax.random.normal(key=key, shape=(4,))
            + jnp.array([0.0, 0.0, jnp.pi, 0.0]) * scale
        )

        state = EnvState(
            state_vals[0],
            state_vals[1],
            state_vals[2],
            state_vals[3],
            0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.array(
            [
                state.x,
                state.x_dot,
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Episode always terminates after single step - Do not reset though!
        return True

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPoleSwingup"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                max_float,
                max_float,
                max_float,
                max_float,
                max_float,
            ]
        )
        return spaces.Box(-high, high, shape=(5,))

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "g": spaces.Box(0, max_float, ()),
                "m_c": spaces.Box(0, max_float, ()),
                "m_p": spaces.Box(0, max_float, ()),
                "l": spaces.Box(0, max_float, ()),
                "force_mag": spaces.Box(0, max_float, ()),
                "dt": spaces.Box(0, max_float, ()),
                "b": spaces.Box(0, max_float, ()),
                "theta_threshold_radians": spaces.Box(0, max_float, ()),
                "x_threshold": spaces.Box(0, max_float, ()),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
