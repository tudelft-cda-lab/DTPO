from gymnax.environments.classic_control.pendulum import Pendulum, EnvParams, EnvState

from gymnax.environments import spaces

import jax

from typing import Tuple, Optional

import chex


class PendulumBangBang(Pendulum):
    def __init__(self):
        super().__init__()

        self.feature_names = ["cos theta", "sin theta", "theta dot"]

        self.action_names = ["left", "right"]

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v1-BangBang"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Discrete(2)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        action_space = super().action_space()
        continuous_action = jax.lax.switch(
            action,
            (
                lambda: action_space.low,
                lambda: action_space.high,
            ),
        )
        return super().step_env(key, state, continuous_action, params)
