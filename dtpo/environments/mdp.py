from typing import Tuple, Optional

import chex

from flax import struct

from gymnax.environments import environment, spaces

import jax.numpy as jnp
import jax
from jax import lax


@struct.dataclass
class MarkovDecisionProcessParams:
    """Encodes a discrete time, discrete action MDP using matrices."""

    trans_probs: chex.Array
    rewards: chex.Array
    initial_state_p: chex.Array
    observations: chex.Array
    max_steps_in_episode: int = 1000


@struct.dataclass
class EnvState:
    state_index: int
    time: int


class MdpEnv(environment.Environment):
    @property
    def default_params(self) -> MarkovDecisionProcessParams:
        """Default environment parameters for Navigation3D."""
        return self.default_params_

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: MarkovDecisionProcessParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # Shape: state, next_state, action
        trans_probs = params.trans_probs[state.state_index, :, action]

        next_state_index = jax.random.choice(
            key, params.trans_probs.shape[0], p=trans_probs
        )

        reward = params.rewards[state.state_index, next_state_index, action]

        next_state = EnvState(
            state_index=next_state_index,
            time=state.time + 1,
        )

        done = self.is_terminal(next_state, params)

        observation = params.observations[next_state_index]

        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(next_state),
            reward,
            done,
            {"discount": self.discount(next_state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: MarkovDecisionProcessParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        if params is None:
            params = self.default_params
        state_index = jax.random.choice(
            key, params.initial_state_p.shape[0], p=params.initial_state_p
        )
        observation = params.observations[state_index]
        state = EnvState(state_index=state_index, time=0)
        return observation, state

    def get_obs(self, state: EnvState) -> chex.Array:
        raise NotImplementedError()

    def is_terminal(self, state: EnvState, params: MarkovDecisionProcessParams) -> bool:
        all_transitions_to_0_reward_state = jnp.logical_and(
            jnp.all(params.trans_probs[state.state_index, state.state_index, :] == 1),
            jnp.all(params.rewards[state.state_index, state.state_index, :] == 0),
        )
        max_steps_reached = state.time == params.max_steps_in_episode

        return jnp.logical_or(all_transitions_to_0_reward_state, max_steps_reached)

    @property
    def name(self) -> str:
        """Environment name."""
        return self.name_

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_names)

    def action_space(
        self, params: Optional[MarkovDecisionProcessParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: MarkovDecisionProcessParams) -> spaces.Box:
        """Observation space of the environment."""
        low = params.observations.min(axis=0)
        high = params.observations.max(axis=0)
        return spaces.Box(
            low, high, shape=(params.observations.shape[1],), dtype=jnp.float32
        )

    def state_space(self, params: MarkovDecisionProcessParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "state_index": spaces.Discrete(params.observations.shape[0]),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def check_mdp(mdp: MarkovDecisionProcessParams, check_jax_array_type=True):
    """
    Asserts that the shapes of the arrays inside the MDP are all valid and compatible.
    """
    if check_jax_array_type:
        assert isinstance(mdp.trans_probs, jax.numpy.ndarray)
        assert isinstance(mdp.rewards, jax.numpy.ndarray)
        assert isinstance(mdp.initial_state_p, jax.numpy.ndarray)
        assert isinstance(mdp.observations, jax.numpy.ndarray)

    # trans_probs and rewards have axes: state, next_state, action
    # observations has axes: state, feature
    assert len(mdp.trans_probs.shape) == 3
    assert jnp.all(jnp.isclose(mdp.trans_probs.sum(axis=1), 1))
    assert jnp.all((mdp.trans_probs >= 0) & (mdp.trans_probs <= 1))

    n_states_ = mdp.trans_probs.shape[0]

    assert mdp.trans_probs.shape[1] == n_states_
    assert mdp.trans_probs.shape == mdp.rewards.shape
    assert mdp.initial_state_p.shape == (n_states_,)
    assert len(mdp.observations.shape) == 2
    assert mdp.observations.shape[0] == n_states_


def remove_unreachable_states_mdp(mdp: MarkovDecisionProcessParams):
    """
    Returns a new MDP with unreachable states removed.

    Reachable states are determined using a depth first search to find all states
    reachable from the non-zero probability initial states.
    """
    visited = set()
    stack = [int(x) for x in jnp.nonzero(mdp.initial_state_p)[0]]
    state_state_probs = mdp.trans_probs.sum(axis=2)
    while stack:
        state = stack.pop()
        visited.add(state)
        for next_state in jnp.nonzero(state_state_probs[state])[0]:
            next_state = int(next_state)
            if next_state not in visited:
                stack.append(next_state)

    if len(visited) == mdp.trans_probs.shape[0]:
        return mdp

    print("Removed states:", mdp.trans_probs.shape[0] - len(visited))

    # Create a new MDP without all the unreachable states
    visited = jnp.array(list(visited))
    new_mdp = MarkovDecisionProcessParams(
        trans_probs=mdp.trans_probs[visited][:, visited],
        rewards=mdp.rewards[visited][:, visited],
        initial_state_p=mdp.initial_state_p[visited],
        observations=mdp.observations[visited],
    )

    return new_mdp
