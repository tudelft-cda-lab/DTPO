import gym

import numpy as np

from .mdp import MarkovDecisionProcessParams, MdpEnv, check_mdp

import jax.numpy as jnp


def _generate_random_map(size=8, p=0.8, random_seed: int = None):
    """Generates a random valid map (one that has a path from start to goal).
    Modified from https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    to make sure the randomness can be seeded.
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    random_state = np.random.RandomState(random_seed)

    while not valid:
        p = min(1, p)
        res = random_state.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]


def generate_env_params(
    map_name="4x4",
    size=None,
    random_seed=None,
    is_slippery=True,
    desc=None,
    max_steps=200,
):
    if map_name == "random":
        if size is None:
            raise ValueError("when using random map size should be defined")

        desc = _generate_random_map(size=size, random_seed=random_seed)

    if desc is None:
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    else:
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Copy the transition probabilities and rewards from the frozenlake
    # gym environment.
    trans_probs = np.zeros((n_states, n_states, n_actions))
    rewards = np.zeros((n_states, n_states, n_actions))
    for state, action_P in env.P.items():
        for action, state_P_list in action_P.items():
            for probability, state_prime, reward, _ in state_P_list:
                rewards[state, state_prime, action] += reward

                # Add this probability to the transition probability matrix
                # instead of assigning it since the same state_prime can occur
                # multiple times with different probabilities
                trans_probs[state, state_prime, action] += probability

    assert np.all(
        np.isclose(np.sum(trans_probs, axis=1), 1)
    ), "Transition probabilities should sum up to 1"

    observations = np.empty((n_states, 2))
    observations[:, 0] = np.mod(np.arange(n_states), env.nrow)
    observations[:, 1] = np.floor_divide(np.arange(n_states), env.nrow)
    feature_names = ["X", "Y"]

    action_names = ["Left", "Down", "Right", "Up"]

    # The initial state is always at the top left corner unless desc is given
    initial_state_p = np.zeros(n_states)
    if desc is None:
        initial_state_p[0] = 1
    else:
        start_indices = []
        for y, row in enumerate(desc):
            for x, character in enumerate(row):
                if character == "S":
                    start_indices.append(len(row) * y + x)
        initial_state_p[start_indices] = 1 / len(start_indices)

    # Set the max_steps for 4x4 specifically to 100 like in gymnasium
    # https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    if desc is None and map_name == "4x4":
        max_steps_in_episode = 100
    else:
        max_steps_in_episode = max_steps

    mdp = MarkovDecisionProcessParams(
        trans_probs=jnp.array(trans_probs),
        rewards=jnp.array(rewards),
        initial_state_p=jnp.array(initial_state_p),
        observations=jnp.array(observations),
        max_steps_in_episode=max_steps_in_episode,
    )

    check_mdp(mdp)

    return mdp, feature_names, action_names


class Frozenlake4x4(MdpEnv):
    def __init__(self):
        super().__init__()

        (
            self.default_params_,
            self.feature_names,
            self.action_names,
        ) = generate_env_params(map_name="4x4")

        self.obs_shape = (len(self.feature_names),)

        self.name_ = "Frozenlake4x4"


class Frozenlake8x8(MdpEnv):
    def __init__(self):
        super().__init__()

        (
            self.default_params_,
            self.feature_names,
            self.action_names,
        ) = generate_env_params(map_name="8x8")

        self.obs_shape = (len(self.feature_names),)

        self.name_ = "Frozenlake8x8"


class Frozenlake12x12(MdpEnv):
    def __init__(self):
        super().__init__()

        desc = [
            "SFFFFFFFFFFF",
            "FFFFFFFFFFFF",
            "FFFHFFFFFFFH",
            "FFFFFHFFFFFF",
            "FFFHFFFFFFFF",
            "FHHFFFHFFHFF",
            "FHFFHFHFFFFF",
            "FFFHFFFFFFFF",
            "FFFFFFFFHFFF",
            "HFFFFHFFFFHH",
            "FFFFFFGFFFFF",
            "FFFFFFFFFFFF",
        ]

        (
            self.default_params_,
            self.feature_names,
            self.action_names,
        ) = generate_env_params(desc=desc)

        self.obs_shape = (len(self.feature_names),)

        self.name_ = "Frozenlake12x12"
