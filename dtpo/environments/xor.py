import numpy as np

from .mdp import MarkovDecisionProcessParams, MdpEnv, check_mdp

import jax.numpy as jnp

from tqdm import tqdm


def generate_env_params(n_states=200, seed=0):
    random_state = np.random.RandomState(seed=seed)
    observations = random_state.rand(n_states, 2)

    feature_names = ["X", "Y"]
    action_names = ["not_xor", "xor"]

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    T[:, :, :] = 1 / len(observations)

    threshold = 0.5

    for s, observation in tqdm(enumerate(observations), total=len(observations)):
        if observation[0] >= threshold:
            if observation[1] >= threshold:
                R[s, :, 1] = -1
                R[s, :, 0] = 1
            else:
                R[s, :, 1] = 1
                R[s, :, 0] = -1
        else:
            if observation[1] >= threshold:
                R[s, :, 1] = 1
                R[s, :, 0] = -1
            else:
                R[s, :, 1] = -1
                R[s, :, 0] = 1

    # The start state is chosen uniformly at random
    initial_state_p = np.ones(n_states)
    initial_state_p /= initial_state_p.sum()

    mdp = MarkovDecisionProcessParams(
        trans_probs=jnp.array(T),
        rewards=jnp.array(R),
        initial_state_p=jnp.array(initial_state_p),
        observations=jnp.array(observations),
    )

    check_mdp(mdp)

    return mdp, feature_names, action_names


class Xor(MdpEnv):
    def __init__(self):
        super().__init__()

        (
            self.default_params_,
            self.feature_names,
            self.action_names,
        ) = generate_env_params()

        self.obs_shape = (len(self.feature_names),)

        self.name_ = "Xor"
