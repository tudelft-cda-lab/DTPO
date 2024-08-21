# DQN with JAX based on the implementation by CleanRL with modifications
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import random
import time

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

# Modified ReplayBuffer from stable_baselines3 to remove huge dependency
from .stable_baselines3 import ReplayBuffer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DqnLearner:
    def __init__(
        self,
        seed=1,
        total_timesteps=1000000,
        learning_rate=2.5e-4,
        num_envs=1,
        buffer_size=10000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=500,
        batch_size=128,
        start_e=1,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=10000,
        train_frequency=10,
        verbose=False,
    ):
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.verbose = verbose

    def learn(self, envs: gym.vector.SyncVectorEnv):
        self.episodic_returns_ = []
        self.iterations_ = []

        # TRY NOT TO MODIFY: seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        key = jax.random.PRNGKey(self.seed)
        key, q_key = jax.random.split(key, 2)

        assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        obs, _ = envs.reset(seed=self.seed)
        q_network = QNetwork(action_dim=envs.single_action_space.n)
        q_state = TrainState.create(
            apply_fn=q_network.apply,
            params=q_network.init(q_key, obs),
            target_params=q_network.init(q_key, obs),
            tx=optax.adam(learning_rate=self.learning_rate),
        )

        q_network.apply = jax.jit(q_network.apply)
        # This step is not necessary as init called on same observation and key will always lead to same initializations
        q_state = q_state.replace(
            target_params=optax.incremental_update(
                q_state.params, q_state.target_params, 1
            )
        )

        rb = ReplayBuffer(
            self.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            handle_timeout_termination=False,
        )

        @jax.jit
        def update(q_state, observations, actions, next_observations, rewards, dones):
            q_next_target = q_network.apply(
                q_state.target_params, next_observations
            )  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            next_q_value = rewards + (1 - dones) * self.gamma * q_next_target

            def mse_loss(params):
                q_pred = q_network.apply(
                    params, observations
                )  # (batch_size, num_actions)
                q_pred = q_pred[
                    jnp.arange(q_pred.shape[0]), actions.squeeze()
                ]  # (batch_size,)
                return ((q_pred - next_q_value) ** 2).mean(), q_pred

            (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
                q_state.params
            )
            q_state = q_state.apply_gradients(grads=grads)
            return loss_value, q_pred, q_state

        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=self.seed)

        total_return = 0
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(
                self.start_e,
                self.end_e,
                self.exploration_fraction * self.total_timesteps,
                global_step,
            )
            if random.random() < epsilon:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                q_values = q_network.apply(q_state.params, obs)
                actions = q_values.argmax(axis=-1)
                actions = jax.device_get(actions)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            total_return += rewards[0]
            if terminations[0] or truncations[0]:
                if self.verbose:
                    print(f"global_step={global_step}, episodic_return={total_return}")

                self.episodic_returns_.append(total_return.item())
                self.iterations_.append(global_step)
                total_return = 0

            rb.add(obs, next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                if global_step % self.train_frequency == 0:
                    data = rb.sample(self.batch_size)
                    loss, old_val, q_state = update(
                        q_state,
                        data.observations,
                        data.actions,
                        data.next_observations,
                        data.rewards.flatten(),
                        data.dones.flatten(),
                    )

                    if self.verbose and global_step % 100 == 0:
                        print("SPS:", int(global_step / (time.time() - start_time)))

                # update target network
                if global_step % self.target_network_frequency == 0:
                    q_state = q_state.replace(
                        target_params=optax.incremental_update(
                            q_state.params, q_state.target_params, self.tau
                        )
                    )

        envs.close()

        self.q_network_ = q_network
        self.q_params_ = q_state.params

    def predict_q(self, observations):
        q_values = self.q_network_.apply(self.q_params_, observations)
        return jax.device_get(q_values)

    def predict(self, observations):
        return self.predict_q(observations).argmax(axis=-1)

    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.q_params_))

    @staticmethod
    def load_model(model_path, envs):
        obs, _ = envs.reset()
        q_network = QNetwork(action_dim=envs.single_action_space.n)
        q_key = jax.random.PRNGKey(1)
        params = q_network.init(q_key, obs)
        with open(model_path, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
        q_network.apply = jax.jit(q_network.apply)

        learner = DqnLearner()
        learner.q_network_ = q_network
        learner.q_params_ = params
        return learner
