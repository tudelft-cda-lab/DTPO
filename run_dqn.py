import argparse

from distutils.util import strtobool

import json

from pathlib import Path

import time

import gymnasium

import jax

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from dtpo.dqn import DqnLearner
from dtpo.utils import make_env_from_name

parser = argparse.ArgumentParser()

# General arguments
parser.add_argument(
    "--env-name", type=str, default="CartPole-v1", help="the name of the environment"
)
parser.add_argument(
    "--seed", type=int, default=1, help="random seed for the experiment"
)
parser.add_argument(
    "--verbose",
    type=lambda x: bool(strtobool(x)),
    default=True,
    help="whether to print debug information or not",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="out",
    help="the directory to output result files into",
)
parser.add_argument(
    "--evaluation-rollouts",
    type=int,
    default=1000,
    help="number of rollouts to do for the final evaluation",
)

# Arguments specific to deep q learning (the teacher model)
parser.add_argument(
    "--total-timesteps",
    type=int,
    default=10000000,
    help="total environment timesteps to use in training",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2.5e-4,
    help="learning rate for the Adam optimizer",
)
parser.add_argument(
    "--buffer-size", type=int, default=10000, help="size of the experience buffer"
)
parser.add_argument(
    "--gamma", type=float, default=0.99, help="discount value for future rewards"
)
parser.add_argument(
    "--tau", type=float, default=1.0, help="the target network update rate"
)
parser.add_argument(
    "--target-network-frequency",
    type=int,
    default=500,
    help="the timesteps it takes to update the target network",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="the batch size of sample from the reply memory",
)
parser.add_argument(
    "--start-e", type=float, default=1, help="the starting epsilon for exploration"
)
parser.add_argument(
    "--end-e", type=float, default=0.05, help="the ending epsilon for exploration"
)
parser.add_argument(
    "--exploration-fraction",
    type=float,
    default=0.5,
    help="the fraction of `total-timesteps` it takes from start-e to go end-e",
)
parser.add_argument(
    "--learning-starts", type=int, default=10000, help="timestep to start learning"
)
parser.add_argument(
    "--train-frequency", type=int, default=10, help="the frequency of training"
)

args = parser.parse_args()

env, env_params, vec_env = make_env_from_name(
    args.env_name, args.seed, return_gym_vec_env=True, num_envs_vec=1
)

timestamp = int(time.time() * 1000)
experiment_name = f"{args.env_name}_dqn_{timestamp}_{args.seed}"
experiment_dir = f"{args.output_dir}/{experiment_name}"

# Create the experiment output directory if it does not exist
Path(experiment_dir).mkdir(parents=True, exist_ok=True)

# Create a json file with the configured hyperparameter values
filename = f"{experiment_dir}/config.json"
with open(filename, "w") as file:
    json.dump(vars(args), file, indent=4)

rng = jax.random.PRNGKey(args.seed)

assert isinstance(
    vec_env.single_action_space, gymnasium.spaces.Discrete
), "only discrete action spaces are supported"

print("=" * 50)
print(vars(args))
print("=" * 50)

# Create the learner and optimize the policy
start_time = time.time()

learner = DqnLearner(
    seed=args.seed,
    total_timesteps=args.total_timesteps,
    learning_rate=args.learning_rate,
    num_envs=1,
    buffer_size=args.buffer_size,
    gamma=args.gamma,
    tau=args.tau,
    target_network_frequency=args.target_network_frequency,
    batch_size=args.batch_size,
    start_e=args.start_e,
    end_e=args.end_e,
    exploration_fraction=args.exploration_fraction,
    learning_starts=args.learning_starts,
    train_frequency=args.train_frequency,
    verbose=args.verbose,
)
learner.learn(vec_env)

# Always save the model in the experiment directory (also when using a pretrained model)
model_path = f"{experiment_dir}/jax_dqn_model.flax"
learner.save_model(model_path)

runtime = time.time() - start_time

sns.set_theme(context="talk", style="whitegrid", palette="colorblind")


# From: https://stackoverflow.com/a/14314054/15406859
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# Plot the policy's return per policy update
episodes = np.arange(len(learner.episodic_returns_))
n_average = 100
sns.lineplot(
    x=episodes[n_average - 1 :], y=moving_average(learner.episodic_returns_, n_average)
)
plt.tight_layout()
filename = f"{experiment_dir}/return_per_episode"
plt.savefig(filename + ".png")
plt.savefig(filename + ".pdf")
plt.close()

print("=" * 50)
print(vars(args))
print("=" * 50)

returns = []
for repetition in range(args.evaluation_rollouts):
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    total_return = 0
    for _ in range(env_params.max_steps_in_episode):
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Since the learner.tree_policy_ is a classifier we don't have to use
        # the argmax, just predict()
        action = learner.predict(np.array(obs).reshape(1, -1))[0]
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        total_return += reward
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    returns.append(total_return.item())

# Create a json file with the result values
filename = f"{experiment_dir}/results.json"
with open(filename, "w") as file:
    json.dump(
        {
            "method": "dqn",
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "sem_return": float(np.std(returns) / np.sqrt(len(returns))),
            "runtime": runtime,
            "iterations": learner.iterations_,
            "mean_discounted_returns": learner.episodic_returns_,
        },
        file,
        indent=4,
    )
