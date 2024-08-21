import argparse

from distutils.util import strtobool

import json

from pathlib import Path

import time

import gymnax

import jax

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from tqdm import tqdm

from dtpo.dtpo import DTPOLearner
from dtpo.utils import make_env_from_name
from dtpo.visualization import export_tree

parser = argparse.ArgumentParser()

parser.add_argument(
    "--env-name", type=str, default="CartPole-v1", help="the name of the environment"
)
parser.add_argument(
    "--max-depth", type=int, default=None, help="maximum depth of the decision tree"
)
parser.add_argument(
    "--max-leaf-nodes",
    type=int,
    default=16,
    help="maximum number of leaf nodes for the decision tree",
)
parser.add_argument(
    "--simulation-steps",
    type=int,
    default=10000,
    help="number of steps to simulate in the environment between every policy update",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="number of parallel environments to sample from",
)
parser.add_argument(
    "--max-iterations",
    type=int,
    default=1500,
    help="maximum number of times the policy is updated (outer loop)",
)
parser.add_argument(
    "--max-policy-updates",
    type=int,
    default=1,
    help="number of gradient updates for every policy update (inner loop)",
)
parser.add_argument(
    "--ppo-epsilon",
    type=float,
    default=0.2,
    help="PPO clipping value epsilon, the paper recommends 0.2",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1.0,
    help="learning rate for gradient updates",
)
parser.add_argument(
    "--gamma", type=float, default=0.99, help="discount value for future rewards"
)
parser.add_argument(
    "--normalize-advantage",
    type=lambda x: bool(strtobool(x)),
    default=True,
    help="whether to normalize GAE advantages or not",
)
parser.add_argument(
    "--early-stop-entropy",
    type=float,
    default=0.01,
    help="entropy value to stop at before discretizing",
)
parser.add_argument(
    "--seed", type=int, default=1, help="random seed for the experiment"
)
parser.add_argument(
    "--evaluation-rollouts",
    type=int,
    default=1000,
    help="number of rollouts to do for the final evaluation",
)
parser.add_argument(
    "--warmup-iterations",
    type=int,
    default=0,
    help="number of iterations used for only updating the value function",
)
parser.add_argument(
    "--anneal-lr",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="whether to linearly decay the learning rate (annealing)",
)
parser.add_argument(
    "--use-linear-value-function",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="whether to use a linear model as the value function instead of a neural network",
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

args = parser.parse_args()

env, env_params = make_env_from_name(args.env_name, args.seed)

experiment_name = f"{args.env_name}_dtpo_{int(time.time() * 10000)}_{args.seed}"
experiment_dir = f"{args.output_dir}/{experiment_name}"

# Create the experiment output directory if it does not exist
Path(experiment_dir).mkdir(parents=True, exist_ok=True)

# Create a json file with the configured hyperparameter values
filename = f"{experiment_dir}/config.json"
with open(filename, "w") as file:
    json.dump(vars(args), file, indent=4)

rng = jax.random.PRNGKey(args.seed)

rng, rng_vis = jax.random.split(rng)

print("=" * 50)
print(vars(args))
print("=" * 50)

# Create the learner and optimize the tree policy
start_time = time.time()
model = DTPOLearner(
    env,
    rng,
    ppo_epsilon=args.ppo_epsilon,
    gamma=args.gamma,
    normalize_advantage=args.normalize_advantage,
    max_depth=args.max_depth,
    max_leaf_nodes=args.max_leaf_nodes,
    max_iterations=args.max_iterations,
    learning_rate=args.learning_rate,
    anneal_lr=args.anneal_lr,
    max_policy_updates=args.max_policy_updates,
    simulation_steps=args.simulation_steps,
    num_envs=args.num_envs,
    early_stop_entropy=args.early_stop_entropy,
    warmup_iterations=args.warmup_iterations,
    use_linear_value_function=args.use_linear_value_function,
    verbose=args.verbose,
    random_state=args.seed,
)
model.learn()
runtime = time.time() - start_time

sns.set_theme(context="talk", style="whitegrid", palette="colorblind")

# Plot the policy's return per policy update
iterations = np.arange(len(model.iteration_policy_entropy_)) * args.simulation_steps
# sns.lineplot(x=iterations, y=model.mean_discounted_returns_)
plt.plot(iterations, model.mean_discounted_returns_)
plt.xlabel("iteration")
plt.ylabel("mean discounted return")
plt.tight_layout()

returns_indicated = np.array(model.mean_discounted_returns_)[
    model.iteration_updated_tree_
]
iterations_indicated = iterations[model.iteration_updated_tree_]
# sns.scatterplot(x=iterations_indicated, y=returns_indicated, s=1)
plt.scatter(
    iterations_indicated,
    returns_indicated,
    s=30,
    marker="X",
    color=sns.color_palette()[1],
)
print(model.iterations_discretized_, model.mean_discretized_discounted_returns_)
plt.plot(
    iterations[model.iterations_discretized_],
    model.mean_discretized_discounted_returns_,
    linestyle="--",
    color=sns.color_palette()[2],
)

filename = f"{experiment_dir}/return_per_episode"
plt.savefig(filename + ".png")
plt.savefig(filename + ".pdf")
plt.close()

# Plot the policy's return per policy update
iterations = np.arange(len(model.ppo_losses_)) * args.simulation_steps
sns.lineplot(x=iterations, y=model.ppo_losses_)
plt.plot(iterations, model.ppo_losses_)
plt.xlabel("iteration")
plt.ylabel("PPO loss")
plt.tight_layout()

returns_indicated = np.array(model.ppo_losses_)[model.iteration_updated_tree_]
iterations_indicated = iterations[model.iteration_updated_tree_]
plt.scatter(
    iterations_indicated,
    returns_indicated,
    s=30,
    marker="X",
    color=sns.color_palette()[1],
)

filename = f"{experiment_dir}/ppo_losses"
plt.savefig(filename + ".png")
plt.savefig(filename + ".pdf")
plt.close()

# Plot the policy's entropy per policy update
sns.lineplot(x=iterations, y=model.iteration_policy_entropy_)
plt.xlabel("iteration")
plt.ylabel("mean policy entropy")
plt.tight_layout()
filename = f"{experiment_dir}/entropy_per_episode"
plt.savefig(filename + ".png")
plt.savefig(filename + ".pdf")
plt.close()

# Plot the decision tree
env_to_feature_action_names = {
    "Pendulum-v1": (["cos theta", "sin theta", "theta dot"], ["left", "right"]),
    "MountainCar-v0": (["position", "velocity"], ["left", "don't accelerate", "right"]),
    "MountainCarContinuous-v0": (["position", "velocity"], ["force"]),
    "CartPole-v1": (
        ["cart position", "cart velocity", "pole angle", "pole angular velocity"],
        ["left", "right"],
    ),
    "Acrobot-v1": (
        [
            "cos joint 1",
            "sin joint 1",
            "cos joint 2",
            "sin joint 2",
            "velocity 1",
            "velocity 2",
        ],
        ["left torque", "no torque", "right torque"],
    ),
}

if hasattr(env, "feature_names") and hasattr(env, "action_names"):
    feature_names = env.feature_names
    action_names = env.action_names
elif args.env_name in env_to_feature_action_names:
    feature_names, action_names = env_to_feature_action_names[args.env_name]
else:
    if isinstance(env, gymnax.environments.environment.Environment):
        n_features = env.observation_space(env_params).shape[0]
    else:
        n_features = env.observation_space.shape[0]

    n_actions = env.num_actions

    feature_names = [f"feature_{i}" for i in range(n_features)]
    action_names = [f"action_{i}" for i in range(n_actions)]

filename = f"{experiment_dir}/discretized_tree"
export_tree(
    model.discretized_tree_,
    filename + ".dot",
    feature_names,
    action_names,
)
export_tree(
    model.discretized_tree_,
    filename + ".pdf",
    feature_names,
    action_names,
)
export_tree(
    model.discretized_tree_,
    filename + ".png",
    feature_names,
    action_names,
)


def default(obj):
    if isinstance(obj, jax.Array):
        if len(obj.shape) == 0:
            return obj.item()

        if len(obj.shape) == 1:
            return list(obj)

        if len(obj.shape) == 2:
            return [list(sublist) for sublist in obj]

    raise TypeError(f"Could not convert {obj} to JSON compatible type")


filename = f"{experiment_dir}/tree_params.json"
with open(filename, "w") as file:
    json.dump(model.best_params_, file, default=default)

print("=" * 50)
print(vars(args))
print("=" * 50)

returns = []
discounted_returns = []
if isinstance(env, gymnax.environments.environment.Environment):
    for repetition in tqdm(range(args.evaluation_rollouts)):
        rng, rng_reset = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)
        total_return = 0
        total_discounted_return = 0
        for i in range(env_params.max_steps_in_episode):
            rng, rng_step = jax.random.split(rng, 2)
            action = np.argmax(
                model.discretized_tree_.predict(np.array(obs).reshape(1, -1))[0]
            )
            next_obs, next_env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )
            total_return += reward
            total_discounted_return += reward * args.gamma**i
            if done:
                break
            else:
                obs = next_obs
                env_state = next_env_state

        returns.append(total_return.item())
        discounted_returns.append(total_discounted_return.item())
else:
    for repetition in tqdm(range(args.evaluation_rollouts)):
        obs, info = env.reset()
        total_return = 0
        total_discounted_return = 0
        i = 0
        while True:
            action = np.argmax(
                model.discretized_tree_.predict(np.array(obs).reshape(1, -1))[0]
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            total_discounted_return += reward * args.gamma**i
            if terminated or truncated:
                break

            obs = next_obs

            i += 1

        returns.append(total_return.item())
        discounted_returns.append(total_discounted_return.item())

# Create a json file with the result values
filename = f"{experiment_dir}/results.json"
n_nodes = int(np.sum(model.discretized_tree_.feature != -2))
with open(filename, "w") as file:
    json.dump(
        {
            "method": "dt",
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "sem_return": float(np.std(returns) / np.sqrt(len(returns))),
            "mean_discounted_return": float(np.mean(discounted_returns)),
            "sem_discounted_return": float(
                np.std(discounted_returns) / np.sqrt(len(discounted_returns))
            ),
            "runtime": runtime,
            "n_nodes": n_nodes,
            "iterations": len(model.mean_discounted_returns_),
            "mean_discounted_returns": model.mean_discounted_returns_,
            "sem_discounted_returns": model.sem_discounted_returns_,
            "mean_policy_entropies": model.iteration_policy_entropy_,
        },
        file,
        indent=4,
    )
