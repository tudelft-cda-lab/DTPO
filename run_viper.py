import argparse

from distutils.util import strtobool

import json

from pathlib import Path

import time

import gymnasium

import gymnax

import jax

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from tqdm import tqdm

from dtpo.dqn import DqnLearner
from dtpo.utils import make_env_from_name
from dtpo.viper import ViperLearner
from dtpo.visualization import export_tree

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

# Arguments specific to deep q learning (the teacher model)
parser.add_argument(
    "--pretrained-model-path",
    type=str,
    default=None,
    help="path to a pretrained JAX DQN model, if None then train one from scratch",
)
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

# Arguments specific to VIPER (the student model)
parser.add_argument(
    "--max-depth", type=int, default=None, help="maximum depth of the decision tree"
)
parser.add_argument(
    "--max-leaf-nodes",
    type=int,
    default=16,
    help="maximum number of leaves of the decision tree",
)
parser.add_argument(
    "--n-batch-rollouts",
    type=int,
    default=10,
    help="number of rollouts per policy training iteration",
)
parser.add_argument(
    "--n-test-rollouts",
    type=int,
    default=50,
    help="number of rollouts per policy selection iteration",
)
parser.add_argument(
    "--max-samples",
    type=int,
    default=200000,
    help="maximum number of samples to collect when training new trees",
)
parser.add_argument(
    "--max-iters",
    type=int,
    default=80,
    help="maximum number of iterations when training new trees",
)
parser.add_argument(
    "--train-frac",
    type=float,
    default=0.8,
    help="fraction of samples to use for training",
)
parser.add_argument(
    "--evaluation-rollouts",
    type=int,
    default=1000,
    help="number of rollouts to do for the final evaluation",
)
parser.add_argument(
    "--is-reweight",
    type=lambda x: bool(strtobool(x)),
    default=True,
    help="whether to reweight samples during training based on q values",
)

args = parser.parse_args()

env, env_params, gym_env, vec_env = make_env_from_name(
    args.env_name,
    seed=args.seed,
    return_gym_env=True,
    return_gym_vec_env=True,
    num_envs_vec=1,
)

timestamp = int(time.time() * 1000)

dqn_name = f"{args.env_name}_dqn_{timestamp}_{args.seed}"
dqn_dir = f"{args.output_dir}/{dqn_name}"

experiment_name = f"{args.env_name}_viper_{timestamp}_{args.seed}"
experiment_dir = f"{args.output_dir}/{experiment_name}"

# Create the experiment output directory if it does not exist
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(dqn_dir).mkdir(parents=True, exist_ok=True)

# Create a json file with the configured hyperparameter values
if args.pretrained_model_path is None:
    filename = f"{dqn_dir}/config.json"
    with open(filename, "w") as file:
        dqn_args = {
            "env_name": args.env_name,
            "seed": args.seed,
            "verbose": args.verbose,
            "output_dir": args.output_dir,
            "pretrained_model_path": args.pretrained_model_path,
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "target_network_frequency": args.target_network_frequency,
            "batch_size": args.batch_size,
            "start_e": args.start_e,
            "end_e": args.end_e,
            "exploration_fraction": args.exploration_fraction,
            "learning_starts": args.learning_starts,
            "train_frequency": args.train_frequency,
        }
        json.dump(dqn_args, file, indent=4)

filename = f"{experiment_dir}/config.json"
with open(filename, "w") as file:
    json.dump(vars(args), file, indent=4)

rng = jax.random.PRNGKey(args.seed)
random_state = np.random.RandomState(args.seed)

rng, rng_vis = jax.random.split(rng)

assert isinstance(
    vec_env.single_action_space, gymnasium.spaces.Discrete
), "only discrete action spaces are supported"

print("=" * 50)
print(vars(args))
print("=" * 50)

# Create the learner and optimize the DQN and tree policy
start_time = time.time()

if args.pretrained_model_path is None:
    if args.verbose:
        print("Training DQN model from scratch")

    teacher = DqnLearner(
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
    teacher.learn(vec_env)
    teacher.save_model(f"{dqn_dir}/jax_dqn_model.flax")
else:
    if args.verbose:
        print("Using a pretrained DQN model, training arguments are ignored")

    teacher = DqnLearner.load_model(args.pretrained_model_path, vec_env)
    teacher.iterations_ = []
    teacher.episodic_returns_ = None

dqn_runtime = time.time() - start_time

# Always save the model in the experiment directory (also when using a pretrained model)
model_path = f"{experiment_dir}/jax_dqn_model.flax"
teacher.save_model(model_path)

learner = ViperLearner(
    max_depth=args.max_depth,
    max_leaf_nodes=args.max_leaf_nodes,
    n_batch_rollouts=args.n_batch_rollouts,
    max_samples=args.max_samples,
    max_iters=args.max_iters,
    train_frac=args.train_frac,
    is_reweight=args.is_reweight,
    n_test_rollouts=args.n_test_rollouts,
    is_train=True,
    random_state=random_state,
)
learner.learn(gym_env, teacher)
runtime = time.time() - start_time

sns.set_theme(context="talk", style="whitegrid", palette="colorblind")

# Only plot DQN training behavior if no pretrained model is given
if args.pretrained_model_path is None:
    # From: https://stackoverflow.com/a/14314054/15406859
    def moving_average(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    # Plot the policy's return per policy update
    episodes = np.arange(len(teacher.episodic_returns_))
    n_average = 100
    sns.lineplot(
        x=episodes[n_average - 1 :],
        y=moving_average(teacher.episodic_returns_, n_average),
    )
    plt.tight_layout()
    filename = f"{dqn_dir}/return_per_episode"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")
    plt.close()

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
    learner.tree_policy_.tree_,
    filename + ".dot",
    feature_names,
    action_names,
)
export_tree(
    learner.tree_policy_.tree_,
    filename + ".pdf",
    feature_names,
    action_names,
)
export_tree(
    learner.tree_policy_.tree_,
    filename + ".png",
    feature_names,
    action_names,
)

print("=" * 50)
print(vars(args))
print("=" * 50)

dqn_predict = lambda obs: teacher.predict(np.array(obs).reshape(1, -1))[0]
tree_predict = lambda obs: learner.tree_policy_.predict(obs.reshape(1, -1))[0]


def evaluate_policy(predict):
    returns = []
    discounted_returns = []
    for repetition in tqdm(range(args.evaluation_rollouts)):
        obs, info = gym_env.reset()
        total_return = 0
        total_discounted_return = 0
        i = 0
        while True:
            action = predict(obs)
            next_obs, reward, terminated, truncated, info = gym_env.step(action)
            total_return += reward
            total_discounted_return += reward * args.gamma**i
            if terminated or truncated:
                break
            else:
                obs = next_obs

            i += 1

        returns.append(total_return.item())
        discounted_returns.append(total_discounted_return.item())

    return returns, discounted_returns


# Create a json file with the result values for DQN
filename = f"{dqn_dir}/results.json"
returns, discounted_returns = evaluate_policy(dqn_predict)
with open(filename, "w") as file:
    json.dump(
        {
            "method": "dqn",
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "sem_return": float(np.std(returns) / np.sqrt(len(returns))),
            "mean_discounted_return": float(np.mean(discounted_returns)),
            "sem_discounted_return": float(
                np.std(discounted_returns) / np.sqrt(len(discounted_returns))
            ),
            "runtime": dqn_runtime,
            "iterations": len(teacher.iterations_),
            "mean_discounted_returns": teacher.episodic_returns_,
        },
        file,
        indent=4,
    )

# Create a json file with the result values for VIPER
filename = f"{experiment_dir}/results.json"
n_nodes = int(np.sum(learner.tree_policy_.tree_.feature != -2))
returns, discounted_returns = evaluate_policy(tree_predict)
with open(filename, "w") as file:
    json.dump(
        {
            "method": "viper",
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "sem_return": float(np.std(returns) / np.sqrt(len(returns))),
            "mean_discounted_return": float(np.mean(discounted_returns)),
            "sem_discounted_return": float(
                np.std(discounted_returns) / np.sqrt(len(discounted_returns))
            ),
            "runtime": runtime,
            "n_nodes": n_nodes,
            "mean_discounted_returns": teacher.episodic_returns_,
        },
        file,
        indent=4,
    )
