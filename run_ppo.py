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

from tqdm import tqdm

from dtpo.ppo_nn import PpoLearner
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

# Arguments specific to PPO
parser.add_argument(
    "--total-timesteps",
    type=int,
    default=10000000,
    help="total timesteps of the experiments",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2.5e-4,
    help="the learning rate of the optimizer",
)
parser.add_argument(
    "--num-envs", type=int, default=4, help="the number of parallel game environments"
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=128,
    help="the number of steps to run in each environment per policy rollout",
)
parser.add_argument(
    "--anneal-lr",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="Toggle learning rate annealing for policy and value networks",
)
parser.add_argument(
    "--gamma", type=float, default=0.99, help="the discount factor gamma"
)
parser.add_argument(
    "--gae-lambda",
    type=float,
    default=0.95,
    help="the lambda for the general advantage estimation",
)
parser.add_argument(
    "--num-minibatches", type=int, default=4, help="the number of mini-batches"
)
parser.add_argument(
    "--update-epochs", type=int, default=4, help="the K epochs to update the policy"
)
parser.add_argument(
    "--norm-adv",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="Toggles advantages normalization",
)
parser.add_argument(
    "--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient"
)
parser.add_argument(
    "--clip-vloss",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
)
parser.add_argument(
    "--ent-coef", type=float, default=0.01, help="coefficient of the entropy"
)
parser.add_argument(
    "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
)
parser.add_argument(
    "--max-grad-norm",
    type=float,
    default=0.5,
    help="the maximum norm for the gradient clipping",
)
parser.add_argument(
    "--target-kl", type=float, default=None, help="the target KL divergence threshold"
)

args = parser.parse_args()

env, env_params, vec_env = make_env_from_name(
    args.env_name, args.seed, return_gym_vec_env=True, num_envs_vec=1
)

timestamp = int(time.time() * 1000)
experiment_name = f"{args.env_name}_ppo_{timestamp}_{args.seed}"
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

learner = PpoLearner(
    seed=args.seed,
    torch_deterministic=True,
    cuda="cpu",
    total_timesteps=args.total_timesteps,
    learning_rate=args.learning_rate,
    num_envs=args.num_envs,
    num_steps=args.num_steps,
    anneal_lr=args.anneal_lr,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    num_minibatches=args.num_minibatches,
    update_epochs=args.update_epochs,
    norm_adv=args.norm_adv,
    clip_coef=args.clip_coef,
    clip_vloss=args.clip_vloss,
    ent_coef=args.ent_coef,
    vf_coef=args.vf_coef,
    max_grad_norm=args.max_grad_norm,
    target_kl=args.target_kl,
    verbose=args.verbose,
)
learner.learn(vec_env)

# Always save the model in the experiment directory (also when using a pretrained model)
model_path = f"{experiment_dir}/jax_ppo_model.torch"
learner.save_model(model_path)

runtime = time.time() - start_time

sns.set_theme(context="talk", style="whitegrid", palette="colorblind")

# Plot the policy's return per policy update
episodes = np.arange(len(learner.episodic_returns_))
sns.lineplot(x=episodes, y=learner.episodic_returns_)
plt.tight_layout()
filename = f"{experiment_dir}/return_per_episode"
plt.savefig(filename + ".png")
plt.savefig(filename + ".pdf")
plt.close()

print("=" * 50)
print(vars(args))
print("=" * 50)

returns = []
for repetition in tqdm(range(args.evaluation_rollouts)):
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    total_return = 0
    for _ in range(env_params.max_steps_in_episode):
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Since the learner.tree_policy_ is a classifier we don't have to use
        # the argmax, just predict()
        action = learner.predict(np.array(obs).reshape(1, -1))[0].item()
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
            "method": "ppo",
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
