import gymnax
from gymnax.visualize import Visualizer

import jax
import jax.numpy as jnp

from dtpo.dtpo import DecisionTreePolicy
from dtpo.utils import make_env_from_name

import math

from pathlib import Path

import json

import argparse

# NOTE: gymnax requires the old gym version 0.19.0 and old pyglet version 1.5.27 for visualizations

parser = argparse.ArgumentParser()

parser.add_argument("env_name", type=str, help="the name of the environment")

parser.add_argument(
    "tree_params_path",
    type=str,
    help="path to the JSON parameters of the tree policy to visualize",
)
parser.add_argument(
    "--seed", type=int, default=1, help="random seed for the visualization"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="out",
    help="the directory to output result files into",
)

args = parser.parse_args()

env, env_params = make_env_from_name(args.env_name, args.seed)

# hacky way to make CartPoleSwingup work with the visualization code for CartPole-v1
if args.env_name == "CartPoleSwingup":
    from dtpo.environments.cartpoleswingup import CartPoleSwingUp

    CartPoleSwingUp.name = property(lambda _: "CartPole-v1")

with open(args.tree_params_path, "r") as file:
    tree_params = json.load(file)

# turn parameters into jax arrays
for parameter_name in (
    "features",
    "thresholds",
    "children_left",
    "children_right",
    "leaf_logits",
):
    tree_params["params"][parameter_name] = jnp.array(
        tree_params["params"][parameter_name]
    )

max_nodes = len(tree_params["params"]["features"])
tree_policy = DecisionTreePolicy(env.num_actions, max_nodes)

tree_policy.apply = jax.jit(tree_policy.apply)

rng = jax.random.PRNGKey(args.seed)
if isinstance(env, gymnax.environments.environment.Environment):
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        logits = tree_policy.apply(tree_params, obs)
        action = jnp.argmax(logits)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

# hacky way to make CartPoleSwingup work with the visualization code for CartPole-v1
if args.env_name == "CartPoleSwingup":
    from gymnax.environments.classic_control.cartpole import EnvParams as CartPoleParams
    from gymnax.environments.classic_control.cartpole import EnvState

    env_params = CartPoleParams(
        gravity=env_params.g,
        masscart=env_params.m_c,
        masspole=env_params.m_p,
        total_mass=env_params.m_c + env_params.m_p,
        length=env_params.l,
        polemass_length=env_params.m_p * env_params.l,
        force_mag=env_params.force_mag,
        tau=env_params.dt,
        theta_threshold_radians=env_params.theta_threshold_radians,
        x_threshold=env_params.x_threshold,
        max_steps_in_episode=env_params.max_steps_in_episode,
    )

    # flip the thetas since these are inverted between CartPole and CartPoleSwingup
    state_seq = [
        EnvState(s.x, s.x_dot, -s.theta, s.theta_dot, s.time) for s in state_seq
    ]

cum_rewards = jnp.cumsum(jnp.array(reward_seq))
visualizer = Visualizer(
    env=env,
    env_params=env_params,
    state_seq=state_seq,
    reward_seq=cum_rewards,
)

visualization_dir = f"{args.output_dir}/visualizations"
Path(visualization_dir).mkdir(parents=True, exist_ok=True)

policy_name = args.tree_params_path.replace("/", "_").rstrip("_tree_params.json")

visualizer.animate(f"{visualization_dir}/{policy_name}_{args.seed}.gif")
