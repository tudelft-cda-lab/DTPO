import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math

import json

import os

sns.set_theme(style="whitegrid", palette="colorblind")

output_dir = "out_size"

results = []
for filename in os.listdir(output_dir):
    if not os.path.isdir(f"{output_dir}/{filename}"):
        continue

    try:
        with open(f"{output_dir}/{filename}/config.json") as file:
            config_dict = json.load(file)

        with open(f"{output_dir}/{filename}/results.json") as file:
            result_dict = json.load(file)
    except:
        print(f"Skipping directory {output_dir}/{filename}")
        continue

    results.append(config_dict | result_dict)

results_df = pd.DataFrame(results)
results_df.drop(
    columns=[
        "mean_discounted_returns",
        "sem_discounted_returns",
        "mean_policy_entropies",
    ],
    inplace=True,
)
results_df.sort_values(
    by=["env_name", "max_policy_updates", "learning_rate", "entropy_coef"], inplace=True
)
results_df.to_csv(f"{output_dir}/results_combined.csv", index=False)

env_names = [
    "Frozenlake4x4",
    "CartPole-v1",
    "PendulumBangBang",
    "CartPoleSwingup",
]

results_df = results_df[results_df["method"] == "dt"]
results_df["max_leaf_nodes"] = results_df["max_leaf_nodes"].map(
    lambda x: f"$2^{int(math.log2(x))}$"
)
order = sorted(results_df["max_leaf_nodes"].unique())

for env_name in env_names:
    env_df = results_df[results_df["env_name"] == env_name].copy()
    if len(env_df) == 0:
        continue

    filename = f"{output_dir}/varying_size_{env_name}"
    _, ax = plt.subplots(figsize=(4, 3))

    maxes = env_df.groupby("max_leaf_nodes")["mean_return"].transform("max")
    env_df["is_best"] = env_df["mean_return"] == maxes
    env_df["max_return"] = maxes

    sns.stripplot(
        data=env_df,
        x="max_leaf_nodes",
        y="mean_return",
        order=order,
        jitter=False,
        color="gray",
        alpha=0.5,
        ax=ax,
    )
    sns.pointplot(
        data=env_df, x="max_leaf_nodes", y="max_return", order=order, ax=ax, zorder=100
    )

    plt.xlabel("number of leaves")
    plt.ylabel("return")
    plt.tight_layout()
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")
    plt.close()
