import pandas as pd

import os

import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output-dir", type=str, default="out", help="the directory with result files"
)

args = parser.parse_args()

all_results = []
for result_dir in os.listdir(args.output_dir):
    path_prefix = f"{args.output_dir}/{result_dir}"

    if not os.path.isdir(path_prefix):
        continue

    result_path = f"{path_prefix}/results.json"
    config_path = f"{path_prefix}/config.json"

    try:
        with open(result_path) as file:
            results = json.load(file)
    except:
        print(f"{path_prefix} missing the result file, skipping")
        continue

    try:
        with open(config_path) as file:
            config = json.load(file)
    except:
        print(f"{path_prefix} missing the config file, skipping")
        continue

    if "method" not in results:
        continue

    # Combine the two dictionaries
    results.update(config)
    all_results.append(results)

result_df = pd.DataFrame(all_results)

first_columns = [
    "env_name",
    "method",
    "max_depth",
    "mean_return",
    "seed",
    "sem_return",
    "runtime",
    "n_nodes",
]
drop_columns = [
    "verbose",
    "output_dir",
    "mean_discounted_returns",
    "sem_discounted_returns",
    "mean_policy_entropies",
    "iterations",
]

# Reorder the columns with the most important columns first
columns = result_df.columns.tolist()
for column in reversed(first_columns):
    col_i = columns.index(column)
    columns = [column] + columns[:col_i] + columns[col_i + 1 :]

# Remove columns that are not informative
for column in drop_columns:
    if column in columns:
        columns.remove(column)

result_df = result_df[columns]
result_df.sort_values(by=first_columns, inplace=True)
result_df.to_csv(f"{args.output_dir}/combined_results.csv", index=False)

result_df = result_df[
    (result_df["max_leaf_nodes"] == 16)
    | (result_df["method"] == "ppo")
    | (result_df["method"] == "dqn")
    | (result_df["method"] == "viper")
]

method_col_order = ["viper", "dt", "dqn", "ppo"]

table_df = pd.pivot_table(
    result_df, index="env_name", columns="method", values="mean_return", aggfunc="mean"
)
sem_table_df = pd.pivot_table(
    result_df, index="env_name", columns="method", values="mean_return", aggfunc="sem"
)
table_df = table_df[method_col_order]
sem_table_df = sem_table_df[method_col_order]

for i in range(table_df.shape[0]):
    for j in range(table_df.shape[1]):
        table_df.iloc[
            i, j
        ] = f"{table_df.iloc[i, j]:.2f} \\tiny $\\pm$ {sem_table_df.iloc[i, j]:.2f}"

table_df = table_df.reset_index()
print(table_df)

table_df.to_latex(f"{args.output_dir}/performance.tex", index=False, escape=False)

result_df["runtime_hours"] = result_df["runtime"] / 3600
print(
    pd.pivot_table(
        result_df,
        index="env_name",
        columns="method",
        values="runtime_hours",
        aggfunc="median",
    )[method_col_order]
    .round(1)
    .to_latex()
)
