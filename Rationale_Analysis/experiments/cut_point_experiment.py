import subprocess
import os

import json

default_values = json.load(open("Rationale_Analysis/second_cut_point.json"))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--run-one", dest="run_one", action="store_true")
parser.add_argument("--cluster", dest="cluster", action="store_true")
parser.add_argument("--all-data", dest="all_data", action="store_true")

parser.add_argument("--output-dir", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--min-scale", type=float)
parser.add_argument("--max-scale", type=float)


def main(args):
    if args.all_data:
        datasets = default_values.keys()
    else:
        datasets = [os.environ["DATASET_NAME"]]

    for dataset in datasets:
        new_env = os.environ.copy()
        new_env.update({k: str(v) for k, v in default_values[dataset].items()})
        new_env["KEEP_PROB"] = str(1.0)
        new_env["DATASET_NAME"] = dataset

        ith_search_space = {}
        ith_search_space["RANDOM_SEED"] = [1000, 2000, 3000, 4000, 5000]

        cmd = (
            [
                "python",
                "Rationale_Analysis/experiments/model_a_experiments.py",
                "--exp-name",
                args.exp_name,
                "--search-space",
                json.dumps(ith_search_space),
                "--script-type",
                args.script_type,
            ]
            + (["--dry-run"] if args.dry_run else [])
            + (["--run-one"] if args.run_one else [])
            + (["--cluster"] if args.cluster else [])
        )

        print(default_values[dataset])
        print(ith_search_space)
        subprocess.run(cmd, check=True, env=new_env)


from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib

# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

datasets = {"SST": "SST", "agnews": "AGNews", "evinf": "Ev. Inf.", "movies": "Movies", "multirc": "MultiRC"}
cut_point_thresh = [[0.1, 0.2], [0.1, 0.2], [0.05, 0.1], [0.15, 0.3], [0.1, 0.2]]


def results(args):
    data = []
    names = ["Lei et al", "[CLS] Attention + Top K"]
    for c, (dataset, dataset_name) in enumerate(datasets.items()):
        output_dirs_point = [
            [
                os.path.join(
                    args.output_dir,
                    "bert_encoder_generator",
                    dataset,
                    "cut_point",
                    "EXP_NAME_HERE",
                    "top_k_rationale",
                    "direct",
                    "test_metrics.json",
                ),
                os.path.join(
                    args.output_dir,
                    "bert_classification",
                    dataset,
                    "direct",
                    "EXP_NAME_HERE",
                    "wrapper_saliency",
                    "top_k_rationale",
                    "second_cut_point",
                    "model_b",
                    "metrics.json",
                ),
            ],
            [
                os.path.join(
                    args.output_dir,
                    "bert_encoder_generator",
                    dataset,
                    "direct",
                    "EXP_NAME_HERE",
                    "top_k_rationale",
                    "direct",
                    "test_metrics.json",
                ),
                os.path.join(
                    args.output_dir,
                    "bert_classification",
                    dataset,
                    "direct",
                    "EXP_NAME_HERE",
                    "wrapper_saliency",
                    "top_k_rationale",
                    "direct",
                    "model_b",
                    "metrics.json",
                ),
            ],
        ]

        for cut, output_dirs in enumerate(output_dirs_point):
            for name, output_dir in zip(names, output_dirs):
                for seed in [1000, 2000, 3000, 4000, 5000]:
                    exp_dict = {"Dataset": dataset_name, "Model": name, "cut_point": cut}
                    exp_name = []
                    for k, v in zip(["RANDOM_SEED"], [seed]):
                        exp_name.append(k + "=" + str(v))
                        exp_dict[k] = v

                    try:
                        metrics = json.load(open(output_dir.replace("EXP_NAME_HERE", ":".join(exp_name))))
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if k.startswith("test_fscore")
                            or k.startswith("test__fscore")
                            or k.startswith("_fscore")
                            or k.startswith("fscore")
                        }
                        m = np.mean(list(metrics.values()))
                        exp_dict["Macro F1"] = max(0, m)
                    except FileNotFoundError:
                        print(name, output_dir, exp_name)
                        continue

                    data.append(exp_dict)

    sns.set(style="ticks", rc={"lines.linewidth": 0.7})
    data = pd.DataFrame(data)
    fig = plt.figure(figsize=(4, 3))
    ax = sns.catplot(
        x="cut_point",
        y="Macro F1",
        hue="Model",
        ci="sd",
        aspect=0.5,
        data=data,
        estimator=np.median,
        markers=["o", "D"],
        kind="point",
        col="Dataset",
        legend=False,
        palette=["blue", "red"],
        dodge=True,
        join=True,
    )

    for c, _ in enumerate(datasets.items()) :
        thresh = cut_point_thresh[c]
        ax.axes[0, c].set_xticklabels(thresh)
        ax.axes[0, c].set_xlabel("")
        
    plt.ylim(args.min_scale, args.max_scale)
    plt.tight_layout()
    plt.legend().remove()
    sns.despine()
    plt.xlabel("Cut Point")
    plt.savefig("cut-point.pdf", bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.script_type == "results":
        results(args)
    else:
        main(args)
