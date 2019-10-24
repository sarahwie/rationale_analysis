import argparse
import os
import json
from itertools import product
import subprocess

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--exp-folder", type=str, required=True)
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--search-space", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--cluster", dest="cluster", action="store_true")

parser.add_argument("--graph-name", type=str, required=True)
parser.add_argument("--value", type=str, required=True)
parser.add_argument("--metric", type=str, required=True)
parser.add_argument("--deviation", type=str, required=True)

def main(args):
    global_exp_name = args.exp_name
    search_space = json.loads(args.search_space)
    keys, values = zip(*search_space.items())

    global_exp_folder = args.exp_folder

    x_axis_field = args.value
    y_axis_field = args.metric
    deviation_field = args.deviation

    metrics = []

    for prod in product(*values):
        exp_dict = dict(zip(keys, prod))
        exp_name = []
        for k, v in zip(keys, prod):
            exp_name.append(k + "=" + str(v))

        exp_name = os.path.join(global_exp_name, ":".join(exp_name))
        
        metrics_file = json.load(open(os.path.join(global_exp_folder, exp_name, 'metrics.json')))
        metric = metrics_file[args.metric]

        metrics.append({
            x_axis_field : exp_dict[x_axis_field],
            y_axis_field : metric,
            deviation_field : exp_dict[deviation_field]
        })

    metrics = pd.DataFrame(metrics)
    sns.boxplot(x=x_axis_field, y=y_axis_field, data=metrics)
    plt.tight_layout()
    plt.savefig(os.path.join(global_exp_folder, global_exp_name, args.graph_name), bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
