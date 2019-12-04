import subprocess
import os

search_space = {"KEEP_PROB": [0.2, 0.4, 0.6, 0.8, 1.0], "RANDOM_SEED": [1000, 2000, 3000, 4000, 5000]}

import json

default_values = json.load(open("Rationale_Analysis/default_values.json"))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--run-one", dest="run_one", action="store_true")
parser.add_argument("--cluster", dest="cluster", action="store_true")
parser.add_argument("--total-data", type=float, required=True)


parser.add_argument("--output-dirs", type=str, nargs='+')
parser.add_argument("--names", type=str, nargs='+')
parser.add_argument("--metric", type=str, nargs='+')

exp_default = {'MU' : 0.0}

def main(args):
    new_env = os.environ.copy()
    dataset = new_env["DATASET_NAME"]
    new_env.update({k:str(v) for k, v in default_values[dataset].items()})
    new_env.update({k:str(v) for k, v in exp_default.items()})

    search_space['KEEP_PROB'] = [x/args.total_data for x in search_space['KEEP_PROB']]

    cmd = (
        [
            "python",
            "Rationale_Analysis/experiments/model_a_experiments.py",
            "--exp-name",
            "learning_curve",
            "--search-space",
            json.dumps(search_space),
            "--script-type", 
            args.script_type
        ]
        + (["--dry-run"] if args.dry_run else [])
        + (["--run-one"] if args.run_one else [])
        + (["--cluster"] if args.cluster else [])
    )

    print(new_env)
    subprocess.run(cmd, check=True, env=new_env)

from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

def results(args) :
    search_space['KEEP_PROB'] = [x/args.total_data for x in search_space['KEEP_PROB']]
    keys, values = zip(*search_space.items())

    data = []
    for name, output_dir, metric in zip(args.names, args.output_dirs, args.metric) :
        for prod in product(*values):
            exp_dict = {'Model' : name}
            exp_name = []
            for k, v in zip(keys, prod):
                exp_name.append(k + "=" + str(v))
                exp_dict[k] = v

            try :
                metrics = json.load(open(output_dir.replace('EXP_NAME_HERE', ":".join(exp_name))))
                if 'test_' + metric in metrics :
                    m = metrics['test_' + metric]
                else :
                    m = metrics[metric]
                exp_dict[args.metric[-1]] = max(0, m)
            except FileNotFoundError:
                print(exp_name)
                continue

            data.append(exp_dict)

    sns.set(style="ticks", rc={"lines.linewidth": 0.7})
    data = pd.DataFrame(data)
    fig = plt.figure(figsize=(4, 3))
    sns.pointplot(x='KEEP_PROB', y=args.metric[-1], hue='Model', ci='sd', data=data, estimator=np.median, markers=['x']*len(args.names))

    plt.tight_layout()
    sns.despine()
    plt.xlabel("Training Set Size")
    plt.savefig('AgNews-comparison.pdf', bbox_inches='tight')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.script_type == 'results' :
        results(args)
    else :
        main(args)
