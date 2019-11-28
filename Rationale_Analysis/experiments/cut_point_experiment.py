import subprocess
import os

import json

search_space = json.load(open('Rationale_Analysis/experiments/exp_spaces/cut_point.json'))
default_values = json.load(open("Rationale_Analysis/default_values.json"))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--run-one", dest="run_one", action="store_true")
parser.add_argument("--cluster", dest="cluster", action="store_true")

def main(args):
    new_env = os.environ.copy()
    dataset = new_env["DATASET_NAME"]
    new_env.update({k:str(v) for k, v in default_values[dataset].items()})
    new_env['KEEP_PROB'] = str(1.0)

    dataset_specific_vars = search_space[dataset]

    for i in range(2) :
        ith_search_space = {k:[v[i]] for k, v in dataset_specific_vars.items()}
        ith_search_space['RANDOM_SEED'] = [1000, 2000, 3000, 4000, 5000]

        cmd = (
            [
                "python",
                "Rationale_Analysis/experiments/model_a_experiments.py",
                "--exp-name",
                "cut_point",
                "--search-space",
                json.dumps(ith_search_space),
                "--script-type", 
                args.script_type
            ]
            + (["--dry-run"] if args.dry_run else [])
            + (["--run-one"] if args.run_one else [])
            + (["--cluster"] if args.cluster else [])
        )

        print(ith_search_space)
        subprocess.run(cmd, check=True, env=new_env)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)