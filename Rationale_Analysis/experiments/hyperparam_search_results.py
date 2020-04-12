import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--exp-folder", type=str, required=True)
parser.add_argument('--num-searches', type=int, required=True)

def traverse_nested_dictionary(d, keys) :
    for k in keys :
        d = d[k]

    return d

import pandas as pd

def main(args):
    mu = []
    lambdav = []
    rtype = []
    value = []

    for i in range(args.num_searches) :
        exp_name = os.path.join(args.exp_folder, "search_" + str(i))
        try :
            for method in ['top_k', 'max_length'] :
                config = json.load(open(os.path.join(exp_name, 'config.json')))
                mu.append(config['model']['reg_loss_mu'])
                lambdav.append(config['model']['reg_loss_lambda'])
                metrics = json.load(open(os.path.join(exp_name, 'dev_metrics.json')))['validation_metric']
                value.append(metrics)
                rtype.append(method)
        except FileNotFoundError:
            continue

    df = pd.DataFrame({'mu' : mu, 'lambda' : lambdav, 'type' : rtype, 'value' : value})

    breakpoint()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
