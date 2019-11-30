import os, json
import numpy as np
import pandas as pd
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir')
parser.add_argument('--lei', dest='lei', action='store_true')

def main_lei(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    values = []
    for d, seed in product(datasets, seeds):
        path = os.path.join(
            args.output_dir,
            "bert_encoder_generator",
            d,
            "direct",
            "RANDOM_SEED=" + str(seed),
        )

        metrics_file_direct = os.path.join(path, 'metrics.json')
        if os.path.isfile(metrics_file_direct) :
            metrics = json.load(open(metrics_file_direct))
            metrics_1 = {k:v for k, v in metrics.items() if k.startswith('test_fscore') or k.startswith('test__fscore')}
            values.append({
                'dataset' : d, 'value' : np.mean(list(metrics_1.values())), 'rat_length' : metrics['test__rat_length']
            })


    values = pd.DataFrame(values)
    print(values.groupby(['dataset']).agg([np.median, np.std, np.max]))

def main_ours(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]
    saliency = ["wrapper", "simple_gradient"]
    rationale = ["top_k", "max_length"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    values = []

    for d, seed in product(datasets, seeds):
        path = os.path.join(
            args.output_dir,
            "bert_classification",
            d,
            "direct",
            "RANDOM_SEED=" + str(seed),
        )
        metrics_file_direct = os.path.join(path, 'metrics.json')
        if os.path.isfile(metrics_file_direct) :
            metrics = json.load(open(metrics_file_direct))
            metrics = {k:v for k, v in metrics.items() if k.startswith('test_fscore') or k.startswith('test__fscore')}
            values.append({
                'dataset' : d, 'saliency' : 'Base', 'rationale' : 'Base', 'extraction' : 'Base', 'value' : np.mean(list(metrics.values()))
            })
        else :
            print("Not found", metrics_file_direct)

    for d, s, r, seed in product(datasets, saliency, rationale, seeds):
        path = os.path.join(
            args.output_dir,
            "bert_classification",
            d,
            "direct",
            "RANDOM_SEED=" + str(seed),
            s + "_saliency",
            r + "_rationale",
            "direct",
        )

        metrics_file_direct = os.path.join(path, 'model_b', 'metrics.json')
        if os.path.isfile(metrics_file_direct) :
            metrics = json.load(open(metrics_file_direct))
            metrics = {k:v for k, v in metrics.items() if k.startswith('test_fscore') or k.startswith('test__fscore')}
            values.append({
                'dataset' : d, 'saliency' : s, 'rationale' : r, 'extraction' : 'direct', 'value' : np.mean(list(metrics.values()))
            })

        metrics_file_direct = os.path.join(path, 'bert_generator_saliency', 'direct', 'model_b', 'metrics.json')
        if os.path.isfile(metrics_file_direct) :
            metrics = json.load(open(metrics_file_direct))
            metrics = {k:v for k, v in metrics.items() if k.startswith('test_fscore') or k.startswith('test__fscore')}
            values.append({
                'dataset' : d, 'saliency' : s, 'rationale' : r, 'extraction' : 'crf', 'value' : np.mean(list(metrics.values()))
            })


    values = pd.DataFrame(values)
    print(values.groupby(['dataset', 'saliency', 'rationale', 'extraction']).agg([np.median, np.std, np.max]))
        

if __name__ == '__main__' : 
    args = parser.parse_args()
    if args.lei :
        main_lei(args)
    else :
        main_ours(args)