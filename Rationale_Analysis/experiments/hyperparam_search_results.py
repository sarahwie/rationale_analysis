import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--exp-folder", type=str, required=True)
parser.add_argument("--config-keys", type=str, required=True)
parser.add_argument('--num-searches', type=int, required=True)
parser.add_argument("--metrics", type=str, required=True)
parser.add_argument("--minimize", dest="minimize", action="store_true")

def traverse_nested_dictionary(d, keys) :
    for k in keys :
        d = d[k]

    return d

def main(args):
    configs = []
    search_metrics = []
    for i in range(args.num_searches) :
        exp_name = os.path.join(args.exp_folder, "search_" + str(i))
        try :
            metrics = json.load(open(os.path.join(exp_name, 'metrics.json')))
            config = json.load(open(os.path.join(exp_name, 'config.json')))
            config = {x:traverse_nested_dictionary(config, x.split('.')) for x in args.config_keys.split(',')}

            metrics = {k:metrics[k] for k in args.metrics.split(',')}
            configs.append(config)
            search_metrics.append(metrics)
        except FileNotFoundError:
            continue

    metric_to_select_on = args.metrics.split(',')[0]
    metrics_selected = list(map(lambda x, y : (x, y[metric_to_select_on]), zip(configs, search_metrics)))
    best_value_set = max(metrics_selected, key=lambda x : x[1] * (-1 if args.minimize else 1))

    print(metrics_selected)
    print(best_value_set)

    json.dump(best_value_set, open(os.path.join(exp_name, 'best_values.json'), 'w'))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
