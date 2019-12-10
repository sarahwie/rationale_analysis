import os, json
datasets = ['evinf', 'SST', 'multirc', 'agnews']
search = ['search_' + str(i) for i in range(20)]
from itertools import product
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt


data = []
for d, s in product(datasets, search) :
    try :
        metrics = json.load(open(os.path.join(d, 'mu_lambda_search', str(s), 'metrics.json')))
        f1 = {k: v for k, v in metrics.items() if k.startswith("test_fscore") or k.startswith("test__fscore")}
        data.append({'dataset' : d, 'seed' : s,
            'time' : metrics['training_duration'], 'rat' : metrics['test__rat_length'], 'acc' : np.mean(list(f1.values()))
            })
    except :
        continue

data = pd.DataFrame(data)


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot(dataset) :
    plt.scatter(rand_jitter(data[data.dataset == dataset]['rat']), rand_jitter(data[data.dataset == dataset]['acc']), s=5)
    plt.xlabel('Average Rationale Length')
    plt.ylabel('Model Performance')
    plt.title(dataset.title())
    plt.savefig('/home/jain.sar/' + dataset + '.pdf')
