import subprocess
import os

search_space = {
    'KEEP_PROB' : [0.2, 0.4, 0.6, 0.8, 1.0],
    'RANDOM_SEED' : [1000, 2000, 3000, 4000, 5000]
}

import json

default_values = json.load(open('Rationale_Analysis/default_values.json'))

def main() :
    pass
