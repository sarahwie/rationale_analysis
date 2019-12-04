import os, json
import numpy as np
import pandas as pd
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--lei", dest="lei", action="store_true")
parser.add_argument("--dataset")

def main(args):
    datasets = [args.dataset]
    saliency = ["wrapper", "simple_gradient"]
    rationale = ["top_k", "max_length"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    values = []

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

        metrics_file_direct = os.path.join(path, "dev.jsonl")
        direct_rationales = [json.loads(line) for line in open(metrics_file_direct)]
        
        direct_span_len = [sum([span['span'][1] - span['span'][0] 
                     for span in doc['rationale']['spans']]) / len(doc['rationale']['spans'])
                     for doc in direct_rationales]

        metrics_file_direct = os.path.join(path, "bert_generator_saliency", "direct", "dev.jsonl")
        crf_rationales = [json.loads(line) for line in open(metrics_file_direct)]

        crf_span_len = []
        for doc in crf_rationales :
            document = doc['metadata']['tokens']
            rat = doc['rationale']['document'].split()
            if len(rat) == 0 :
                breakpoint()
            rat_tokens = [0]
            spans = 0
            j = 0
            for i, t in enumerate(document) :
                if j < len(rat) and t == rat[j] :
                    if rat_tokens[-1] == 0 :
                        spans += 1
                    rat_tokens.append(1)
                    j += 1
                else :
                    rat_tokens.append(0)

            assert j == len(rat), breakpoint()

            crf_span_len.append(sum(crf_span_len) / spans)

        breakpoint()

        values.append({
            'dataset' : d,
            'saliency' : s,
            'rationale' : r,
            'seed' : seed,
            'diff' : np.mean(np.array(crf_span_len) - np.array(direct_span_len))
        })

    values = pd.DataFrame(values)
    breakpoint()

if __name__ == '__main__' :
    args = parser.parse_args()
    main(args)