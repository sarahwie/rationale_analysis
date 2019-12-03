import os, json
import numpy as np
import pandas as pd
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--lei", dest="lei", action="store_true")


def main_lei(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    rationale = ["top_k", "max_length"]
    values = []
    for d, r, seed in product(datasets, rationale, seeds):
        path = os.path.join(
            args.output_dir,
            "bert_encoder_generator",
            d,
            "direct",
            "RANDOM_SEED=" + str(seed),
            r + "_rationale",
            "direct",
        )

        metrics_file_direct = os.path.join(path, "test_metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))
            metrics_1 = {k: v for k, v in metrics.items() if k.startswith("_fscore") or k.startswith("fscore")}
            values.append(
                {
                    "dataset": d,
                    "rationale": r,
                    "saliency": "-",
                    "extraction": "-",
                    "seed": seed,
                    "value": np.mean(list(metrics_1.values())),
                }
            )

    values = pd.DataFrame(values)
    idx = values.groupby(["dataset", "saliency", "rationale", "extraction"])["value"].transform(max) == values["value"]
    print(values[idx])

    values_g = values.groupby(["dataset", "saliency", "rationale", "extraction"]).agg(
        lambda x: "{:0.2f}".format(np.median(x))
        + " ("
        + "{:0.2f}".format(np.min(x))
        + "-"
        + "{:0.2f}".format(np.max(x))
        + ")"
    )

    print(values_g)
    print(values_g["value"].unstack(level=0).to_latex())


def main_ours(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]
    saliency = ["wrapper", "simple_gradient"]
    rationale = ["top_k", "max_length", "global_top_k", "global_contig"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    values = []

    for d, seed in product(datasets, seeds):
        path = os.path.join(args.output_dir, "bert_classification", d, "direct", "RANDOM_SEED=" + str(seed))
        metrics_file_direct = os.path.join(path, "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))
            metrics = {k: v for k, v in metrics.items() if k.startswith("test_fscore") or k.startswith("test__fscore")}
            values.append(
                {
                    "dataset": d,
                    "saliency": "Base",
                    "rationale": "Base",
                    "extraction": "Base",
                    "seed": seed,
                    "value": np.mean(list(metrics.values())),
                }
            )
        else:
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

        metrics_file_direct = os.path.join(path, "model_b", "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))
            metrics = {k: v for k, v in metrics.items() if k.startswith("test_fscore") or k.startswith("test__fscore")}
            values.append(
                {
                    "dataset": d,
                    "saliency": s,
                    "rationale": r,
                    "extraction": "direct",
                    "seed": seed,
                    "value": np.mean(list(metrics.values())),
                }
            )

        if r.startswith("global"):
            continue

        metrics_file_direct = os.path.join(path, "bert_generator_saliency", "direct", "model_b", "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))
            metrics = {k: v for k, v in metrics.items() if k.startswith("test_fscore") or k.startswith("test__fscore")}
            values.append(
                {
                    "dataset": d,
                    "saliency": s,
                    "rationale": r,
                    "extraction": "crf",
                    "seed": seed,
                    "value": np.mean(list(metrics.values())),
                }
            )

    values = pd.DataFrame(values)
    idx = values.groupby(["dataset", "saliency", "rationale", "extraction"])["value"].transform(max) == values["value"]
    print(values[idx])

    values_g = values.groupby(["dataset", "saliency", "rationale", "extraction"]).agg(
        lambda x: "{:0.2f}".format(np.median(x))
        + " ("
        + "{:0.2f}".format(np.min(x))
        + "-"
        + "{:0.2f}".format(np.max(x))
        + ")"
    )

    print(values_g)
    print(values_g["value"].unstack(level=0).to_latex())

    analyse_globality(values)

    return values


from scipy.stats import ttest_ind


def analyse_globality(values):
    m = {"top_k": "top_k", "max_length": "contiguous", "global_top_k": "top_k", "global_contig": "contiguous"}

    values = values[values.extraction == "direct"]
    values["global"] = values["rationale"].apply(lambda x: "global" in x)
    values["rationale"] = values["rationale"].apply(lambda x: m[x])

    values = values.groupby(["dataset", "saliency", "rationale"]).apply(
        lambda x: ttest_ind(x[x["global"] == "True"]['value'], x[x["global"] == "False"]['value'], equal_var=False)
    )
    print(values)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.lei:
        main_lei(args)
    else:
        main_ours(args)

