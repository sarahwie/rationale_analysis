import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
pd.set_option('display.max_colwidth', -1)



def main(args):
    keys = ["train", "dev", "test"]
    data = {k: None for k in keys}

    latex_data = {k: {} for k in keys}

    for k in keys:
        df = pd.read_json(os.path.join(args.dataset, "data", k + ".jsonl"), lines=True)
        document_length = df["document"].apply(lambda x: len(x.split()))
        desc = document_length.describe()
        data[k] = desc

        latex_data[k]["N"] = str(len(df))
        latex_data[k]["Doc Length"] = str(int(desc["mean"].round(0))) + " / " + str(int(desc["max"]))
        

    print("Document Only Length")
    print("=" * 20)
    print(pd.DataFrame(data))

    if "query" in df.columns:
        for k in keys:
            df = pd.read_json(os.path.join(args.dataset, "data", k + ".jsonl"), lines=True)
            query_length = df["query"].apply(lambda x: len(x.split()))
            desc = query_length.describe()
            data[k] = desc
            latex_data[k]["Query Length"] = str(int(desc["mean"].round(0))) + " / " + str(int(desc["max"]))

        print("Query Length")
        print("=" * 20)
        print(pd.DataFrame(data))

    if "rationale" in df.columns:
        for k in keys:
            df = pd.read_json(os.path.join(args.dataset, "data", k + ".jsonl"), lines=True)
            rationale_length = df["rationale"].apply(lambda x: sum([y[1] - y[0] for y in x]))
            data[k] = rationale_length.describe()

        print("Rationale Length Absolute")
        print("=" * 20)
        print(pd.DataFrame(data))

        for k in keys:
            df = pd.read_json(os.path.join(args.dataset, "data", k + ".jsonl"), lines=True)
            rationale_length = df["rationale"].apply(lambda x: sum([y[1] - y[0] for y in x]))
            document_length = df["document"].apply(lambda x: len(x.split()))
            desc = (rationale_length / document_length).describe()
            data[k] = desc
            latex_data[k]["Rationale Length"] = str(desc['mean'].round(2)) + ' / ' + str(desc['max'].round(2))

        print("Rationale Length Relative")
        print("=" * 20)
        print(pd.DataFrame(data))

    if "label" in df.columns:
        for k in keys:
            df = pd.read_json(os.path.join(args.dataset, "data", k + ".jsonl"), lines=True)
            data[k] = df["label"].value_counts() / len(df)

            latex_data[k]["Label Dist"] = " / ".join([str(x) for x in data[k].round(2).values])

        print("/".join([str(x) for x in data[k].index]))
        print("Label Distribution")
        print("=" * 20)
        print(pd.DataFrame(data))

    print(pd.DataFrame(latex_data).T.to_latex())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

