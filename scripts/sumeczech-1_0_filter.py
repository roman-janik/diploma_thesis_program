# Author: Roman Janík
# Script for filtering corrupt SumeCzech-NER 1.0 dataset train jsonl file.
#

import json
import sys


def main():
    train_split_file = "sumeczech-1.0-train.jsonl"
    train_split_out_file = "sumeczech-1.0-train_unique.jsonl"

    # Open the output file and load
    print("Loading previously downloaded data.", file=sys.stderr)
    datasets = {}
    dataset = "train"
    already_downloaded = 0

    datasets[dataset] = {
        "file": open(train_split_file.format(dataset), encoding="utf-8"),
        "out_file": open(train_split_out_file.format(dataset), "w", encoding="utf-8"),
        "md5s": set()
    }

    for i, line in enumerate(datasets[dataset]["file"]):
        assert line.endswith("\n"), "The last line of {} is not properly ended".format(
            train_split_file)
        try:
            entry = json.loads(line)
            if not entry["md5"] in datasets[dataset]["md5s"]:
                datasets[dataset]["md5s"].add(entry["md5"])
                datasets[dataset]["out_file"].write(json.dumps(
                    entry, ensure_ascii=False, sort_keys=True, indent=None, separators=(", ", ": ")) + "\n")
        except:
            print("Cannot decode the line {} from '{}'   - skipping line".format(
                i + 1, train_split_file.format(dataset)), file=sys.stderr)
    already_downloaded += len(datasets[dataset]["md5s"])
    datasets[dataset]["file"].close()
    datasets[dataset]["out_file"].close()

    print(f"Number of unique entries:   {already_downloaded}\n")


if __name__ == '__main__':
    main()
