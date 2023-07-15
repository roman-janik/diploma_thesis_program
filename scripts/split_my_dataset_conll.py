# Author: Roman Janík
# Script for creating splits of a CoNLL format of my dataset.
#

import argparse
import os

from random import shuffle
from itertools import islice


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file", required=True, help="Path to source Label Studio json annotations file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir for CoNLL dataset splits.")
    parser.add_argument("-m", "--shuffle", default=True, dest="shuffle", action="store_true",
                        help="Shuffle annotations.")
    parser.add_argument("--shuffle_chunk", type=int, default=1, dest="shuffle_chunk",
                        help="Shuffle annotations chunk. Chunk annotations, shuffle and join back.")
    parser.add_argument("-p", "--split_ratios", required=True,
                        help="Ratios for train/dev/test splits, nums separated by ':'")
    args = parser.parse_args()
    return args


def parse_splits(split_str: str):
    split_list = split_str.split(":")
    assert len(split_list) == 3
    splits = {k: float(v) for k, v in zip(["train", "dev", "test"], split_list)}
    assert sum(splits.values()) == 1.0

    return splits


def batched_func(iterable, chunk_size):
    iterator = iter(iterable)
    return list(iter(
        lambda: list(islice(iterator, chunk_size)),
        list()
    ))


def main():
    args = get_args()

    print("Script for creating a CoNLL format of my dataset from text files and Label Studio annotations. "
          "Script goes through page text files and their annotations json record. "
          "Output CoNLL dataset split files are saved to output directory.")

    splits_ratios = parse_splits(args.split_ratios)

    with open(args.source_file, encoding="utf-8") as f:
        content = f.read()
        annotations = content.split("\n\n")

    print(f"Total training examples: {len(annotations)}\nStarting documents processing...")

    if args.shuffle:
        if args.shuffle_chunk > 1:
            chunked_annotations = batched_func(annotations, args.shuffle_chunk)
            shuffle(chunked_annotations)
            annotations = [item for sublist in chunked_annotations for item in sublist]
        else:
            shuffle(annotations)

    splits = {}
    lower_bound = 0
    for k, ratio in splits_ratios.items():
        upper_bound = lower_bound + int(ratio * len(annotations))
        splits[k] = annotations[lower_bound:upper_bound]
        lower_bound = upper_bound

    for name, split in splits.items():
        with open(os.path.join(args.output_dir, "poner_" + name + ".conll"), "w", encoding="utf-8") as f:
            for example in split:
                f.write(example + "\n\n")

        print(f"Split '{name}' is processed.")


if __name__ == '__main__':
    main()
