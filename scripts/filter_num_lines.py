# Author: Roman Janík
# Script for filtering a dataset, files with <= 8 lines are deleted.
#

import argparse
import os

from glob import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="Path to dataset dir.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    for filename in glob(os.path.normpath(args.dataset_dir + "/*.txt")):
        with open(filename, encoding="utf-8") as file:
            for lines_count, line in enumerate(file):
                pass
            lines_count += 1
        if lines_count <= 8:
            os.remove(filename)


if __name__ == "__main__":
    main()
