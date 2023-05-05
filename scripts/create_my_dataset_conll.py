# Author: Roman Janík
# Script for creating a CoNLL format of my dataset from text files and Label Studio annotations.
#

import argparse
import json
import os

from nltk.tokenize import word_tokenize, sent_tokenize
from random import shuffle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file", required=True, help="Path to source Label Studio json annotations file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir for CoNLL dataset splits.")
    parser.add_argument("-p", "--split_ratios", required=True,
                        help="Ratios for train/dev/test splits, nums separated by ':'")
    parser.add_argument("-i", "--index", default=1, help="Start index for tasks/pages.")
    parser.add_argument("-d", "--drop_num", default=0, type=int,
                        help="Drop first x predictions. For non-document header entities.")
    args = parser.parse_args()
    return args


def parse_splits(split_str: str):
    split_list = split_str.split(":")
    assert len(split_list) == 3
    splits = {k: float(v) for k, v in zip(["train", "dev", "test"], split_list)}
    assert sum(splits.values()) == 1.0

    return splits


def process_text_annotations(text, annotations):
    entity_types_map = {
        "Personal names": "p",
        "Institutions": "i",
        "Geographical names": "g",
        "Time expressions": "t",
        "Artifact names/Objects": "o"
    }

    sentences = sent_tokenize(text, language="czech")
    sentences_t = [word_tokenize(x, language="czech") for x in sentences]

    sentences_idx = []
    start = 0
    for sentence in sentences_t:
        sentence_idx = []
        for word in sentence:
            end = start + len(word)
            sentence_idx.append({"word": word, "start": start, "end": end, "entity_type": "O"})
            start = end + 1 if text[end].isspace() else end
        sentences_idx.append(sentence_idx)

    for n_entity in annotations:
        begin = True
        done = False
        for sentence_idx in sentences_idx:
            for word_idx in sentence_idx:
                if word_idx["start"] >= n_entity["start"] and word_idx["end"] <= n_entity["end"] and begin:
                    word_idx["entity_type"] = "B-" + entity_types_map[n_entity["labels"][0]]
                    begin = False
                elif word_idx["start"] > n_entity["start"] and word_idx["end"] <= n_entity["end"]:
                    word_idx["entity_type"] = "I-" + entity_types_map[n_entity["labels"][0]]
                elif word_idx["end"] > n_entity["end"]:
                    done = True
                    break
            if done:
                break

    connl_sentences = []
    for sentence_idx in sentences_idx:
        connl_sentence = map(lambda w: w["word"] + " " + w["entity_type"], sentence_idx)
        connl_sentences.append("\n".join(connl_sentence))
    connl_sentences = "\n\n".join(connl_sentences)

    return connl_sentences


def main():
    args = get_args()

    print("Script for creating a CoNLL format of my dataset from text files and Label Studio annotations."
          "Script goes through page text files and their annotations json record. "
          "Output CoNLL dataset split files are saved to output directory.")

    splits_ratios = parse_splits(args.split_ratios)

    with open(args.source_file, encoding="utf-8") as f:
        annotations = json.load(f)

    print("Starting documents processing...")

    shuffle(annotations)

    splits = {}
    lower_bound = 0
    for k, ratio in splits_ratios.items():
        upper_bound = lower_bound + int(ratio * len(annotations))
        splits[k] = annotations[lower_bound:upper_bound]
        lower_bound = upper_bound

    for name, split in splits.items():
        with open(os.path.join(args.output_dir, "my_dataset_" + name + ".conll"), "w", encoding="utf-8") as f:
            for page in split:
                page_text_path = page["text"].replace("http://localhost:8081", "../../../datasets/my_dataset/data")
                with open(page_text_path, encoding="utf-8") as p_f:
                    page_text = p_f.read()
                    processed_page = process_text_annotations(page_text, page["ner"])
                    f.write(processed_page)

        print(f"Split '{name}' is processed.")


if __name__ == '__main__':
    main()
