# Author: Roman Janík
# Script for creating a CoNLL format of my dataset from text files and Label Studio annotations.
#

import argparse
import json
import os

from nltk.tokenize import word_tokenize, sent_tokenize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file", required=True, help="Path to source Label Studio json annotations file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir for CoNLL dataset splits.")
    args = parser.parse_args()
    return args


def fix_quotes(sentence):
    return map(lambda word: word.replace('``', '"').replace("''", '"'), sentence)


def process_text_annotations(text, annotations):
    entity_types_map = {
        "Personal names": "p",
        "Institutions": "i",
        "Geographical names": "g",
        "Time expressions": "t",
        "Artifact names/Objects": "o"
    }

    sentences = sent_tokenize(text, language="czech")
    sentences_t = [fix_quotes(word_tokenize(x, language="czech")) for x in sentences]

    sentences_idx = []
    start = 0
    for i, c in enumerate(text):
        if not c.isspace():
            start = i
            break

    for sentence in sentences_t:
        sentence_idx = []
        for word in sentence:
            end = start + len(word)
            sentence_idx.append({"word": word, "start": start, "end": end, "entity_type": "O"})
            for i, _ in enumerate(text):
                if end + i < len(text) and not text[end + i].isspace():
                    start = end + i
                    break
        sentences_idx.append(sentence_idx)

    for n_entity in annotations:
        begin = True
        done = False
        for sentence_idx in sentences_idx:
            for word_idx in sentence_idx:
                if word_idx["start"] >= n_entity["start"]\
                        and (word_idx["end"] <= n_entity["end"]
                             or (not text[word_idx["end"]-1].isalnum()) and len(word_idx["word"]) > 1) and begin:
                    word_idx["entity_type"] = "B-" + entity_types_map[n_entity["labels"][0]]
                    begin = False
                    if word_idx["end"] >= n_entity["end"]:
                        done = True
                        break
                elif word_idx["start"] > n_entity["start"] and (word_idx["end"] <= n_entity["end"]
                    or (not text[word_idx["end"]-1].isalnum() and text[word_idx["start"]].isalnum())):
                    word_idx["entity_type"] = "I-" + entity_types_map[n_entity["labels"][0]]
                    if word_idx["end"] >= n_entity["end"]:
                        done = True
                        break
                elif word_idx["end"] > n_entity["end"]:
                    done = True
                    break
            if done:
                break

    conll_sentences = []
    for sentence_idx in sentences_idx:
        conll_sentence = map(lambda w: w["word"] + " " + w["entity_type"], sentence_idx)
        conll_sentences.append("\n".join(conll_sentence))
    conll_sentences = "\n\n".join(conll_sentences)

    return conll_sentences


def main():
    args = get_args()

    print("Script for creating a CoNLL format of my dataset from text files and Label Studio annotations."
          "Script goes through page text files and their annotations json record. "
          "Output CoNLL dataset file is saved to output directory.")

    with open(args.source_file, encoding="utf-8") as f:
        annotations = json.load(f)

    print("Starting documents processing...")

    with open(os.path.join(args.output_dir, "poner.conll"), "w", encoding="utf-8") as f:
        for page in annotations:
            page_text_path = page["text"].replace("http://localhost:8081", "../../../datasets/poner1.0/data")
            with open(page_text_path, encoding="utf-8") as p_f:
                page_text = p_f.read()
                processed_page = process_text_annotations(page_text, page["ner"])
                f.write(processed_page + "\n\n")

    print("Annotations are processed.")


if __name__ == '__main__':
    main()
