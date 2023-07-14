# Author: Roman Janík
# Script for removing whitespace at the start of entity annotation span in text. Label Studio marks additional
# whitespace at the start of entity, which do not belong to it (usually newline). Phantom entities created by
# a model with only one newline char in front of real entity are removed. Single non alphanumeric entities are
# removed too.
#

import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file", required=True, help="Path to source Label Studio json annotations file.")
    args = parser.parse_args()
    return args


def save(annotations_file, annotations):
    with open(annotations_file, "w", encoding='utf8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print("Annotations were saved!")


def main():
    args = get_args()

    print("Script for removing whitespace at the start of entity annotation span in text. "
          "Script goes through page text files and their annotations json record. "
          "Whitespace and quotes at the start of entity are removed and annotations are saved to the same file."
          "Single non alphanumeric entities are removed too.")

    with open(args.source_file, encoding="utf-8") as f:
        annotations = json.load(f)

    short_count = 0
    rem_count = 0
    total_entities = 0
    for i, page in enumerate(annotations):
        page_text_path = annotations[i]["text"].replace(
            "http://localhost:8081", "../../../datasets/poner1.0/data")
        with open(page_text_path, encoding="utf-8") as p_f:
            page_text = p_f.read()

        for j, n_entity in enumerate(page["ner"]):
            n_entity_text = page_text[n_entity["start"]:n_entity["end"]]
            if n_entity_text[0].isspace():
                if len(n_entity_text) == 1:
                    rem_count += 1
                    annotations[i]["ner"].pop(j)
                else:
                    short_count += 1
                    annotations[i]["ner"][j]["start"] += 1
            if len(n_entity_text) == 1 and not n_entity_text.isalnum():
                rem_count += 1
                annotations[i]["ner"].pop(j)
            if n_entity_text[0] == "„":
                short_count += 1
                annotations[i]["ner"][j]["start"] += 1
        total_entities += len(annotations[i]["ner"])

    save(args.source_file, annotations)
    average_entities = total_entities / len(annotations)
    print(f"All pages were processed! Num corrected entities: {short_count}, num removed phantom entities: {rem_count}"
          f"\nAverage entities in 1 page: {average_entities}, Total entities: {total_entities}")


if __name__ == '__main__':
    main()
