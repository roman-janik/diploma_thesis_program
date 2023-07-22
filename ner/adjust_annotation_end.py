# Author: Roman Janík
# Script for adjusting the end of entity annotation span in text. Label Studio marks additional characters
# at the end of entity, which do not belong to it (usually dot or comma).
#

import argparse
import json
import re

from pynput import keyboard
from pynput.keyboard import Key


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file", required=True, help="Path to source Label Studio json annotations file.")
    args = parser.parse_args()
    return args


def save(annotations_file, annotations):
    with open(annotations_file, "w", encoding='utf8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print("Annotations were saved!")


def auto_correct(n_entity_text, n_entity_type):
    if len(n_entity_text) >= 2 and n_entity_text[-2].isalnum() and n_entity_text[-1] == "\n":
        return 1

    if n_entity_type == "Artifact names/Objects":
        monetary_u = ["zl.", "kr.", "zl. r. m.", "kr. r. m.", "tol.", "zl. r. č.", "h.", "hal.", "fl.", "gr."]
        if n_entity_text in monetary_u:
            return 0
        to_be_determined = ["Uh.", "př.", "č.", "R.", "lt.", "r.", "J. K. K.", "sv. Karla\nBor.",
                            "čís. 80. Sov. zák. a nař.", "čís. 80\nSb. z. a n."]
        if n_entity_text in to_be_determined:
            return -1
        if n_entity_text.endswith("Sb."):
            return 0
        if len(n_entity_text) >= 2 and n_entity_text[-2].isalnum():
            return 1

    if n_entity_type == "Time expressions":
        # year
        if re.search("^[0-9]{4}$", n_entity_text[:-1]):
            return 1
        # single day
        if re.search("^[0-9][.]$|^[0-9]{2}[.]$", n_entity_text):
            return 0
        # single day with comma
        if re.search("^[0-9][,]$|^[0-9]{2}[,]$", n_entity_text):
            return 1
        # day and month
        if re.search("^[0-9][.][ ]?[0-9][.]$|^[0-9][.][ ]?[0-9]{2}[.]$|^[0-9]{2}[.][ ]?[0-9][.]$|^[0-9]{2}[.][ ]?["
                     "0-9]{2}[.]$", n_entity_text):
            return 0
        # year span
        if re.search("^[0-9]{4}[ ]?[-][ ]?[0-9]{4}$|^[0-9]{4}[ ]?[-][ ]?[0-9]{2}$", n_entity_text[:-1]):
            return 1
        # date with "hod." or "t.r." at the end
        if n_entity_text.endswith(("hod.", "t.r.", "t. r.", "t.m.", "t. m.")):
            return 0

    if n_entity_type == "Geographical names":
        # street name correct
        if n_entity_text.endswith(("ul.", "tř.", "nám.")):
            return 0
        # street name incorrect
        if n_entity_text.endswith(("ul.,", "tř.,", "nám.,")):
            return 1

    if n_entity_type == "Personal names":
        # names ending with "ml." or "st." correct
        if n_entity_text.endswith(("ml.", "st.")):
            return 0
        # names ending with "ml." or "st." incorrect
        if n_entity_text.endswith(("ml.,", "st.,")):
            return 1

    if n_entity_type == "Institutions":
        # M.N.V., O.N.V., K.N.V., N.J., N.F., J.Z.D.
        if re.search("^[M][.][ ]?[N][.][ ]?[V][.]$|^[O][.][ ]?[N][.][ ]?[V][.]$|^[K][.][ ]?[N][.][ ]?[V][.]$|^[N][.][ "
                     "]?[J][.]$|^[N][.][ ]?[F][.]|^[J][.][ ]?[Z][.][ ]?[D][.]$", n_entity_text):
            return 0
        # M.N.V., O.N.V., K.N.V., N.J., N.F., J.Z.D. with additional char
        if re.search(
                "^[M][.][ ]?[N][.][ ]?[V][.]$|^[O][.][ ]?[N][.][ ]?[V][.]$|^[K][.][ ]?[N][.][ ]?[V][.]$|^[N][.][ "
                "]?[J][.]$|^[N][.][ ]?[F][.]|^[J][.][ ]?[Z][.][ ]?[D][.]$", n_entity_text[:-1]):
            return 1
        if n_entity_text[:-1] in ["JZD", "KSČ", "Ksč", "ksč", "Kčs", "ksč", "NF", "MNV", "ONV", "KNV"]:
            return 1

    if n_entity_type in ["Geographical names", "Personal names", "Institutions"]:
        # shorten entities with non-dot char at the end
        if n_entity_text[-1] != ".":
            words = n_entity_text[:-1].split()
            if all([word.isalnum() for word in words]):
                return 1

    return -1


def main():
    args = get_args()

    print("Script for adjusting the end of entity annotation span in text. "
          "Script goes through page text files and their annotations json record. "
          "End of entity is adjusted manually or automatically is possible and annotations are saved to the same file."
          "Adjusting starts at last adjusted entity, controls: right arrow - next entity, left arrow - previous entity,"
          "up arrow - +1 length, down arrow - -1 length, s - save"
          "automatic save after 10 entities\n")

    def on_key_release(key):
        nonlocal annotations, n_entity_idx, page_idx, page_text, stop_edit, adjusted_n_entities

        # next entity
        if key == Key.right:
            annotations[page_idx]["ner"][n_entity_idx]["adjusted"] = True
            n_entity_idx += 1
            adjusted_n_entities += 1
            exit()
        # previous adjusted entity
        elif key == Key.left:
            while n_entity_idx > 0:
                n_entity_idx -= 1
                if "adjusted" in annotations[page_idx]["ner"][n_entity_idx].keys():
                    annotations[page_idx]["ner"][n_entity_idx]["adjusted"] = False
                    adjusted_n_entities -= 1
                    break
            exit()
        # +1 length
        elif key == Key.up:
            if annotations[page_idx]["ner"][n_entity_idx]["end"] + 1 == len(page_text):
                print("Entity span cannot be prolonged, end of page text is reached!")
            else:
                annotations[page_idx]["ner"][n_entity_idx]["end"] += 1
                print("Entity length + 1, end:   {}".format(annotations[page_idx]["ner"][n_entity_idx]["end"]))
        # -1 length
        elif key == Key.down:
            if annotations[page_idx]["ner"][n_entity_idx]["end"] - 1 == 0:
                print("Entity span cannot be shortened, start of page text is reached!")
            else:
                annotations[page_idx]["ner"][n_entity_idx]["end"] -= 1
                print("Entity length - 1, end:   {}".format(annotations[page_idx]["ner"][n_entity_idx]["end"]))
        # save
        elif key == Key.esc:
            save(args.source_file, annotations)
            print("Editing stopped!")
            stop_edit = True
            exit()

    with open(args.source_file, encoding="utf-8") as f:
        annotations = json.load(f)

    page_idx = 0
    n_entity_idx = 0
    for i, page in enumerate(annotations):
        if "adjusted" not in page.keys() or not page["adjusted"]:
            page_idx = i
            for j, n_entity in enumerate(page["ner"]):
                if "adjusted" not in n_entity.keys():
                    n_entity_idx = j
                    break
            break

    adjusted_pages = 0
    adjusted_n_entities = 1
    stop_edit = False
    while True:
        page_text_path = annotations[page_idx]["text"].replace(
            "http://localhost:8081", "../../../datasets/poner1.0/data")
        with open(page_text_path, encoding="utf-8") as p_f:
            page_text = p_f.read()
        page_name = annotations[page_idx]["page_name"]
        print(f"Page:   {adjusted_pages}\n{page_name}\n{page_text}\n\nEntities without alphanum end:\n")

        while True:
            n_entity = annotations[page_idx]["ner"][n_entity_idx]
            n_entity_text = page_text[n_entity["start"]:n_entity["end"]]
            n_entity_type = n_entity["labels"][0]
            if not n_entity_text[-1].isalnum() or ("adjusted" in n_entity.keys() and not n_entity["adjusted"]):
                # try auto correct function, if auto correction is not possible, manual correction is applied
                end_shift = auto_correct(n_entity_text, n_entity_type)
                if end_shift == -1 or ("adjusted" in n_entity.keys() and not n_entity["adjusted"]):
                    context_start = 0 if n_entity["start"] - 100 < 0 else n_entity["start"] - 100
                    context_end = len(page_text) - 1 if n_entity["start"] + 100 >= len(page_text) \
                        else n_entity["start"] + 100
                    print(
                        f"\n{n_entity_text}\n------------------------------\n{n_entity_type}\n"
                        f"------------------------------\n{page_text[context_start:context_end]}\n")
                    with keyboard.Listener(on_release=on_key_release) as listener:
                        listener.join()
                    if stop_edit:
                        return
                else:
                    # auto correction application
                    annotations[page_idx]["ner"][n_entity_idx]["end"] -= end_shift
                    annotations[page_idx]["ner"][n_entity_idx]["auto_adjusted"] = True
                    n_entity_idx += 1
                    adjusted_n_entities += 1

                # auto save
                if adjusted_n_entities % 10 == 0:
                    save(args.source_file, annotations)
            else:
                n_entity_idx += 1
            if n_entity_idx == len(annotations[page_idx]["ner"]):
                n_entity_idx = 0
                break

        adjusted_pages += 1

        annotations[page_idx]["adjusted"] = True
        page_idx += 1
        if page_idx == len(annotations):
            save(args.source_file, annotations)
            break

    print("All pages were adjusted!")


if __name__ == '__main__':
    main()
