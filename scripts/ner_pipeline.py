# Author: Roman Janík
# Script for using a NER model in Transformers pipeline.
#

import argparse
import os
import pprint

import transformers
import sys


class NerPipeline:
    def __init__(self, tokenizer_path: str, model_path: str):
        self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(tokenizer_path, model_max_length=512,
                                                                           add_prefix_space=True)
        self.token_classifier = transformers.pipeline(
            "token-classification", model=model_path, tokenizer=self.tokenizer, aggregation_strategy="simple"
        )
        parent_path = os.path.dirname(os.path.normpath(model_path))
        self.model_version = os.path.basename(parent_path)

    def __call__(self, text, *args, **kwargs):
        return self.token_classifier(text)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", dest="example", action="store_true", default=False,
                        help="Print example text.")
    parser.add_argument("-t", "--tokenizer_path", required=True, help="Path to tokenizer.")
    parser.add_argument("-m", "--model_path", required=True, help="Path to model for generation of predictions.")
    args = parser.parse_args()
    return args


def main():
    model_path = "../../results/results_2022-11-20/2022-11-21-00-03-linear_lr_3e5_10_epochs_wr-3h/model"
    example_text = "Česká republika vznikla 1.ledna 1993 po rozdělení společného státu Čechů a Slováků. První lednový " \
                   "den je v České republice i na Slovensku státním svátkem.\nČeská národní banka (ČNB) je centrální " \
                   "banka České republiky a orgán, který vykonává dohled nad finančním trhem v zemi. Svatovítská " \
                   "koruna je uložena v Praze. "
    args = get_args()

    ner_pipeline = NerPipeline(args.tokenizer_path, args.model_path)

    if args.example:
        print(f"Example text:\n{example_text}")
        pprint.pprint(ner_pipeline(example_text))

    print("Type your text:\n")
    text = sys.stdin.read()
    result = ner_pipeline(text)
    print(result["input_ids"])

    return


if __name__ == "__main__":
    main()
