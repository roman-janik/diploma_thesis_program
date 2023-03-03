# Author: Roman Janík
# Script for preparing CNEC 2.0 CoNNL and CHNEC 1.0 datasets and converting them to Hugging Face datasets format.
#


import os

import datasets
import nltk


# delete ner types media and numbers in addresses
def lower_ner_tag_types(examples):
    for i, example_ner_tags in enumerate(examples["ner_tags"]):
        for j, ner_tag in enumerate(example_ner_tags):
            if ner_tag > 10:
                examples["ner_tags"][i][j] = 0
    return examples


nltk.download('punkt')

datasets_path = "../../../datasets"
cnec_dir = "cnec2.0_extended"
chnec_dir = "chnec1.0"
sumeczech_dir = "sumeczech-1.0-ner"

# # load CNEC 2.0 CoNLL dataset with loading script
# cnec_dataset = datasets.load_dataset("cnec2_0_conll.py")
#
# # transform CNEC 2.0 CoNLL dataset into CHNEC 1.0 format -> change in NER tags
# cnec_dataset = cnec_dataset.cast_column("ner_tags", datasets.Sequence(
#     datasets.ClassLabel(
#         names=[
#             "O",
#             "B-p",
#             "I-p",
#             "B-i",
#             "I-i",
#             "B-g",
#             "I-g",
#             "B-t",
#             "I-t",
#             "B-o",
#             "I-o"
#         ]
#     )
# )
#                                         )
# cnec_dataset = cnec_dataset.map(lower_ner_tag_types, batched=True)
#
# # remove columns irrelevant to NER task
# cnec_dataset = cnec_dataset.remove_columns(["lemmas", "morph_tags"])
#
# # save CNEC 2.0 CoNLL dataset in Hugging Face Datasets format (not tokenized)
# cnec_dataset.save_to_disk(os.path.join(datasets_path, cnec_dir))
#
#
# # load CHNEC 1.0 dataset with loading script
# chnec_dataset = datasets.load_dataset("chnec1_0.py")
#
# # remove columns irrelevant to NER task
# chnec_dataset = chnec_dataset.remove_columns(["lemmas", "language"])
#
# # save CHNEC 1.0 dataset in Hugging Face Datasets format (not tokenized)
# chnec_dataset.save_to_disk(os.path.join(datasets_path, chnec_dir))


# load SumeCzech-NER 1.0 dataset with loading script
sumeczech_dataset = datasets.load_dataset("sumeczech-1_0.py")

# transform SumeCzech-NER 1.0 dataset into CHNEC 1.0 format -> change in NER tags
sumeczech_dataset = sumeczech_dataset.cast_column("ner_tags", datasets.Sequence(
    datasets.ClassLabel(
        names=[
            "O",
            "B-p",
            "I-p",
            "B-i",
            "I-i",
            "B-g",
            "I-g",
            "B-t",
            "I-t",
            "B-o",
            "I-o"
        ]
    )
)
                                        )
sumeczech_dataset = sumeczech_dataset.map(lower_ner_tag_types, batched=True)

# save SumeCzech-NER 1.0 dataset in Hugging Face Datasets format (not tokenized)
sumeczech_dataset.save_to_disk(os.path.join(datasets_path, sumeczech_dir))
