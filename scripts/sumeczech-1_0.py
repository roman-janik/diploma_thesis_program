# Author: Roman Janík
# Script for local loading SumeCzech-NER 1.0 dataset and converting it to Hugging Face dataset format.
#

import os
import json
import time

import datasets

from glob import glob
from typing import List
from nltk.tokenize import word_tokenize

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{SumeCzech-NER,
        title={Text Summarization of Czech News Articles Using Named Entities},
        volume={116},
        ISSN={0032-6585},
        url={http://dx.doi.org/10.14712/00326585.012},
        DOI={10.14712/00326585.012},
        number={1},
        journal={Prague Bulletin of Mathematical Linguistics},
        publisher={Charles University in Prague, Karolinum Press},
        author={Marek, Petr and M{\"u}ller, {\v S}t{\v e}p{\'a}n and Konr{\'a}d, Jakub and Lorenc, Petr and Pichl, Jan \
        and {\v S}ediv{\'y}, Jan},
        year={2021},
        month={Apr},
        pages={5–26}
    }
"""

_DESCRIPTION = """\
SumeCzech-NER contains named entity annotations of SumeCzech 1.0 (Straka et al. 2018, SumeCzech: Large Czech News-Based
Summarization Dataset).

Format

The dataset is split into four files. Files are in jsonl format. There is one JSON object on each line of the file. The
most important fields of JSON objects are:

 - dataset: train, dev, test, oodtest
 - ne_abstract: list of named entity annotations of article's abstract
 - ne_headline: list of named entity annotations of article's headline
 - ne_text: list of name entity annotations of article's text
 - url: article's URL that can be used to match article across SumeCzech and SumeCzech-NER

Annotations
We used SpaCy's NER model trained on CoNLL-based extended CNEC 2.0. The model achieved a 78.45 F-Score on the dataset's
testing set. The annotations are in IOB2 format. The entity types are: Numbers in addresses, Geographical names,
Institutions, Media names, Artifact names, Personal names, and Time expressions.

http://hdl.handle.net/11234/1-3505
"""


class SumeCzechNER(datasets.GeneratorBasedBuilder):
    """Page segmentation dataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="sumeczech-ner", version=VERSION, description="SumeCzech-NER.")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-P",
                                "I-P",
                                "B-I",
                                "I-I",
                                "B-G",
                                "I-G",
                                "B-T",
                                "I-T",
                                "B-O",
                                "I-O",
                                "B-M",
                                "I-M",
                                "B-A",
                                "I-A"
                            ]
                        )
                    ),
                }
            ),
            homepage="http://hdl.handle.net/11234/1-3505",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, dataset_path="../../../datasets/sumeczech-1.0-ner"):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"dataset_path": dataset_path,
                                                                           "split": "train", "skip": [3]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"dataset_path": dataset_path,
                                                                                "split": "dev", "skip": [0, 1]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"dataset_path": dataset_path,
                                                                          "split": "test", "skip": [0, 1, 2]}),
            datasets.SplitGenerator(name=datasets.Split("oodtest"), gen_kwargs={"dataset_path": dataset_path,
                                                                                "split": "oodtest", "skip": [0, 1, 2]})
        ]

    def _generate_examples(self, dataset_path, split, skip):
        logger.info("⏳ Generating examples from = %s", split)
        data_file_template = "sumeczech-1.0-ner-{}.jsonl"
        sumeczech_dir = "../sumeczech-1.0"
        sumeczech_split_file = "sumeczech-1.0-{}.jsonl"
        sumeczech_entries = {}
        guid = 0

        # load SumeCzech split file
        with open(os.path.join(dataset_path, sumeczech_dir, sumeczech_split_file.format(split)), encoding="utf-8") \
                as sumeczech_f:
            for line in sumeczech_f:
                entry = json.loads(line)
                text = entry["headline"] + "\n" + entry["abstract"] + "\n" + entry["text"]
                sumeczech_entries[entry["md5"]] = text

        for dataset_file in glob(dataset_path + "/*.jsonl"):
            if os.path.basename(dataset_file) in [data_file_template.format(str(num)) for num in skip]:
                continue

            with open(dataset_file) as dataset_f:
                for line in dataset_f:
                    entry = json.loads(line)
                    if entry["dataset"] != split:
                        continue
                    ner_tags = entry["ne_headline"] + entry["ne_abstract"] + entry["ne_text"]

                    try:
                        sumeczech_text = sumeczech_entries[entry["md5"]]
                    except KeyError:
                        print("Example with md5 '{}' not found in SumeCzech dataset split: {}".format(
                            entry["md5"], split))
                        continue
                    tokens = self.tokenize(sumeczech_text)

                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags
                    }
                    guid += 1

    def tokenize(self, text: str) -> List[str]:
        for mark in ('.', ',', '?', '!', '-', '–', '/'):
            text = text.replace(mark, f' {mark} ')
        tokens = word_tokenize(text)
        return tokens
