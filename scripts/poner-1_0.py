# Author: Roman Janík
# Script for local loading PONER 1.0 CoNNL dataset and converting it to Hugging Face dataset format.
#
# This script if a modified version of conll2003/conll2003.py script by HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
-
"""

_DESCRIPTION = """\
This is a Pero OCR NER 1.0 dataset in the CoNLL format. Each line in
the corpus contains information about one word/token. The first column is the actual
word, the second column is a Named Entity class in a BIO format. An empty line is a sentence separator.
"""

_TRAINING_FILE = "poner_train.conll"
_DEV_FILE = "poner_dev.conll"
_TEST_FILE = "poner_test.conll"


class Poner1_0ConllConfig(datasets.BuilderConfig):
    """BuilderConfig for PONER 1.0 CoNNL"""

    def __init__(self, **kwargs):
        """BuilderConfig for PONER 1.0 CoNNL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Poner1_0ConllConfig, self).__init__(**kwargs)


class Poner1_0Conll(datasets.GeneratorBasedBuilder):
    """PONER 1.0 CoNNL dataset."""

    BUILDER_CONFIGS = [
        Poner1_0ConllConfig(name="poner1_0conll", version=datasets.Version("1.0.0"),
                            description="PONER 1.0 CoNNL dataset"),
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
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://pero-ocr.fit.vutbr.cz",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, dataset_path="../../../datasets/poner1.0"):
        """Returns SplitGenerators."""
        data_files = {
            "train": os.path.join(dataset_path, _TRAINING_FILE),
            "dev": os.path.join(dataset_path, _DEV_FILE),
            "test": os.path.join(dataset_path, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
