# Author: Roman Janík
# Script for local loading page segmentation dataset and converting it to Hugging Face dataset format.
#

import os

import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = ""

_DESCRIPTION = """\
Small dataset of 50 examples for segmenting document pages. Model will segment original document page scan 
and added header and footer, in order to use segmentation map to remove header and footer. Source of data 
are Czech chronicles from PERO-OCR project. The data consist of images and their annotation masks. They are stored
in their respective folders 'images' and 'masks'. Images are stored in .jpg and masks in .png formats 
with the same name. Images are of various resolution.
"""


class PageSegmentation(datasets.GeneratorBasedBuilder):
    """Page segmentation dataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="page_segmentation", version=VERSION, description="Page segmentation dataset.")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "mask": datasets.Image()
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, dataset_path="../../../datasets/page_segmentation_dataset/data"):
        """Returns SplitGenerators."""
        data_files = {
            "train": os.path.join(dataset_path, "train"),
            "dev": os.path.join(dataset_path, "validation")
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"dir_path": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"dir_path": data_files["dev"]})
        ]

    def _generate_examples(self, dir_path):
        logger.info("⏳ Generating examples from = %s", dir_path)
        image_dir = os.path.join(dir_path, "images")
        mask_dir = os.path.join(dir_path, "masks")
        for idx, image_file in enumerate(os.listdir(image_dir)):
            mask_file = image_file.replace(".jpg", ".png")
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, mask_file)
            with open(image_path, "rb") as image_f, open(mask_path, "rb") as mask_f:
                yield idx, {
                    "image": {"path": image_path, "bytes": image_f.read()},
                    "mask": {"path": mask_path, "bytes": mask_f.read()}
                }
