# Author: Roman Janík
# Script for preparing Pero OCR Books and Pero OCR Periodicals datasets and converting them to Hugging Face datasets
# format.
#


import os

from datasets import load_dataset, Version

datasets_path = "../../../datasets"
pero_ocr_books_dir = "pero_ocr_books"
pero_ocr_periodicals_dir = "pero_ocr_periodicals"
pero_ocr_corpus_dir = "pero_ocr_corpus"
books_text = "books/text"
periodicals_text = "periodicals/text"

# pero_ocr_books dataset info
BOOKS_VERSION = Version("1.0.0")
BOOKS_DESCRIPTION = """\
Pero OCR Books contains scanned and OCR-processed pages of books from PERO-OCR project. Pages are filtered
by OCR model confidence outputs. Confidence of each paragraph / ocr-region is calculated from line confidences.
Paragraphs with confidence <= 0.65 were filtered out. Pages with total lines <= 8 were filtered out too.
Pages were additionally filtered for language, keeping only Czech language. The decision if a page is in Czech language
or not is taken from first 5 paragraphs. If majority of these first 5 paragraphs are in Czech, page is included.
A language classifier model was employed. 

https://pero-ocr.fit.vutbr.cz
"""

# pero_ocr_periodicals dataset info
PERIODICALS_VERSION = Version("1.0.0")
PERIODICALS_DESCRIPTION = """
Pero OCR Periodicals contains scanned and OCR-processed pages of periodicals from PERO-OCR project. Pages are filtered
by OCR model confidence outputs. Confidence of each paragraph / ocr-region is calculated from line confidences.
Paragraphs with confidence <= 0.65 were filtered out. Pages with total lines <= 8 were filtered out too.

https://pero-ocr.fit.vutbr.cz
"""

# load from a directory
books_dataset = load_dataset("text", data_dir=os.path.join(datasets_path, pero_ocr_corpus_dir, books_text),
                             sample_by="document", name="pero_ocr_books", version=BOOKS_VERSION,
                             description=BOOKS_DESCRIPTION)
periodicals_dataset = load_dataset("text", data_dir=os.path.join(datasets_path, pero_ocr_corpus_dir, periodicals_text),
                                   sample_by="document", name="pero_ocr_periodicals", version=PERIODICALS_VERSION,
                                   description=PERIODICALS_DESCRIPTION)

# save datasets
books_dataset.save_to_disk(os.path.join(datasets_path, pero_ocr_books_dir))
periodicals_dataset.save_to_disk(os.path.join(datasets_path, pero_ocr_periodicals_dir))
