# Author: Roman Janík
# Script for using a NER model in Transformers pipeline.
#

from transformers import pipeline
import transformers

model_path = "../results/2022-10-17-22-36-main-22h/results"
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(model_path, add_prefix_space=True)
token_classifier = pipeline(
    "token-classification", model=model_path, aggregation_strategy="simple"
)

token_classifier("Zde bude nějaký český testovací text.")
