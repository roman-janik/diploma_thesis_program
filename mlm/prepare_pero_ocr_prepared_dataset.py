# Author: Roman Janík
# Script for preparing Pero OCR MLM dataset from Pero OCR Books and Pero OCR Periodicals datasets.
# Dataset is tokenized, then concatenated and chunked by model input size. Result is a dataset ready for training.
#

import datasets
import transformers

import time
import datetime


def prepare_datasets(dataset_paths: dict, tokenizer_path, model_max_length):
    raw_datasets = {key: datasets.load_from_disk(value) for (key, value) in dataset_paths.items()}

    # concatenate datasets
    concat_dataset_train = datasets.concatenate_datasets(
        [raw_dataset["train"] for raw_dataset in raw_datasets.values()]
    )
    # concat_dataset_train = concat_dataset_train.select(range(42))
    # print(concat_dataset_train[0])
    # print(concat_dataset_train[1])

    # initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize(examples):
        return tokenizer(examples["text"])

    t_concat_dataset_train = concat_dataset_train.map(
        tokenize,
        batched=True,
        remove_columns=concat_dataset_train.column_names
    )

    def group_texts(examples, block_size=model_max_length):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    g_t_concat_dataset_train = t_concat_dataset_train.map(
        group_texts,
        batched=True
    )

    return g_t_concat_dataset_train


new_tokenizer_path = "../../resources/tokenizers/new_tokenizer"
new_tokenizer_26k_path = "../../resources/tokenizers/new_tokenizer_26k"
new_tokenizer_12k_path = "../../resources/tokenizers/new_tokenizer_12k"
old_tokenizer_path = "../../resources/robeczech-base-pytorch"

datasets_dict = {"books": "../../../datasets/pero_ocr_books", "periodicals": "../../../datasets/pero_ocr_periodicals"}

model_input_max = 512  # same as RoBERTa, RobeCzech

start_time = time.monotonic()
prepared_dataset_new_52k = prepare_datasets(datasets_dict, new_tokenizer_path, model_input_max)
prepared_dataset_new_26k = prepare_datasets(datasets_dict, new_tokenizer_26k_path, model_input_max)
prepared_dataset_new_12k = prepare_datasets(datasets_dict, new_tokenizer_12k_path, model_input_max)
prepared_dataset_old = prepare_datasets(datasets_dict, old_tokenizer_path, model_input_max)
end_time = time.monotonic()
print("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


# create validation split
prepared_dataset_new_52k = prepared_dataset_new_52k.train_test_split(test_size=0.005)
prepared_dataset_new_26k = prepared_dataset_new_26k.train_test_split(test_size=0.005)
prepared_dataset_new_12k = prepared_dataset_new_12k.train_test_split(test_size=0.005)
prepared_dataset_old = prepared_dataset_old.train_test_split(test_size=0.005)

# save dataset
prepared_dataset_new_52k.save_to_disk("../../../datasets/pero_ocr_prepared/new_tokenizer_dts")
prepared_dataset_new_26k.save_to_disk("../../../datasets/pero_ocr_prepared/new_tokenizer_26k_dts")
prepared_dataset_new_12k.save_to_disk("../../../datasets/pero_ocr_prepared/new_tokenizer_12k_dts")
prepared_dataset_old.save_to_disk("../../../datasets/pero_ocr_prepared/old_tokenizer_dts")
