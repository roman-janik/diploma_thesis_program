# Author: Roman Janík
# Script for creating a Hugging Face fast tokenizer.
#

import argparse

from transformers import AutoTokenizer, RobertaTokenizerFast, PreTrainedTokenizerFast
from datasets import load_from_disk, concatenate_datasets
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    AddedToken
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer_dir", required=True,
                        help="Path to directory with tokenizer and model config.")
    parser.add_argument("-n", "--new_tokenizer_dir", required=True,
                        help="Path to directory to save new tokenizer.")
    parser.add_argument("-d", "--dataset_dirs", required=True, help="List of paths to directories containing datasets.")
    parser.add_argument("-v", "--vocab_size", type=int, required=True, help="Size of vocabulary.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dataset_dirs = args.dataset_dirs.split()

    raw_datasets = [load_from_disk(d_path) for d_path in dataset_dirs]

    # concatenate datasets
    concat_dataset_train = concatenate_datasets(
        [raw_dataset["train"] for raw_dataset in raw_datasets]
    )

    def get_training_corpus():
        dataset = concat_dataset_train
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx: start_idx + 1000]
            yield samples["text"]

    # init tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special_tokens = [AddedToken("[CLS]", normalized=True),
                      AddedToken("[PAD]", normalized=True),
                      AddedToken("[SEP]", normalized=True),
                      AddedToken("[UNK]", normalized=True),
                      AddedToken("[MASK]", normalized=True, lstrip=True)]

    # train ner tokenizer
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # set tokenizer post processor and decoder
    tokenizer.post_processor = processors.RobertaProcessing(add_prefix_space=False, sep=("[SEP]", 2), cls=("[CLS]", 0))
    tokenizer.decoder = decoders.ByteLevel()

    # test tokenizer
    enc = tokenizer.encode("Skákal pes, přes oves, přes zelenou louku!")
    print(enc.tokens)

    wrapped_tokenizer = RobertaTokenizerFast(tokenizer_object=tokenizer,
                                             bos_token="[CLS]",
                                             eos_token="[SEP]",
                                             sep_token="[SEP]",
                                             cls_token="[CLS]",
                                             unk_token="[UNK]",
                                             pad_token="[PAD]",
                                             mask_token="[MASK]")

    # save tokenizer
    wrapped_tokenizer.save_pretrained(args.new_tokenizer_dir)


if __name__ == "__main__":
    main()
