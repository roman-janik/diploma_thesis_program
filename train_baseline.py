# Author: Roman Janík
# Script for training baseline model from RobeCzech on CNEC 2.0 CoNNL and CHNEC 1.0 datasets.
#
# Partially taken over from Hugging Face Course Chapter 7 Token classification:
# https://huggingface.co/course/en/chapter7/2?fw=pt
#

import argparse
import os

import datasets
import transformers
import torch
from accelerate import Accelerator
import evaluate
import numpy as np
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path', required=True, help='Path to datasets.')
    parser.add_argument('--model_path', required=True, help='Path to model.')
    parser.add_argument('--batch_size', required=True, help='Training batch size.', type=int)
    parser.add_argument('--val_batch_size', required=True, help='Evaluation batch size.', type=int)
    args = parser.parse_args()
    return args


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def prepare_datasets(datasets_path: str, model_path: str):
    cnec_dir = "cnec2.0_extended"
    chnec_dir = "chnec1.0"
    cnec_dataset = datasets.load_from_disk(os.path.join(datasets_path, cnec_dir))
    chnec_dataset = datasets.load_from_disk(os.path.join(datasets_path, chnec_dir))
    label_names = cnec_dataset["train"].features["ner_tags"].feature.names

    # concatenate datasets
    cnec_chnec_dataset_train = datasets.concatenate_datasets([cnec_dataset["train"], chnec_dataset["train"]])
    cnec_chnec_dataset_validation = datasets.concatenate_datasets(
        [cnec_dataset["validation"], chnec_dataset["validation"]])
    cnec_chnec_dataset_test = datasets.concatenate_datasets([cnec_dataset["test"], chnec_dataset["test"]])

    # initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    t_cnec_chnec_dataset_train = cnec_chnec_dataset_train.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=cnec_chnec_dataset_train.column_names,
    )

    t_cnec_chnec_dataset_validation = cnec_chnec_dataset_validation.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=cnec_chnec_dataset_train.column_names,
    )

    t_cnec_chnec_dataset_test = cnec_chnec_dataset_test.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=cnec_chnec_dataset_train.column_names,
    )

    return tokenizer, label_names, {
                    "train": t_cnec_chnec_dataset_train,
                    "validation": t_cnec_chnec_dataset_validation,
                    "test": t_cnec_chnec_dataset_test
    }


def main():
    args = parse_arguments()

    tokenizer, label_names, tokenized_datasets = prepare_datasets(args.datasets_path, args.model_path)
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=args.val_batch_size
    )

    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        id2label=id2label,
        label2id=label2id,
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = 5
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    def postprocess(predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    progress_bar = tqdm(range(num_training_steps))
    output_dir = "../results"

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
