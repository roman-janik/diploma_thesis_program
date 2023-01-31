# Author: Roman Janík
# Script for training a semantic segmentation model for cropping document pages. Model will segment original document
# page scan and added header and footer, in order to use segmentation map to remove header and footer.
#

import argparse

import numpy as np
import torchvision
import torch
import evaluate
import datasets
import transformers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True,
                        help="Path to directory with dataset files. Directory must contain subdirectories images"
                             "and masks with respective files.")
    parser.add_argument("-e", "--num_epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("-b", "--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("-l", "--lr_rate", default=6.e-5, type=float, help="Learning rate.")
    parser.add_argument("-i", "--image_width_height", default=800, type=int,
                        help="Image width and height for training.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_dir = "../../results/model"
    checkpoint = "../../resources/nvidia_mit-b0"

    print("Script for training a semantic segmentation model for cropping document pages. Model will segment "
          "original document page scan and added header and footer, in order to use segmentation map "
          "to remove header and footer. Model is a pretrained nvidia/mit-b0"
          "model from Hugging Face Transformers library. Resulting model will be saved in '../../results' directory.\n")

    # dataset
    dataset = datasets.load_dataset("page_segmentation_dataset.py", data_dir=args.dataset_dir)
    dataset_train = dataset["train"]
    dataset_validation = dataset["validation"]

    # preprocess
    image_processor = transformers.SegformerFeatureExtractor.from_pretrained(checkpoint)

    jitter = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def train_transforms(example_batch):
        images = [jitter(x) for x in example_batch["image"]]
        labels = [x for x in example_batch["mask"]]
        inputs = image_processor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["mask"]]
        inputs = image_processor(images, labels)
        return inputs

    dataset_train.set_transform(train_transforms)
    dataset_validation.set_transform(val_transforms)

    # metric
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = torch.nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=2,
                ignore_index=255,
                reduce_labels=False,
            )
            for key, value in metrics.items():
                if type(value) is np.ndarray:
                    metrics[key] = value.tolist()
            return metrics

    # train
    id2label = {0: "background", 1: "page"}
    label2id = {"background": 0, "page": 1}

    model = transformers.SegformerForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label,
                                                                          label2id=label2id)

    training_args = transformers.TrainingArguments(
        output_dir=model_dir,
        learning_rate=args.lr_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        eval_steps=50,
        logging_steps=5,
        eval_accumulation_steps=5,
        remove_unused_columns=False
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
