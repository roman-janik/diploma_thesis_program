# Author: Roman Janík
# Script for pretraining masked language model from scratch on Pero OCR Books and Pero OCR Periodicals datasets.
#
# Partially taken over from Hugging Face Course Chapter 7 Training a causal language model from scratch:
# https://huggingface.co/course/en/chapter7/6?fw=pt
#

import argparse
import os

import datasets
import transformers
import torch
import logging
import datetime
import time

from accelerate import Accelerator
from tqdm.auto import tqdm
from yaml import safe_load, safe_dump
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import find_executable_batch_size


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    parser.add_argument('-t', '--timeout', type=float, required=True, help='Training timeout in hours.')
    parser.add_argument('-s', '--from_state', default=False, action="store_true",
                        help='Load training state from checkpoint.')
    args = parser.parse_args()
    return args


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def log_summary(exp_name: str, config: dict):
    log_msg("{:<24}{}\n{:<24}{}".format(
        "Name:", exp_name.removeprefix("exp_configs_mlm/").removesuffix(".yaml"), "Description:", config["desc"]))
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = os.path.basename(config["tokenizer"]) + "_dts"
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n".format(
        "Start time:", ct, "Model:", config["models"]["trained_model"]["name"],
        "Datasets:", config["datasets"][dataset_name]["name"]))

    cf_t = config["training"]
    log_msg("Parameters:\n{:<24}{}\n{:<24}{}".format(
        "Num train epochs:", cf_t["num_train_epochs"], "Batch size:", cf_t["batch_size"]))
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Learning rate:", cf_t["optimizer"]["learning_rate"], "Weight decay:", cf_t["optimizer"]["weight_decay"],
        "Lr scheduler:",
        cf_t["lr_scheduler"]["name"], "Warmup ratio:", cf_t["lr_scheduler"]["warmup_ratio"]))


def save_epoch_step(epoch_step_path: str, epoch: int, step: int):
    with open(epoch_step_path, "w", encoding="utf-8") as f:
        safe_dump({"epoch": epoch, "step": step}, f)


def load_epoch_step(epoch_step_path: str):
    with open(epoch_step_path, encoding="utf-8") as f:
        epoch_step_dict = safe_load(f)
    return epoch_step_dict["epoch"], epoch_step_dict["step"]


# noinspection PyArgumentList
def main():
    start_time = time.monotonic()
    output_dir = "../results"
    model_dir = "../results/model"
    train_state_dir = "../results/train_state"
    epoch_step_file = "epoch_step.yaml"
    log_dir = "../results/logs"
    start_from_last_state = False
    args = parse_arguments()

    # Set timeout limit
    ten_min = 600.
    sec_in_hour = 3600.
    time_limit = 1800.  # 10 min before timeout
    # time_limit = (args.timeout / sec_in_hour) - ten_min  # 10 min before timeout

    # Load config file
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Start logging, print experiment configuration
    logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"), level=logging.INFO,
                        encoding='utf-8', format='%(message)s')
    log_msg("Experiment summary:\n")
    log_summary(args.config, config)
    log_msg("-" * 80 + "\n")

    # Init tensorboard writer
    writer = SummaryWriter(log_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"], model_max_length=512)
    dataset_name = os.path.basename(config["tokenizer"]) + "_dts"
    dataset_path = config["datasets"][dataset_name]["path"]
    pero_ocr_dataset = datasets.load_from_disk(dataset_path)
    print(pero_ocr_dataset)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=config["training"]["batch_size"])
    def inner_training_loop(batch_size, from_l_state=start_from_last_state):
        log_msg(f"Current batch size: {batch_size}")
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        train_dataloader = torch.utils.data.DataLoader(
            pero_ocr_dataset["train"],
            collate_fn=data_collator,
            batch_size=batch_size,
            # batch_size=config["training"]["batch_size"],
        )

        eval_dataloader = torch.utils.data.DataLoader(
            pero_ocr_dataset["test"],
            collate_fn=data_collator,
            batch_size=batch_size,
            # batch_size=config["training"]["batch_size"],
        )

        roberta_config = transformers.RobertaConfig.from_pretrained(config["models"]["trained_model"]["config"])

        model = transformers.RobertaForMaskedLM(config=roberta_config)

        # print model size
        model_size = sum(t.numel() for t in model.parameters())
        log_msg(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")

        cf_optimizer = config["training"]["optimizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cf_optimizer["learning_rate"],
                                      betas=(cf_optimizer["beta1"], cf_optimizer["beta2"]),
                                      weight_decay=cf_optimizer["weight_decay"])

        num_train_epochs = config["training"]["num_train_epochs"]
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch
        start_epoch = 0
        start_step = 1

        lr_scheduler = transformers.get_scheduler(
            config["training"]["lr_scheduler"]["name"],
            optimizer=optimizer,
            num_warmup_steps=int(config["training"]["lr_scheduler"]["warmup_ratio"] * num_training_steps),
            num_training_steps=num_training_steps,
        )

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        accelerator.register_for_checkpointing(lr_scheduler)

        # load training state from checkpoint
        if args.from_state:
            accelerator.load_state(train_state_dir)
            last_state_epoch, last_state_step = load_epoch_step(os.path.join(train_state_dir, epoch_step_file))
            train_dataloader_last = accelerator.skip_first_batches(train_dataloader, last_state_step)
            start_epoch, start_step = last_state_epoch, last_state_step + 1
            from_l_state = True
            log_msg(f"---------- Start training from last state. ----------")
            log_msg(f"Start epoch:   {start_epoch}, start step:   {start_step}\n")

        def save_model_and_state():
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(model_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(model_dir)
                save_epoch_step(os.path.join(train_state_dir, epoch_step_file), epoch, step)
            accelerator.save_state(train_state_dir)
            accelerator.print(f"Model and train state were successfully saved, last step:   {step}")

        progress_bar = tqdm(range(num_training_steps + 1), initial=start_step)
        completed_steps = 0
        gradient_accumulation_steps = 8_192 / batch_size
        eval_steps = 200  # 2_000
        eval_loss, perplexity = torch.Tensor(1), torch.Tensor(1)
        log_msg(f"Current gradient accumulation steps: {gradient_accumulation_steps}")

        # Training loop
        for epoch in range(start_epoch, num_train_epochs):
            if from_l_state:
                from_l_state = False
                curr_train_dataloader = train_dataloader_last
            else:
                curr_train_dataloader = train_dataloader
                start_step = 1

            # Training
            model.train()
            for step, batch in enumerate(curr_train_dataloader, start=start_step):
                # forward an backward pass
                outputs = model(**batch)
                print("Tensor device:   {}".format(batch["input_ids"].device))
                loss = outputs.loss
                if (completed_steps + 1) % 100 == 0:
                    writer.add_scalar("Loss/train", loss.item() * gradient_accumulation_steps, epoch)
                    accelerator.print({
                        "lr": lr_scheduler.get_last_lr()[0],
                        "steps": completed_steps,
                        "Loss/train": loss.item() * gradient_accumulation_steps,
                    })
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    writer.add_scalar("Learning_rate/train", lr_scheduler.get_last_lr()[0], step)
                    accelerator.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                progress_bar.update(1)

                # check for time limit
                curr_time = time.monotonic()
                if curr_time - start_time > time_limit:
                    save_model_and_state()
                    if accelerator.is_main_process:
                        log_msg("\nTraining timed out!")
                    return

                # Evaluation
                if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                    model.eval()
                    eval_losses = []
                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                        eval_losses.append(accelerator.gather(outputs.loss).reshape(1))

                    eval_loss = torch.mean(torch.cat(eval_losses))

                    try:
                        perplexity = torch.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    accelerator.print(f"\nepoch {epoch}:",
                                      {"loss/eval": "{:.4f}".format(eval_loss.item()),
                                       "perplexity": "{:.4f}".format(perplexity.item())})

                    writer.add_scalar("Loss/eval", eval_loss.item(), epoch)
                    writer.add_scalar("Perplexity", perplexity.item(), epoch)
                    model.train()

                    # Save
                    save_model_and_state()

        writer.flush()
        time.sleep(3)

        # Log last validation results
        log_msg("-" * 80 + "\nExperiment results:\n")
        log_msg("Validation set evaluation:")
        last_val_results = {"Loss/eval": "{:.4f}".format(eval_loss.item()),
                            "perplexity": "{:.4f}".format(perplexity.item())}
        log_msg(f"Last epoch {epoch}:\n{last_val_results}\n")

    inner_training_loop()

    end_time = time.monotonic()
    log_msg("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()