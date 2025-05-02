# based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
import sys
import time
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import wandb
import optuna
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from gpu_free import wait_until_gpu_memory_free
import transformers
import scheduler
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import shutil

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from functools import partial

from model import custom_config, custom_model, check_tensors

check_min_version("4.50.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def group_texts_train(examples, block_size, stride):
    """
    Main data processing function that will concatenate all texts
    from our dataset and generate chunks of block_size.
    you can use a stride lower then block_size if the dataset is small
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = total_length - block_size + 1
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, stride)] for k, t in concatenated_examples.items()
    }
    assert "labels" not in result
    result["labels"] = result["input_ids"].copy()
    for i in range(len(result["attention_mask"])):
        if result["attention_mask"][i] is not None and min(result["attention_mask"][i]) == 1:
            assert result["attention_mask"][i][0] == 1, "Expected the first element of attention_mask to be 1"
            result["attention_mask"][i] = None  # speed up computation
    return result


def group_texts_val(examples, block_size, stride):
    """
    group texts in blocks with a given stride with labels -100 (ignored) when stride<block_size is used
    this is used only for validation
    """
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {k: [] for k in examples.keys()}
    assert "labels" not in result
    result["labels"] = []
    for i in range(0, total_length - block_size + 1, stride):
        for k in examples.keys():
            result[k].append(concatenated_examples[k][i : i + block_size])
        result["labels"].append(concatenated_examples["input_ids"][i : i + block_size].copy())
        if i > 0:
            for j in range(block_size - stride):
                result["labels"][-1][j] = -100
        if result["attention_mask"][-1] is not None and min(result["attention_mask"][-1]) == 1:
            assert result["attention_mask"][-1][0] == 1, "Expected the first element of attention_mask to be 1"
            result["attention_mask"][-1] = None  # speed up computation
    return result


def tokenize_function(examples, tokenizer, text_column_name, prepend_eos):
    tokens = tokenizer(examples[text_column_name])  # ,max_length=config.max_position_embeddings)
    if prepend_eos:
        for i in range(len(tokens["input_ids"])):
            tokens["input_ids"][i] = [tokenizer.eos_token_id] + tokens["input_ids"][i]
    return tokens


def parse_args(args_list):
    parser = argparse.ArgumentParser(description="Train/finetune a model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Skylion007/openwebtext",  # "Salesforce/wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="",  # "wikitext-103-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument("--config_overrides", type=str, default=None, help="override config")

    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument("--precision", type=str, default="fp16", help="precision")
    parser.add_argument("--use_tpu", action="store_true", help="use tpu")
    parser.add_argument(
        "--validation_split_ratio",
        default=0.0005,
        help="The ratio of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument("--name_exp", type=str, help="name experiment", required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "The stride to use when chunking the dataset. This is useful for models that have a maximum input length."
            " A smaller stride will result in more chunks and potentially better performance, but will also increase"
            " the training time."
        ),
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=6e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=0.000002,
        help="final learning rate",
    )
    parser.add_argument(
        "--ratio_min_learning_rate",
        type=float,
        default=0.01,
        help="ratio steps at final learning rate if the mod scheduler is used",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")

    parser.add_argument("--num_train_epochs", type=int, default=12, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=5000,
        help="Total number of training steps to log",
    )
    parser.add_argument(
        "--steps2think",
        type=int,
        default=1,
        help="Number of steps to think before generating the answer for math dataset",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=12,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--grad_clip_value",
        type=float,
        default=None,
        help="Value to clip the gradient norm to.",
    )
    parser.add_argument(
        "--eval_mode_epoch", type=int, default=None, help="after such epoch the model is trained in eval mode"
    )
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Whether to save checkpoints during training.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine_with_min_lr",
        help="The scheduler type to use.",
        choices=[
            "ReduceLROnPlateau_mod",
            "cosine_with_min_lr",
            "cosine_with_min_lr_mod",
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for AdamW optimizer (default: 0.9)")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="experiments", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--max_math_value",
        type=int,
        default=5000,
        help="maximum value for the math operations dataset in the training set",
    )

    parser.add_argument("--no_wait", action="store_true", help="Do not wait for GPU")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--no_compile", action="store_false", dest="compile", help="don't compile the model")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--no_tracking",
        action="store_false",
        dest="with_tracking",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args(args_list)

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main(args_list=None, trial=None):

    # Get the full command line used to launch the script
    command_list = sys.argv.copy()
    for i in range(len(command_list)):
        if command_list[i].startswith("--config_overrides"):
            command_list[i] = "'" + command_list[i] + "'"
    command_line = " ".join(command_list)
    with open("exp.txt", "a") as f:
        f.write("python " + command_line + "\n")

    args = parse_args(args_list)
    if args.dataset_name == "Skylion007/openwebtext":
        prepend_eos = True
        args.dataset_config_name = ""
    else:
        prepend_eos = False
    if not args.no_wait:
        wait_until_gpu_memory_free(min_free_memory=args.per_device_train_batch_size * 1300 + 1500, check_interval=100)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    config_name = args.config_name + " " if args.config_name is not None else ""
    config_name = config_name.replace("custom:", "")
    config_overrides_str = args.config_overrides + " " if args.config_overrides is not None else ""
    config_overrides_str = config_overrides_str.replace('"selfatt_class"', "att")
    config_overrides_str = config_overrides_str.replace('"selfatt_class_kwargs"', "att_args")
    config_overrides_str = config_overrides_str.replace('"', "").replace("'", "")
    run_name = f"{args.name_exp} b{args.per_device_train_batch_size*args.gradient_accumulation_steps} lr{args.learning_rate} {config_name}{config_overrides_str}"

    if args.weight_decay > 0:
        run_name += f" wd{args.weight_decay}"

    used_dataset_name = args.dataset_config_name if args.dataset_config_name else args.dataset_name
    if used_dataset_name == "Skylion007/openwebtext":
        used_dataset_name = "openwebtext"
    if args.dataset_name.startswith("custom:"):
        used_dataset_name = args.dataset_name.split(":")[1]
        custom_dataset = True
        if args.dataset_name == "custom:math_dataset":
            run_name += f" max_val{args.max_math_value}"
            run_name += f" steps2think{args.steps2think}"
    else:
        custom_dataset = False
    args.output_dir = os.path.join(args.output_dir, used_dataset_name, run_name)
    if os.path.exists(args.output_dir):
        args.output_dir += f" {time.strftime('%d %m %Y %H %M %S')}"

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    if args.precision and not args.use_tpu:
        accelerator_log_kwargs["mixed_precision"] = args.precision
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.dataset_name == "custom:math_dataset":
            # Custom dataset for math operations
            from math_dataset import MathOperationsDataset

            lm_datasets = {}
            lm_datasets["train"] = MathOperationsDataset(
                num_samples=2000000,
                operations=["+", "-", "*", "/"],
                min_number=0,
                max_number=args.max_math_value,
                steps2think=args.steps2think,
            )
            lm_datasets["validation"] = MathOperationsDataset(
                num_samples=10000,
                operations=["+", "-", "*", "/"],
                min_number=0,
                max_train_number=args.max_math_value,
                max_number=args.max_math_value * 2,
                tokenizer=lm_datasets["train"].tokenizer,
                steps2think=args.steps2think,
            )
            tokenizer = lm_datasets["train"].tokenizer
        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets = raw_datasets["train"].train_test_split(
                    test_size=args.validation_split_ratio, seed=2357, shuffle=True
                )
                raw_datasets["validation"] = raw_datasets.pop("test")
                # raw_datasets["validation"] = load_dataset(
                #     args.dataset_name,
                #     args.dataset_config_name,
                #     split=f"train[:{args.validation_split_percentage}%]",
                #     trust_remote_code=args.trust_remote_code,
                # )
                # raw_datasets["train"] = load_dataset(
                #     args.dataset_name,
                #     args.dataset_config_name,
                #     split=f"train[{args.validation_split_percentage}%:]",
                #     trust_remote_code=args.trust_remote_code,
                # )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=args.validation_split_ratio, seed=2357, shuffle=True
            )
            raw_datasets["validation"] = raw_datasets.pop("test")
            # raw_datasets["validation"] = load_dataset(
            #     extension,
            #     data_files=data_files,
            #     split=f"train[:{args.validation_split_percentage}%]",
            #     **dataset_args,
            # )
            # raw_datasets["train"] = load_dataset(
            #     extension,
            #     data_files=data_files,
            #     split=f"train[{args.validation_split_percentage}%:]",
            #     **dataset_args,
            # )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        if args.config_name.startswith("custom:"):
            config = custom_config(args.config_name)
        else:
            config = AutoConfig.from_pretrained(
                args.config_name,
                trust_remote_code=args.trust_remote_code,
            )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if args.config_overrides is not None:
        # logger.info(f"Old config: {config}")
        logger.info(f"Overriding config: {args.config_overrides}")
        config.update_from_string(args.config_overrides)
        # if hasattr(config, "selfatt_class_kwargs"):
        #     if isinstance(config.selfatt_class_kwargs,str):
        #         config.selfatt_class_kwargs = json.loads(config.selfatt_class_kwargs)
    print(f"New config: {config}")
    if not custom_dataset:
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code
            )
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        if args.config_name.startswith("custom:"):
            model = custom_model(config)
        else:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing model token embeddings to match tokenizer size")
        model.resize_token_embeddings(len(tokenizer))

    if not custom_dataset:
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

    with accelerator.main_process_first():
        if not custom_dataset:
            tokenized_datasets = raw_datasets.map(
                partial(
                    tokenize_function, tokenizer=tokenizer, text_column_name=text_column_name, prepend_eos=prepend_eos
                ),
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    if args.stride is None:
        args.stride = block_size
        print(f"Using stride {args.stride} with block size {block_size}")
    with accelerator.main_process_first():
        if not custom_dataset:
            lm_datasets = {}
            lm_datasets["train"] = tokenized_datasets["train"].map(
                partial(group_texts_train, block_size=block_size, stride=args.stride),
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
            lm_datasets["validation"] = tokenized_datasets["validation"].map(
                partial(group_texts_train, block_size=block_size, stride=block_size),
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
            if args.stride != block_size:
                logger.info(f"Using stride {args.stride} instead of {block_size}")
                group_texts_val_func = partial(group_texts_val, block_size=block_size, stride=args.stride)
                lm_datasets["validation_stride"] = tokenized_datasets["validation"].map(
                    group_texts_val_func,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    if args.stride != block_size:
        eval_dataset_stride = lm_datasets["validation_stride"]
    else:
        eval_dataset_stride = None

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=2,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        num_workers=2,
        batch_size=args.per_device_eval_batch_size,
    )
    if eval_dataset_stride is not None:
        eval_dataloader_stride = DataLoader(
            eval_dataset_stride,
            collate_fn=default_data_collator,
            num_workers=2,
            batch_size=args.per_device_eval_batch_size,
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    # assuming that if the dim=1 is layer norm or bias so no decay
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.dim() > 1],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.dim() == 1],
            "weight_decay": 0.0,
        },
    ]
    # for n, p in model.named_parameters():
    #     print(n,p.shape,p.dim())

    # print("no decay")
    # for n, p in model.named_parameters():
    #     if p.dim()==1:
    #         print(n)

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, 0.95))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.min_learning_rate is None:
        args.min_learning_rate = args.learning_rate / 20.0
    num_training_steps = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    if "_mod" in args.lr_scheduler_type:
        lr_scheduler = getattr(scheduler, args.lr_scheduler_type)(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=int(num_training_steps * (1.0 - args.ratio_min_learning_rate)),
            min_lr=args.min_learning_rate,
            log_every=300 * args.gradient_accumulation_steps,
        )
    else:
        if "min_lr" in args.lr_scheduler_type:
            scheduler_specific_kwargs = {"min_lr": args.min_learning_rate}
        else:
            scheduler_specific_kwargs = {}
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
        )
    if args.compile:
        print("compiling model")
        model = torch.compile(model)
        print("compiling done")
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    eval_dataloader_stride = (
        accelerator.prepare(eval_dataloader_stride) if eval_dataset_stride is not None else None
    )  # Only prepare if eval_dataset_stride is not None
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.XLA:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    print(accelerator.device.type)
    if accelerator.device.type == "cpu":
        print("no gpu, stopping...")
        return None

    num_m_param = sum(p.numel() for p in model.parameters()) / 1e6
    args.num_m_param = num_m_param
    run_name += f" param{num_m_param:.1f}M"

    args.total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        init_kwargs = {"wandb": {"name": run_name}}  # , "group": "experiment_group"}}
        accelerator.init_trackers(used_dataset_name, experiment_config, init_kwargs=init_kwargs)

    # Train!
    print("***** Running training *****")
    print(f"  Using {accelerator.num_processes} GPUs")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  Number of parameters: {num_m_param:.1f}M")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 1
    perplexity = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        if path.endswith("/"):
            path = path[:-1]
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = (starting_epoch - 1) * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader) + 1
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= resume_step // len(train_dataloader) * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    if args.with_tracking:
        total_loss_mini = 0
        num_loss_mini = 0

    weight_decay_set = False
    for epoch in range(starting_epoch, args.num_train_epochs + 1):
        if args.eval_mode_epoch is not None and epoch >= args.eval_mode_epoch:
            model.eval()
            print("EVAL MODE!")
        else:
            model.train()
        if args.with_tracking:
            total_loss = 0
            num_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # print(batch["input_ids"][0])
            # print(tokenizer.decode(batch["input_ids"][0]))
            # print(batch["labels"][0])

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if torch.isnan(loss).any():
                    print("Loss contains NaN values!")
                    check_tensors(outputs.logits, "logits")
                    accelerator.free_memory()  # Free memory before exiting
                    exit()
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    total_loss_mini += loss.detach().float()
                    num_loss += 1
                    num_loss_mini += 1
                    if (step + 1) % (args.log_every_steps // 8 * args.gradient_accumulation_steps) == 0:
                        print(
                            f"step {completed_steps+1} loss {total_loss.detach().item() / num_loss:.2f} {loss.detach().item():.2f}"
                        )
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.grad_clip_value is not None:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_value)
                #     for name, param in model.named_parameters():
                #         if param.grad is not None and param.grad.norm().item() > 1:
                #             print(completed_steps,name, param.grad.norm().item())
                optimizer.step()
                if isinstance(lr_scheduler.scheduler, scheduler.ReduceLROnPlateau_mod):
                    lr_scheduler.scheduler.update(loss.detach().float().item(), step)
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if (
                    completed_steps % 100 == 0
                    and step != len(train_dataloader) - 1
                    and ((step + 1) % (args.log_every_steps * args.gradient_accumulation_steps) != 0)
                    and args.with_tracking
                ):
                    accelerator.log(
                        {
                            "train_loss_100steps": total_loss_mini.detach().item() / num_loss_mini,
                            "lr0": lr_scheduler.get_lr()[0],
                        },
                        step=completed_steps,
                        log_kwargs={"wandb": {"commit": True}},
                    )
                    total_loss_mini = 0
                    num_loss_mini = 0

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if ((step + 1) % (args.log_every_steps * args.gradient_accumulation_steps) == 0) or (
                step == len(train_dataloader) - 1
            ):
                eval_loss, perplexity = compute_val(
                    model, eval_dataloader, accelerator, args.per_device_eval_batch_size
                )
                if eval_dataloader_stride is not None:
                    eval_loss_stride, perplexity_stride = compute_val(
                        model, eval_dataloader_stride, accelerator, args.per_device_eval_batch_size
                    )
                else:
                    eval_loss_stride = None
                    perplexity_stride = None
                if args.dataset_name == "custom:math_dataset":
                    acc_val, acc_dict_val = compute_acc_math(model, eval_dataloader, tokenizer)
                    # FIXME! using again the train_dataloader! is this ok??
                    acc_train, acc_dict_train = compute_acc_math(model, train_dataloader, tokenizer)
                    print(f"Accuracy train: {acc_train*100:.1f}%")
                    for op in acc_dict_train:
                        print(f"Accuracy train {op}: {acc_dict_train[op]*100:.1f}%")

                    print(f"Accuracy val: {acc_val*100:.1f}%")
                    for op in acc_dict_val:
                        print(f"Accuracy val {op}: {acc_dict_val[op]*100:.1f}%")

                print(command_line)
                print(run_name)
                if args.dataset_name == "custom:math_dataset":
                    equal = "=" * args.steps2think
                    generate(model, tokenizer, "121+4" + equal, 0.1)
                    generate(model, tokenizer, "12-2" + equal, 0.1)
                    generate(model, tokenizer, "7*2" + equal, 0.1)
                    generate(model, tokenizer, "5021+5" + equal, 0.1)
                    generate(model, tokenizer, "5010-7" + equal, 0.1)
                else:
                    generate(model, tokenizer)
                    generate(model, tokenizer, "There are")
                    generate(model, tokenizer, "was born")
                logger.info(
                    f"epoch {epoch}:{(step+1)//args.gradient_accumulation_steps}: perplexity: {perplexity} eval_loss: {eval_loss}"
                    f"\nperplexity_stride: {perplexity_stride} eval_loss_stride: {eval_loss_stride}"
                )

                if args.with_tracking:
                    log = {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "perplexity_stride": perplexity_stride,
                        "eval_loss_stride": eval_loss_stride,
                        "train_loss": total_loss.detach().item() / num_loss,
                        "epoch": epoch,
                        "lr0": lr_scheduler.get_lr()[0],
                    }
                    if args.dataset_name == "custom:math_dataset":
                        log["acc_val"] = acc_val
                        log["acc_train"] = acc_train
                        dict_str_ops = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

                        for op in acc_dict_val:
                            log[f"acc_val_{dict_str_ops[op]}"] = acc_dict_val[op]
                            log[f"acc_train_{dict_str_ops[op]}"] = acc_dict_train[op]
                    accelerator.log(
                        log,
                        step=completed_steps,
                        log_kwargs={"wandb": {"commit": True}},
                    )
                    if trial is not None:
                        trial.report(perplexity, completed_steps // args.log_every_steps)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                        if completed_steps > 20000:
                            return perplexity
                    total_loss = 0
                    num_loss = 0
            if completed_steps >= args.max_train_steps:
                break

        if args.push_to_hub and epoch < args.num_train_epochs:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch" and args.save_checkpoint:
            if epoch > 1:
                output_dir_old = f"epoch_{epoch-1}"
                if args.output_dir is not None:
                    output_dir_old = os.path.join(args.output_dir, output_dir_old)
                if os.path.exists(output_dir_old):
                    shutil.rmtree(output_dir_old)
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    return perplexity


def generate(model, tokenizer, prompt="Once upon a time in a faraway land,", temperature=0.7, stop_at_eos=True):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to("cuda")
    # Generate text
    output = model.generate(input_ids, max_length=100, temperature=temperature)
    if stop_at_eos:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            try:
                output = output[:, : output[0].tolist().index(eos_token_id) + 1]
            except ValueError:
                pass
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Print the generated text
    print(generated_text)
    model.train()


def compute_val(model, eval_dataloader, accelerator, per_device_eval_batch_size, generalized=True):
    model.eval()
    losses = []
    if generalized:
        losses_generalized = []
        valid_values = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        loss = accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)).double()
        losses.append(loss)
        if generalized:
            # we remove the last element (batch size in total) because labels are not shifted
            num_valid_labels = (batch["labels"] != -100).sum() - batch["labels"].size(0)
            # not sure this is correct with multiple GPUs
            num_valid_labels = accelerator.gather_for_metrics(num_valid_labels)
            valid_values += num_valid_labels.sum()
            losses_generalized.append(loss.mean().view(-1) * num_valid_labels)

    losses = torch.cat(losses)
    if generalized:
        losses_generalized = torch.cat(losses_generalized)
        eval_loss_generalized = losses_generalized.sum() / valid_values
        try:
            perplexity_generalized = math.exp(eval_loss_generalized)
        except OverflowError:
            perplexity_generalized = float("inf")

    eval_loss = torch.mean(losses)
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    model.train()
    print(f"perplexity_generalized - perplexity = {perplexity_generalized-perplexity:.4f}")
    if generalized:
        return eval_loss_generalized, perplexity_generalized
    else:
        return eval_loss, perplexity


def compute_acc_math(model, eval_dataloader, tokenizer, max_data=10000):
    debug = False
    model.eval()
    sum_acc = 0
    acc_num = 0
    acc_dict = {"+": 0, "-": 0, "*": 0, "/": 0}
    count_dict = {"+": 0, "-": 0, "*": 0, "/": 0}
    operations_dict = {op: tokenizer.convert_tokens_to_ids(op) for op in ["+", "-", "*", "/"]}

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if acc_num >= max_data:
                break
            outputs = model(**batch)
            logits = outputs.logits
            predicted_tokens = torch.argmax(logits, dim=-1)[:, :-1]
            gt = batch["labels"][:, 1:]
            input_ids = batch["input_ids"]
            assert predicted_tokens.shape == gt.shape
            predicted_tokens[gt == -100] = -100
            acc = (predicted_tokens == gt).float().min(dim=-1).values
            for op in operations_dict.keys():
                res = torch.where(input_ids == operations_dict[op])
                if len(res[0]) > 0:
                    op_indices = res[0]
                    for i in range(len(op_indices)):
                        relevant = (
                            res[1][i]
                            < torch.where(input_ids[op_indices[i]] == tokenizer.convert_tokens_to_ids("="))[0].min()
                        )
                        if relevant:
                            count_dict[op] += 1
                            acc_dict[op] += acc[op_indices[i]].item()
                    # relevant = res[1] < (torch.where(input_ids[res[0]] == tokenizer.convert_tokens_to_ids("="))[1])
                    # op_indices = op_indices[relevant]
                    # count_dict[op] += len(op_indices)
                    # acc_dict[op] += acc[op_indices].sum().item()
            sum_acc += acc.sum().item()
            acc_num += acc.shape[0]
            assert acc_num == sum(count_dict.values())
            if debug:
                predicted_tokens[gt == -100] = tokenizer.pad_token_id
                gt[gt == -100] = tokenizer.pad_token_id
                for b in range(len(batch["input_ids"])):
                    if tokenizer.decode(batch["input_ids"][b], skip_special_tokens=True) in ["12-2=10", "7*2=14"]:
                        print("input", tokenizer.decode(batch["input_ids"][b], skip_special_tokens=True))
                        print("predicted", tokenizer.decode(predicted_tokens[b], skip_special_tokens=True))
                        print("gt", tokenizer.decode(gt[b], skip_special_tokens=True))
                        print("acc", acc[b])
                        print((predicted_tokens == gt), (predicted_tokens == gt).shape)
                        break
                    # print("full",tokenizer.decode(input_full[0]))
                    # print("inp",tokenizer.decode(input_ids[0]))
                    # print("gt",tokenizer.decode(gt_res[0]))
                    # print("out",tokenizer.decode(output[0]))
                    # print("out compared",tokenizer.decode(output[0][idx + 1: idx_end+1]))
                    # breakpoint()
            # print("---------------------")
            # print(tokenizer.decode(batch["input_ids"][1],skip_special_tokens=True))
            # print(tokenizer.decode(predicted_tokens[1],skip_special_tokens=True))
            # print(tokenizer.decode(gt[1],skip_special_tokens=True))
            # print((predicted_tokens == gt).float().min(dim=-1).values[1])
            # print("---------------------")
    avg_acc = sum_acc / acc_num
    assert acc_num == sum(count_dict.values())
    assert sum_acc == sum(acc_dict.values())
    for op in operations_dict.keys():
        if count_dict[op] > 0:
            acc_dict[op] = acc_dict[op] / count_dict[op]
        else:
            acc_dict[op] = -1
    model.train()
    return avg_acc, acc_dict


if __name__ == "__main__":
    main()
