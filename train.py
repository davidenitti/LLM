# based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
import faulthandler

faulthandler.enable()
from utils.save_load_ckpt import save_checkpoint
from utils.save_load_ckpt import resume_training
import sys
import time
import argparse
import json
import logging
import math
import os
import random
import traceback
import wandb
import datasets
import torch
from utils.config_utils import compact_run_name, get_explicit_cli_args
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.gpu_free import wait_until_gpu_memory_free
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
)
from transformers import PreTrainedTokenizerFast
from model import CustomGPTConfig, CustomGPTmodel
from model_standalone import CustomGPTmodel as StandaloneCustomGPTmodel
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from scheduler import create_lr_scheduler, create_optimizer
from functools import partial

from custom_func import custom_config, custom_model
from utils.checks import check_tensors
from utils.config_utils import convert_string_format_to_json_like

from utils.metrics_utils import TrainMetricsAccumulator

from evaluation import compute_valv2, compute_dataset_accuracy, sample_generations
from utils.cuda_utils import _is_cuda_oom_error, retry_on_cuda_oom, clean_gpu
from preprocess import (
    tokenize_function,
    group_texts_train,
    group_texts_val,
    build_conversations,
    build_conversations_v2,
    crop_data_collator,
    shift_batch,
)

check_min_version("4.50.0.dev0")

logger = get_logger(__name__)

require_version(
    "datasets>=2.14.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Register your config and model with the Transformers library
AutoConfig.register("custom_gptv0", CustomGPTConfig)
AutoModelForCausalLM.register(CustomGPTConfig, CustomGPTmodel)


def parse_args(args_list):
    """Parse CLI arguments and merge optional JSON overrides safely.

    Values from `--args_from_json` are applied only to options not set
    explicitly on the command line, and overlapping settings raise an error
    instead of silently overriding the user's intent.
    """
    parser = argparse.ArgumentParser(description="Train/finetune a model")
    parser.add_argument(
        "--args_from_json",
        type=str,
        default=None,
        help="If provided, load the args from a json file.",
    )
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
    parser.add_argument("--config", type=str, default=None, help="override config")
    parser.add_argument(
        "--config_dataset",
        type=str,
        default="configs/default_arc_agi_dataset.json",
        help="dataset config for `custom:arc_agi` only",
    )

    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--precision", type=str, default="bf16", help="precision")
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
    parser.add_argument("--detach_every", type=int, default=None, help="detach every N steps")
    parser.add_argument("--reset_instead_detach", action="store_true", help="reset instead of detach")
    parser.add_argument(
        "--reset_prob",
        type=float,
        default=-1.0,
        help="multiple of 1/mini_steps to compute the probability to reset for rnn models",
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
        default=32,
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
        default=None,
        help="final learning rate",
    )
    parser.add_argument(
        "--crop_not_valid",
        action="store_true",
        help="Whether to crop the input sequences to the max valid length.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument(
        "--scale_lowrank",
        type=float,
        default=0.3,
        help="Scale factor for weight decay on LowRankLinear params (multiplied by --weight_decay).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=12,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--logval_steps",
        type=int,
        default=2000000,
        help="log validation every N examples",
    )
    parser.add_argument(
        "--logtrain_steps",
        type=int,
        default=20000,
        help="log train every N examples",
    )

    parser.add_argument(
        "--num_runs_accuracy",
        type=int,
        default=100,
        help="Total number of runs to compute the accuracy by majority voting (e.g. for ARC-AGI)",
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
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--grad_clip_value",
        type=float,
        default=1.0,
        help="Value to clip the gradient norm to.",
    )
    parser.add_argument(
        "--save_checkpoint_steps",
        type=int,
        default=50000,
        help="how often to save a checkpoint of the model.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine_with_min_lr",
        help="The scheduler type to use.",
        choices=[
            "cosine_with_min_lr",
            "fast_start_cosine_with_min_lr",
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 for AdamW optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        help="Type of optimizer to use.",
        choices=["adamw", "sgd"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=2000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="experiments",
        help="Where to store the final model.",
    )
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
    parser.add_argument(
        "--use_generate",
        action="store_true",
        help="use generate for accuracy evaluation",
    )
    parser.add_argument("--wait", type=int, default=None, help="Gb free to wait for GPU")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_compile",
        action="store_false",
        dest="compile",
        help="don't compile the model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="if not None it loads the provided checkpoint path. "
        "empty string loads most recent checkpoint if available. "
        "empty string reuses existing output_dir, overwriting the content.",
    )
    parser.add_argument(
        "--reset_resume",
        action="store_true",
        help="Whether or not to reset the optimizer and scheduler when resuming from a checkpoint.",
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
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--rand_shift_prob",
        default=0.0,
        help="do a random shift of the input text",
        type=float,
    )
    parser.add_argument(
        "--max_rand_shift",
        default=1,
        help="maximum number of tokens to shift",
        type=int,
    )
    parser.add_argument(
        "--not_aligned_labels",
        action="store_false",
        dest="aligned_labels",
        help="disable align labels earlier (instead of shifting inside the loss function).",
    )
    args = parser.parse_args(args_list)
    if args.args_from_json is not None:
        print("Loading args from json:", args.args_from_json)
        with open(args.args_from_json, "r") as f:
            args_json = json.load(f)
        explicit_cli_args = get_explicit_cli_args(parser, args_list)
        for k, v in args_json.items():
            if k in ["name_exp", "args_from_json", "output_dir"]:
                raise ValueError(f"Argument '{k}' cannot be set in {args.args_from_json} as it is reserved.")
            if not hasattr(args, k):
                print(f"Warning: arg {k} from json not in args")
            else:
                if k in explicit_cli_args:
                    raise ValueError(
                        f"Argument '{k}' is set in both CLI and {args.args_from_json}; remove one to avoid override."
                    )
                setattr(args, k, v)
    return args


def assert_embedding_tying(model):
    """Assert that input and output embeddings match the model tying configuration."""
    output_embeddings = model.get_output_embeddings()
    assert output_embeddings is not None
    input_weight = model.get_input_embeddings().weight
    output_weight = output_embeddings.weight
    if model.config.tie_word_embeddings:
        assert input_weight is output_weight
        assert input_weight.data_ptr() == output_weight.data_ptr()
    else:
        assert input_weight is not output_weight
        assert input_weight.data_ptr() != output_weight.data_ptr()


def build_run_name_and_dataset_metadata(args):
    """Build the initial run name and dataset metadata from CLI args."""
    config_name = args.config_name + " " if args.config_name is not None else ""
    config_name = config_name.replace("custom:", "")
    if isinstance(args.config, dict):
        config_str = "CUSTOM_CONFIG"  # FIXME
    elif args.config is None:
        config_str = ""
    elif args.config.endswith(".json"):
        config_str = os.path.basename(args.config).replace(".json", " ")
    else:
        config_str = args.config + " "

    run_name = f"{args.name_exp} {config_name}{config_str}"
    if args.args_from_json is not None:
        run_name += f" {os.path.basename(args.args_from_json).replace('.json',' ')}"
    if args.detach_every is not None:
        run_name += f" detach{args.detach_every}"
    if args.weight_decay > 0:
        run_name += f" wd{args.weight_decay}"
    run_name += f" lr{args.learning_rate}"

    used_dataset_name = args.dataset_config_name if args.dataset_config_name else args.dataset_name
    if used_dataset_name == "Skylion007/openwebtext":
        used_dataset_name = "openwebtext"
    elif used_dataset_name == "OpenAssistant/oasst1":
        used_dataset_name = "openassistant"
    if args.dataset_name.startswith("custom:"):
        used_dataset_name = args.dataset_name.split(":")[1]
        custom_dataset = True
        if args.dataset_name == "custom:math_dataset":
            run_name += f" max_val{args.max_math_value}"
            run_name += f" steps2think{args.steps2think}"
    else:
        custom_dataset = False

    return run_name, used_dataset_name, custom_dataset


def configure_process_logging(accelerator: Accelerator):
    """Configure per-process logging and library verbosity."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def initialize_experiment_tracking(args, accelerator, used_dataset_name, run_name):
    """Initialize trackers, persist the resolved config, and reuse a saved wandb run id."""
    wandb_run_id = None
    wandb_run_id_path = None
    if args.with_tracking and ("wandb" in args.report_to or args.report_to == "all"):
        wandb_run_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
        if args.resume is not None and os.path.exists(wandb_run_id_path):
            with open(wandb_run_id_path, "r") as f:
                wandb_run_id = f.read().strip() or None
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
            json.dump(experiment_config, f, indent=2)
        init_kwargs = {"wandb": {"name": run_name}}  # , "group": "experiment_group"}}
        if wandb_run_id is not None:
            init_kwargs["wandb"]["id"] = wandb_run_id
            if args.resume is not None:
                init_kwargs["wandb"]["resume"] = "allow"
        accelerator.init_trackers(used_dataset_name, experiment_config, init_kwargs=init_kwargs)
    if accelerator.is_main_process:
        run = wandb.run
        if run is not None:
            run.log_code(root=".", name=f"code-{run.id}")
            if wandb_run_id_path is not None:
                with open(wandb_run_id_path, "w") as f:
                    f.write(run.id)


def load_training_datasets(args, config=None):
    """Load or build training/validation datasets and return the tokenizer to use."""
    raw_datasets = None
    lm_datasets = None

    if args.dataset_name == "custom:math_dataset":
        # Custom dataset for math operations
        from math_dataset import MathOperationsDataset

        assert args.block_size is not None, "block_size must be set for math dataset"
        lm_datasets = {}
        lm_datasets["train"] = MathOperationsDataset(
            num_samples=None,
            operations=["+", "-", "*", "/"],
            min_number=0,
            max_number=args.max_math_value,
            steps2think=args.steps2think,
            padding=args.block_size,
            aligned_labels=args.aligned_labels,
        )
        lm_datasets["validation"] = MathOperationsDataset(
            num_samples=50000,
            operations=["+", "-", "*", "/"],
            min_number=0,
            max_train_number=args.max_math_value,
            max_number=args.max_math_value * 2,
            tokenizer=lm_datasets["train"].tokenizer,
            steps2think=args.steps2think,
            padding=args.block_size,
            aligned_labels=args.aligned_labels,
        )
        tokenizer = lm_datasets["train"].tokenizer
    elif args.dataset_name == "custom:arc_agi":
        needs_position_idx = bool(
            getattr(config, "use_pos_emb_2d", False) or getattr(config, "use_rot_emb_2d", False)
        )
        needs_generation_mask_input = bool(
            getattr(args, "use_generate", False) or getattr(args, "debug", False)
        )
        if isinstance(args.config_dataset, dict):
            db = args.config_dataset
        elif args.config_dataset.lower().endswith(".json"):
            with open(args.config_dataset, "r", encoding="utf-8") as f:
                db = json.load(f)
        else:
            config_dataset = convert_string_format_to_json_like(args.config_dataset)
            print(config_dataset)
            db = json.loads(config_dataset)

        kwargs_db = {
            "aug": not args.debug,
        }
        kwargs_db.update(db)
        args.config_dataset = kwargs_db.copy()
        print("updated args", kwargs_db)

        include_val_train = False
        if "include_val_train" in kwargs_db:
            include_val_train = kwargs_db["include_val_train"]
            del kwargs_db["include_val_train"]

        include_concept = False
        if "include_concept" in kwargs_db:
            include_concept = kwargs_db["include_concept"]
            del kwargs_db["include_concept"]

        no_train = False
        if "no_train" in kwargs_db:
            no_train = kwargs_db["no_train"]
            del kwargs_db["no_train"]

        re_arc = False
        if "re_arc" in kwargs_db:
            re_arc = kwargs_db["re_arc"]
            del kwargs_db["re_arc"]

        # Custom dataset for ARC-AGI
        from arc_agi_dataset import ArcAGI, get_arc_agi_dataset

        raw_dataset = get_arc_agi_dataset(
            include_val_train, include_concept, no_train=no_train, re_arc=re_arc
        )
        assert args.block_size is not None, "block_size must be set for ARC-AGI dataset"
        lm_datasets = {}
        train_kwargs = kwargs_db.copy()
        train_kwargs["compute_mask_input"] = needs_generation_mask_input
        train_kwargs["include_position_idx"] = needs_position_idx
        lm_datasets["train"] = ArcAGI(
            raw_dataset["train"],
            padding=args.block_size,
            aligned_labels=args.aligned_labels,
            **train_kwargs,
        )
        kwargs_db_val = kwargs_db.copy()
        kwargs_db_val["aug"] = False
        kwargs_db_val["num_synthetic_data"] = 0
        kwargs_db_val["compute_mask_input"] = needs_generation_mask_input
        kwargs_db_val["include_position_idx"] = needs_position_idx
        lm_datasets["validation"] = ArcAGI(
            raw_dataset["val"],
            padding=args.block_size,
            aligned_labels=args.aligned_labels,
            **kwargs_db_val,
        )

        # Dedicated datasets for TTA pass@2 evaluation.
        # These must satisfy compute_acc_tta_pass2 constraints (invertible, stable spans).
        tta_train_kwargs = kwargs_db.copy()
        tta_train_kwargs.update(
            {
                # Ensure eval-style behavior
                "num_synthetic_data": 0,
                # Requirements from compute_acc_tta_pass2
                "aug": not args.debug,
                "text_out": True,
                "encode": True,
                "compute_mask_input": True,
                "include_position_idx": needs_position_idx,
                "shuffle_examples": False,
                "reduce_train": False,
                # Hard constraints to keep mask_input identical across augmentations
                "max_aug_idx": 7,
                "shuffle_prob": 0.0,
                "rand_think": False,
                "repeat_test": 1,
                "return_aug_params": True,
            }
        )

        tta_val_kwargs = kwargs_db_val.copy()
        tta_val_kwargs.update(
            {
                "num_synthetic_data": 0,
                "aug": True,
                "text_out": True,
                "encode": True,
                "compute_mask_input": True,
                "include_position_idx": needs_position_idx,
                "shuffle_examples": False,
                "reduce_train": False,
                "max_aug_idx": 7,
                "shuffle_prob": 0.0,
                "rand_think": False,
                "repeat_test": 1,
                "return_aug_params": True,
            }
        )

        lm_datasets["train_test"] = ArcAGI(
            raw_dataset["train"],
            padding=args.block_size,
            aligned_labels=args.aligned_labels,
            **tta_train_kwargs,
        )
        lm_datasets["validation_test"] = ArcAGI(
            raw_dataset["val"],
            padding=args.block_size,
            aligned_labels=args.aligned_labels,
            **tta_val_kwargs,
        )
        tokenizer = lm_datasets["train"].get_tokenizer()
    else:
        if args.tokenizer_name:
            if args.tokenizer_name.endswith(".json"):
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=args.tokenizer_name,
                    bos_token="[BOS]",
                    eos_token="[EOS]",
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name,
                    use_fast=True,
                    trust_remote_code=True,
                )
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=True,
                trust_remote_code=True,
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            trust_remote_code=True,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=args.validation_split_ratio,
                seed=2357,
                shuffle=True,
            )
            raw_datasets["validation"] = raw_datasets.pop("test")

        # Special preprocessing for OpenAssistant/oasst1: reconstruct multi-turn conversations using parent_id/message_id
        if args.dataset_name == "OpenAssistant/oasst1":
            train_texts = build_conversations(raw_datasets["train"], eos=tokenizer.eos_token)
            raw_datasets["train"] = datasets.Dataset.from_list(train_texts)
            if "validation" in raw_datasets:
                val_texts = build_conversations_v2(raw_datasets["validation"], eos=tokenizer.eos_token)
                raw_datasets["validation"] = datasets.Dataset.from_list(val_texts)

    return raw_datasets, lm_datasets, tokenizer


def resolve_block_size(args, tokenizer, config):
    """Resolve the effective sequence length used for chunking.

    Returns the effective block size derived from `args.block_size` and the
    tokenizer/model limits.

    Side effects:
    - Mutates `args.stride` when it is `None`, setting it to the resolved
      `block_size`.
    """
    context_len = config.context_len

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > context_len:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(4096, context_len)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(4096, context_len)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    if args.stride is None:
        args.stride = block_size
        print(f"Using stride {args.stride} with block size {block_size}")

    return block_size


def get_prepend_bos(args):
    """
    Determine whether to prepend a beginning-of-sequence (BOS)
    token to each example based on the dataset.
    """
    prepend_bos = False
    if args.dataset_name == "Skylion007/openwebtext":
        assert args.dataset_config_name == "", "Skylion007/openwebtext does not use a dataset config name."
        prepend_bos = True
    # Keep other corpora, such as Salesforce/wikitext, without a prepended BOS
    # because adjacent records may continue the same sentence.

    return prepend_bos


def prepare_lm_datasets(
    args,
    block_size,
    raw_datasets,
    tokenizer,
    config,
    accelerator,
):
    """Prepare non-custom language-model datasets for a resolved `block_size`.

    Tokenizes `raw_datasets`, groups training and validation into fixed-size
    chunks, and optionally creates `lm_datasets["validation_stride"]` when
    `args.stride != block_size`.
    """
    prepend_bos = get_prepend_bos(args)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = 0
    if args.dataset_name == "OpenAssistant/oasst1":
        padding = args.block_size if args.block_size is not None else config.context_len

    common_map_kwargs = {
        "batched": True,
        "num_proc": args.preprocessing_num_workers,
        "load_from_cache_file": not args.overwrite_cache,
    }
    group_desc = f"Grouping texts in chunks of {block_size}"

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            partial(
                tokenize_function,
                tokenizer=tokenizer,
                text_column_name=text_column_name,
                prepend_bos=prepend_bos,
                eos=tokenizer.eos_token,  # "<|endoftext|>",
                padding=padding,
            ),
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
            **common_map_kwargs,
        )

        lm_datasets = {}
        lm_datasets["train"] = tokenized_datasets["train"].map(
            partial(
                group_texts_train,
                block_size=block_size,
                stride=args.stride,
                shift_label=args.aligned_labels,
            ),
            desc=group_desc,
            **common_map_kwargs,
        )
        lm_datasets["validation"] = tokenized_datasets["validation"].map(
            partial(
                group_texts_train,
                block_size=block_size,
                stride=block_size,
                shift_label=args.aligned_labels,
            ),
            desc=group_desc,
            **common_map_kwargs,
        )
        if args.stride != block_size:
            logger.info(f"Using stride {args.stride} instead of {block_size}")
            lm_datasets["validation_stride"] = tokenized_datasets["validation"].map(
                partial(
                    group_texts_val,
                    block_size=block_size,
                    stride=args.stride,
                    shift_label=args.aligned_labels,
                ),
                desc=group_desc,
                **common_map_kwargs,
            )

    return lm_datasets


def main(args_list=None):
    """Parse arguments and retry training after CUDA OOM when wait-based recovery is enabled."""
    wait_override = None
    while True:
        args = parse_args(args_list)
        if wait_override is not None:
            args.wait = wait_override
        try:
            return train(args)
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            if not _is_cuda_oom_error(e) or args.wait is None:
                print("Not a CUDA OOM error or resume specified, exiting.")
                print(_is_cuda_oom_error(e), args.wait is None)
                return -1
            print("Retrying after cleaning GPU...")
            if e.__traceback__ is not None:
                traceback.clear_frames(e.__traceback__)
            del e
            clean_gpu()
            wait_override = args.wait + 1
            time.sleep(random.randint(10, 60))


def load_model_config(args):
    """Load the training config and return `(config, is_custom_config_name)`.

    Side effects:
    - If the loaded config has an `aligned_labels` attribute, it is overwritten with
      `args.aligned_labels`.
    - If `args.config` is not `None`, it is consumed as an override source and then
      replaced with the dict returned by the config update method:
      `config.update_from_dict(args.config)` when it is already a dict,
      `config.update_from_dict(json.load(...))` when it is a JSON path,
      or `config.update_from_string(args.config)` otherwise.
    """
    is_custom_config_name = bool(args.config_name and args.config_name.startswith("custom:"))
    if args.config_name:
        if is_custom_config_name:
            config = custom_config(args.config_name)
        else:
            config = AutoConfig.from_pretrained(
                args.config_name,
                trust_remote_code=True,
            )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if hasattr(config, "aligned_labels"):
        config.aligned_labels = args.aligned_labels
    if args.config is not None:
        logger.info(f"Overriding config: {args.config}")
        if isinstance(args.config, dict):
            args.config = config.update_from_dict(args.config)
        elif args.config.lower().endswith(".json"):
            with open(args.config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            args.config = config.update_from_dict(config_dict)
        else:
            args.config = config.update_from_string(args.config)
    assert config.aligned_labels == args.aligned_labels
    assert args.config["aligned_labels"] == args.aligned_labels
    print(f"New config: {config}")
    return config, is_custom_config_name


def load_model(args, config, is_custom_config_name, tokenizer):
    """Instantiate the training model and apply post-load model setup."""
    # Because of https://github.com/huggingface/transformers/issues/42418,
    # custom models load weights through `torch.load` instead of `from_pretrained`.
    if args.model_name_or_path and not is_custom_config_name:
        print("Loading model with from_pretrained:", args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=True,
        )
    elif is_custom_config_name:
        model = custom_model(config)
        if args.model_name_or_path:
            logger.info(f"Loading model weights from {args.model_name_or_path}")
            state_dict = torch.load(args.model_name_or_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                logger.warning(f"missing: {missing}, unexpected: {unexpected}")
        else:
            logger.info("Training new model from scratch")
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    if args.aligned_labels and not isinstance(model, (CustomGPTmodel, StandaloneCustomGPTmodel)):
        raise ValueError("aligned_labels=True is only supported with CustomGPTmodel.")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    try:
        if len(tokenizer) > embedding_size:
            print("Resizing model token embeddings to match tokenizer size")
            model.resize_token_embeddings(len(tokenizer))
    except:
        pass
    assert_embedding_tying(model)
    return model


def get_command_line(log_path=None):
    """Build the current CLI command and optionally append it to a log file."""
    command_list = sys.argv.copy()
    for index, arg in enumerate(command_list):
        if arg == "":
            command_list[index] = f"'{arg}'"

    command_line = "python " + " ".join(command_list)

    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(command_line + "\n")

    return command_line


def run_evaluation(
    *,
    args,
    model,
    eval_dataloader,
    eval_dataloader_stride,
    accelerator: Accelerator,
    completed_steps: int,
    train_dataloader_single,
    eval_dataloader_single,
    tokenizer,
    train_test_dataset,
    validation_test_dataset,
    train_metrics,
    num_update_steps_per_epoch: int,
):
    """
    Run periodic validation, main-process accuracy checks, and tracker logging.

    This helper contains `accelerator.wait_for_everyone()` barriers, so every
    rank must call it consistently.
    """
    logger.info("Running Evaluation")
    generalized = args.dataset_name != "custom:math_dataset"
    train_progress = completed_steps / args.max_train_steps

    eval_loss, perplexity, valid_values, eval_metrics = compute_valv2(
        model,
        eval_dataloader,
        accelerator,
        args.detach_every,
        generalized=generalized,
        precision=args.precision,
        aligned_labels=args.aligned_labels,
        train_progress=train_progress,
    )
    logger.info(f"Validation perplexity: {perplexity}")
    logger.info(f"eval loss : {eval_loss}")

    eval_loss_stride = None
    perplexity_stride = None
    eval_stride_metrics = None
    if eval_dataloader_stride is not None:
        eval_loss_stride, perplexity_stride, _, eval_stride_metrics = compute_valv2(
            model,
            eval_dataloader_stride,
            accelerator,
            args.detach_every,
            generalized=generalized,
            precision=args.precision,
            aligned_labels=args.aligned_labels,
            train_progress=train_progress,
        )
        logger.info(f"Stride Validation perplexity: {perplexity_stride}")
        logger.info(f"Stride eval loss : {eval_loss_stride}")

    # Synchronize before running main-process-only evaluation/logging.
    accelerator.wait_for_everyone()

    acc_val_return = None
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        acc_results = compute_dataset_accuracy(
            args,
            unwrapped_model,
            train_dataloader_single,
            eval_dataloader_single,
            tokenizer,
            completed_steps,
            train_test_dataset=train_test_dataset,
            validation_test_dataset=validation_test_dataset,
            tta_batch_size_val=args.per_device_eval_batch_size,
        )

        sample_generations(
            args.dataset_name,
            unwrapped_model,
            tokenizer,
            args.precision,
            train_progress=train_progress,
            steps2think=args.steps2think,
        )
        logger.info(
            f"epoch {completed_steps / num_update_steps_per_epoch:.2f}: perplexity: {perplexity} eval_loss: {eval_loss}"
            f"\nperplexity_stride: {perplexity_stride} eval_loss_stride: {eval_loss_stride}"
        )
        acc_val_return = acc_results["acc_val"]
        if args.with_tracking:
            log = {
                "perplexity": perplexity,
                "eval_loss": eval_loss,
                "perplexity_stride": perplexity_stride,
                "eval_loss_stride": eval_loss_stride,
                "train_loss": train_metrics.total_loss_avg(),
                "epochs": completed_steps / num_update_steps_per_epoch,
                "total_examples": completed_steps * args.total_batch_size,
            }
            if eval_metrics:
                for k, v in eval_metrics.items():
                    log[f"eval_{k}"] = v
            if eval_dataloader_stride is not None and eval_stride_metrics:
                for k, v in eval_stride_metrics.items():
                    log[f"eval_stride_{k}"] = v
            for k, v in acc_results.items():
                if isinstance(v, dict):
                    raise NotImplementedError()
                if v is not None:
                    log[k] = v
                else:
                    logger.warning(f"{k} returned None value.")
            if valid_values:
                log["eval_loss_sum"] = eval_loss * valid_values / 1000_000
                log["perplexity_sum"] = math.exp(log["eval_loss_sum"])
            accelerator.log(
                log,
                step=completed_steps + 1,
                log_kwargs={"wandb": {"commit": True}},
            )
            logger.info(f"completed_steps: {completed_steps}")
            print(log)

            train_metrics.reset_total_losses()

    accelerator.wait_for_everyone()
    return perplexity, acc_val_return


def build_detached_batch(
    orig_batch,
    mini_step,
    detach_size,
    hidden_state,
    reset_prob,
    reset_instead_detach,
):
    """Slice a sequence batch for truncated BPTT and attach the next hidden state."""
    batch = {}
    start = mini_step * detach_size
    end = (mini_step + 1) * detach_size
    for key, value in orig_batch.items():
        if value is None:
            batch[key] = None
        else:
            batch[key] = value[:, start:end]

    if reset_prob > 0 and random.random() < reset_prob:
        batch["hidden"] = None
    else:
        batch["hidden"] = hidden_state if not reset_instead_detach else None

    return batch


def resolve_detach_schedule(
    sequence_length: int, detach_every: int | None, completed_steps: int
) -> tuple[int, int]:
    """Return `(detach_size, mini_steps)` for truncated BPTT."""
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    if detach_every is None:
        return sequence_length, 1

    max_detach_size = max(1, min(detach_every, completed_steps // 500 + 2, sequence_length))
    for detach_size in range(max_detach_size, 0, -1):
        if sequence_length % detach_size == 0:
            return detach_size, sequence_length // detach_size

    raise AssertionError(f"Failed to find a valid detach size for sequence length {sequence_length}")


def train(args):
    if args.wait is not None:
        wait_until_gpu_memory_free(min_free_memory=args.wait * 1000, check_interval=140)
    command_line = get_command_line(log_path="exp.txt" if args.with_tracking else None)

    accelerator_log_kwargs = {}
    run_name, used_dataset_name, custom_dataset = build_run_name_and_dataset_metadata(args)

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
    if args.precision:
        accelerator_log_kwargs["mixed_precision"] = args.precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    configure_process_logging(accelerator)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    assert args.dataset_name is not None, "Need a dataset name."
    config, is_custom_config_name = load_model_config(args)
    raw_datasets, lm_datasets, tokenizer = load_training_datasets(args, config=config)
    model = load_model(args, config, is_custom_config_name, tokenizer)
    block_size = resolve_block_size(args, tokenizer, config)

    if not custom_dataset:
        # Note that with `batched=True`, dataset.map processes 1,000 texts together by default, so
        # group_texts may drop a remainder within each batch. A higher batch_size can reduce that at
        # the cost of slower preprocessing.
        lm_datasets = prepare_lm_datasets(
            args=args,
            block_size=block_size,
            raw_datasets=raw_datasets,
            tokenizer=tokenizer,
            config=config,
            accelerator=accelerator,
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    train_test_dataset = lm_datasets.get("train_test", None)
    validation_test_dataset = lm_datasets.get("validation_test", None)
    if not custom_dataset and args.stride != block_size:
        eval_dataset_stride = lm_datasets["validation_stride"]
    else:
        assert (
            args.stride == block_size
        ), "Stride-based eval is only supported when not using a custom dataset"
        eval_dataset_stride = None

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.dataset_name == "custom:math_dataset":
        workers = 1
    elif args.dataset_name == "custom:arc_agi":
        workers = 6
    else:
        workers = 2

    loader_common_kwargs = {
        "num_workers": workers,
        "pin_memory": accelerator.device.type == "cuda",
    }
    if workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = 4

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=(default_data_collator if not args.crop_not_valid else crop_data_collator),
        batch_size=args.per_device_train_batch_size,
        **loader_common_kwargs,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=(default_data_collator if not args.crop_not_valid else crop_data_collator),
        batch_size=args.per_device_eval_batch_size,
        **loader_common_kwargs,
    )
    if eval_dataset_stride is not None:
        eval_dataloader_stride = DataLoader(
            eval_dataset_stride,
            collate_fn=default_data_collator,
            batch_size=args.per_device_eval_batch_size,
            **loader_common_kwargs,
        )

    optimizer = create_optimizer(
        model,
        args.weight_decay,
        args.learning_rate,
        args.beta1,
        beta2=0.95,
        optimizer_type=args.optimizer_type,
        scale_lowrank=args.scale_lowrank,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.min_learning_rate is None:
        args.min_learning_rate = args.learning_rate / 100.0
    num_training_steps = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    if args.detach_every is not None:
        num_training_steps *= block_size // (args.detach_every // 2)  # FIXME
    scheduler_specific_kwargs = (
        {"min_lr": args.min_learning_rate} if "min_lr" in args.lr_scheduler_type else {}
    )
    warmup_steps = args.num_warmup_steps * accelerator.num_processes
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=args.min_learning_rate,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )
    if args.compile:
        model = torch.compile(model)
        logger.info("compiling done")
    train_dataloader_single = train_dataloader
    eval_dataloader_single = eval_dataloader
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
    if args.detach_every is not None:
        num_update_steps_per_epoch *= block_size // (args.detach_every // 2)  # FIXME
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    args.total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    assert accelerator.device.type != "cpu"
    run_name += f" b{args.total_batch_size}"
    num_m_param = sum(p.numel() for p in model.parameters()) / 1e6
    args.num_m_param = num_m_param
    run_name += f" p{num_m_param:.1f}M"
    args.output_dir = os.path.join(args.base_output_dir, used_dataset_name, run_name)
    run_name = compact_run_name(run_name)
    args.output_dir = compact_run_name(args.output_dir)
    existing_exp = os.path.exists(args.output_dir)
    if existing_exp and args.resume is None:
        raise ValueError(f"Output dir {args.output_dir} already exists.")
    if not existing_exp and args.resume == "":
        args.resume = None
    os.makedirs(args.output_dir, exist_ok=True)

    initialize_experiment_tracking(args, accelerator, used_dataset_name, run_name)
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

    perplexity = None
    acc_val_return = None
    # Potentially load in the weights and states from a previous save
    ckpt_path, optimizer, lr_scheduler, start_epoch, completed_steps, resume_step = resume_training(
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        num_training_steps=num_training_steps,
        warmup_steps=warmup_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        bar_format=bar_format,
        initial=completed_steps,
        smoothing=0.3,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    train_metrics = None
    if args.with_tracking:
        train_metrics = TrainMetricsAccumulator(accelerator, weight_key="num_valid")
    hidden_state = None
    best_perplexity = float("inf")
    best_acc_val = 0.0
    for epoch in range(start_epoch, args.num_train_epochs + 1):
        clean_gpu()
        model.train()
        if (
            ckpt_path is not None
            and not args.reset_resume
            and epoch == start_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for orig_batch in active_dataloader:
            if (
                args.save_checkpoint_steps > 0
                and completed_steps % args.save_checkpoint_steps == 0
                and completed_steps > 0
            ):
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    completed_steps=completed_steps,
                )
            if random.random() < args.rand_shift_prob:
                shift = random.randint(1, args.max_rand_shift)
                orig_batch = shift_batch(orig_batch, shift, eos=tokenizer.eos_token_id)
            # We split the forward pass into mini_steps to perform truncated BPTT when args.detach_every is set.
            # if detach_every is None, mini_steps=1
            detach_size, mini_steps = resolve_detach_schedule(
                sequence_length=orig_batch["input_ids"].shape[1],
                detach_every=args.detach_every,
                completed_steps=completed_steps,
            )
            hidden_state = None
            for mini_step in range(mini_steps):
                if mini_steps == 1:
                    batch = orig_batch
                else:
                    batch = build_detached_batch(
                        orig_batch=orig_batch,
                        mini_step=mini_step,
                        detach_size=detach_size,
                        hidden_state=hidden_state,
                        reset_prob=args.reset_prob / mini_steps,
                        reset_instead_detach=args.reset_instead_detach,
                    )
                with accelerator.accumulate(model):
                    train_progress = completed_steps / args.max_train_steps
                    outputs = model(train_progress=train_progress, **batch)
                    hidden_state = outputs.hidden_states
                    loss = outputs.loss
                    if torch.isnan(loss).any():
                        logger.error("Loss contains NaN values!")
                        check_tensors(outputs.logits, "logits")
                        accelerator.free_memory()  # Free memory before exiting
                        exit()
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        train_metrics.update(loss, getattr(outputs, "metrics", None))
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.grad_clip_value is not None:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), max_norm=args.grad_clip_value
                        )
                        if grad_norm is not None and args.with_tracking:
                            train_metrics.update_grad(grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    if (
                        completed_steps % (args.logtrain_steps // args.total_batch_size) == 0
                        and args.with_tracking
                    ):
                        log_payload = train_metrics.build_log_payload()
                        log_payload.update(
                            {
                                "total_examples": completed_steps * args.total_batch_size,
                                "epoch": epoch,
                                "epochs": completed_steps / num_update_steps_per_epoch,
                                "lr0": lr_scheduler.get_lr()[0],
                            }
                        )
                        accelerator.log(
                            log_payload,
                            step=completed_steps,
                            log_kwargs={"wandb": {"commit": True}},
                        )

                        train_metrics.reset_window()

                if (
                    accelerator.sync_gradients
                    and completed_steps % (args.logval_steps // args.total_batch_size) == 0
                ):
                    if accelerator.is_main_process:
                        logger.info(command_line)
                        logger.info(run_name)
                    perplexity, acc_val_return = run_evaluation(
                        args=args,
                        model=model,
                        eval_dataloader=eval_dataloader,
                        eval_dataloader_stride=eval_dataloader_stride,
                        accelerator=accelerator,
                        completed_steps=completed_steps,
                        train_dataloader_single=train_dataloader_single,
                        eval_dataloader_single=eval_dataloader_single,
                        tokenizer=tokenizer,
                        train_test_dataset=train_test_dataset,
                        validation_test_dataset=validation_test_dataset,
                        train_metrics=train_metrics,
                        num_update_steps_per_epoch=num_update_steps_per_epoch,
                    )
                    if perplexity < best_perplexity:
                        best_perplexity = perplexity
                        logger.info(f"New best perplexity: {best_perplexity}")
                    if acc_val_return is not None:
                        best_acc_val = max(best_acc_val, acc_val_return)
                if completed_steps >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    if args.dataset_name == "custom:math_dataset" or args.dataset_name == "custom:arc_agi":
        return acc_val_return
    else:
        return perplexity


if __name__ == "__main__":
    main()
