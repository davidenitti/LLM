"""
This script defines a dataset class `MathOperationsDataset` for generating simple mathematical expressions
and their results, tokenized for use in LLM models. It also includes a custom tokenizer
for character-level tokenization of mathematical symbols.
"""
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers import decoders
import os

SYMBOLS = "0123456789+-*/=N "

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_tokenizer():
    tokenizer = Tokenizer(models.Unigram([(token, idx) for idx, token in enumerate(list(SYMBOLS))]))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer.decoder = decoders.Replace("", "")  # No-op, char-level
    char_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    char_tokenizer.add_special_tokens({"unk_token": "<unk>", "pad_token": "<pad>", "eos_token": "<eos>"})
    return char_tokenizer

def tokenizer_math(text):
    conversion = {char: i for i, char in enumerate(SYMBOLS)}
    out = []
    for char in text:
        if char in conversion:
            out.append(conversion[char])
        else:
            raise ValueError(f"Unsupported character: {char}")
    return out


class MathOperationsDataset(Dataset):
    def __init__(
        self,
        num_samples=10000,
        operations=None,
        min_number=0,
        max_number=100,
        max_train_number=None,
        padding=30,
        tokenizer=None,
        steps2think=1,
    ):
        """
        Initialize the dataset.

        Args:
            num_samples (int): Number of samples to generate.
            operations (list): List of operations to include (e.g., ['+', '-', '*', '/']).
            max_number (int): Maximum value for the numbers in the operations.
        """
        self.num_samples = num_samples
        self.operations = operations if operations else ["+", "-", "*", "/"]
        self.min_number = min_number
        self.max_number = max_number
        self.padding = padding
        self.max_train_number = max_train_number
        self.steps2think = steps2think
        assert self.steps2think >= 1, "Only steps2think>=1 is supported for now"
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
            self.tokenizer.model_max_length = self.padding
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate two random numbers
        number1 = random.randint(self.min_number, self.max_number)
        if self.max_train_number is None:
            number2 = random.randint(self.min_number, self.max_number)
        else:
            # Ensure that the numbers are not in the training range
            if number1 <= self.max_train_number:
                number2 = random.randint(self.max_train_number + 1, self.max_number)
            else:
                number2 = random.randint(self.min_number, self.max_number)
            numbers = [number1, number2]
            random.shuffle(numbers)
            number1, number2 = numbers
            assert number1 > self.max_train_number or number2 > self.max_train_number, f"Both numbers are in the training range: {number1}, {number2}"
        # Randomly select an operation
        operation = random.choice(self.operations)
        if operation == "/":
            operation_used = "//"
        else:
            operation_used = operation
        try:
            result = eval(f"{number1} {operation_used} {number2}")
        except ZeroDivisionError:
            result = "N"
        # Create the string representation
        math_expression = f"{number1}{operation}{number2}"
        for _ in range(self.steps2think):
            math_expression += f"="
        math_expression += f"{result}<eos>"
        if len(math_expression) >= self.padding:
            raise ValueError(f"Math expression too long: {math_expression}")
        out = self.tokenizer(
            math_expression, padding="max_length", max_length=self.padding, truncation=True, return_token_type_ids=False
        )
        out["labels"] = out["input_ids"].copy()
        found_start_label = False
        for i in range(len(out["labels"])):
            id = out["labels"][i]
            if not found_start_label:
                out["labels"][i] = -100
            if id == self.tokenizer.convert_tokens_to_ids("="):
                found_start_label = True
                out["labels"][i] = -100
            if out["labels"][i] == self.tokenizer.pad_token_id:
                out["labels"][i] = -100
        assert (
            len(out["input_ids"]) == self.padding
        ), f"Math expression length mismatch: len {out['input_ids']} != {self.padding}"
        del out["attention_mask"]
        return out


if __name__ == "__main__":
    dataset = MathOperationsDataset(num_samples=10, max_number=10)
    for i in range(len(dataset)):
        input_text = dataset[i]
        print(f"{input_text}")
        print(dataset.tokenizer.decode(input_text["input_ids"]))
        print(input_text["labels"])
        print(input_text["input_ids"])
