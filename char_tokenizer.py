from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import unicodedata
from datasets import load_dataset
from transformers import AutoTokenizer


def all_unicode_chars():
    return [
        chr(i)
        for i in range(0x110000)
        if unicodedata.category(chr(i)) != "Cn" and not (0xD800 <= i <= 0xDFFF)
    ]


def bmp_unicode_chars():
    return [
        chr(i)
        for i in range(0x10000)
        if unicodedata.category(chr(i)) != "Cn" and not (0xD800 <= i <= 0xDFFF)
    ]


def get_char_tokenizer(dataset=["dummy"], output_file="char_tokenizer.json"):
    # 1. Create tokenizer with Unigram model
    tokenizer = Tokenizer(models.Unigram())

    # 2. Use pre-tokenizer that splits into single characters
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern="",  # Match every character
        behavior="isolated",
        invert=False,
    )

    # 3. Prepare trainer with your vocab of Unicode characters
    special_tokens = ["[UNK]", "[BOS]", "[EOS]"]
    initial_vocab = special_tokens  # +[" ","Ġ"] # + bmp_unicode_chars()
    trainer = trainers.UnigramTrainer(
        vocab_size=5000,
        special_tokens=special_tokens,
        initial_alphabet=initial_vocab,  # Force Unigram to include your char set
        unk_token="[UNK]",
        max_piece_length=1,  # Each piece is a single character
    )

    tokenizer.train_from_iterator(dataset, trainer=trainer)
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = None
    tokenizer.decoder = decoders.Sequence([])
    # save the tokenizer to a file if needed
    tokenizer.save(output_file)
    return tokenizer


def convert_to_new_tokens(tokens, tokenizer1, tokenizer2):
    new_tokens = []
    for token in tokens:
        decoded = tokenizer1.decode([token])
        decoded = decoded.replace(tokenizer1.eos_token, "[EOS]")
        decoded = decoded.replace(tokenizer1.bos_token, "[BOS]")
        decoded = decoded.replace(tokenizer1.unk_token, "[UNK]")
        # decoded = decoded.replace(tokenizer1.pad_token, "[PAD]")
        print(f"Decoded token: {decoded} {tokenizer2.encode(decoded).ids}")
        new_tokens += tokenizer2.encode(decoded).ids
    return new_tokens


if __name__ == "__main__":
    recompute = True
    if recompute:
        raw_datasets = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", trust_remote_code=True
        )
        tokenizer = get_char_tokenizer(
            raw_datasets["train"]["text"], "char_tokenizer.json"
        )

    tokenizer = Tokenizer.from_file("char_tokenizer_old.json")
    string = "hello world[EOS]hello\nworld[EOS]"
    encoded = tokenizer.encode(string, add_special_tokens=False)
    print("decoded")
    print(tokenizer.decode(encoded.ids, skip_special_tokens=False))
    assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == string
    print(encoded.ids)
    print(encoded.tokens)
    print(len(tokenizer.get_vocab()))

    print("--- new without pre_tokenizer = pre_tokenizers.ByteLevel")
    tokenizer = Tokenizer.from_file("char_tokenizer.json")
    encoded = tokenizer.encode(string, add_special_tokens=False)
    print("decoded")
    print(tokenizer.decode(encoded.ids, skip_special_tokens=False))
    assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == string
    print(encoded.ids)
    print(encoded.tokens)
    print(len(tokenizer.get_vocab()))
    print("---")
    # tokenizer2 = AutoTokenizer.from_pretrained(
    #             "gpt2", use_fast=True, trust_remote_code=True
    #         )
    # ids = tokenizer2.encode("hello world<|endoftext|>n<|endoftext|>")
    # new_tokens = convert_to_new_tokens(ids, tokenizer2, tokenizer)
    # print(new_tokens)
    # print(tokenizer.decode(new_tokens, skip_special_tokens=False))
