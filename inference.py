from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from model import CustomGPTConfig, CustomGPTmodel


def generate(model, tokenizer, prompt="Once upon a time in a faraway land", temperature=0.7, top_k=None, stop_at_eos=True):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to("cuda")
    if stop_at_eos:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = None
    output = model.generate(input_ids, max_length=100, temperature=temperature, top_k=top_k,
                             eos_token_id=eos_token_id)

    generated_text = tokenizer.decode(output[0])  # , skip_special_tokens=True)
    print(generated_text)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a pre-trained language model.")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the directory where the model and tokenizer are saved."
    )
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the generated text.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for text generation.")
    args = parser.parse_args()
    # Path to the directory where the model and tokenizer were saved

    # Register your config and model with the Transformers library
    AutoConfig.register("custom_gptv0", CustomGPTConfig)
    AutoModelForCausalLM.register(CustomGPTConfig, CustomGPTmodel)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    while True:
        print("INSERT PROMPT")
        prompt = input()
        print("REPLY")
        generate(model, tokenizer, prompt=prompt, temperature=args.temperature)


if __name__ == "__main__":
    main()
