import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from model import CustomGPTConfig, CustomGPTmodel
from utils.amp_utils import get_amp_context, get_model_device


def generate(
    model, tokenizer, prompt, precision, max_length=100, temperature=0.7, top_k=None, train_progress=1.0
):
    ctx = get_amp_context(precision)
    with torch.inference_mode(), ctx:
        model.eval()
        device = get_model_device(model)
        input_ids = tokenizer(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        output = model.generate(
            **input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            train_progress=train_progress,
        )

        generated_text = tokenizer.decode(output[0])  # , skip_special_tokens=True)
        print(generated_text)
        model.train()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a pre-trained language model.")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the directory where the model and tokenizer are saved.",
    )
    parser.add_argument("--assistant", action="store_true", help="Use assistant mode for the model.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=300,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation.",
    )
    args = parser.parse_args()
    # Path to the directory where the model and tokenizer were saved

    # Register your config and model with the Transformers library
    AutoConfig.register("custom_gptv0", CustomGPTConfig)
    AutoModelForCausalLM.register(CustomGPTConfig, CustomGPTmodel)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    num_m_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters: {num_m_param:.1f}M")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    while True:
        print("INSERT PROMPT")
        prompt = input()
        if prompt == "":
            print("Exiting...")
            break
        if args.assistant:
            prompt = f"User: {prompt}<|endoftext|>Assistant:"
        print("REPLY")
        generate(
            model,
            tokenizer,
            prompt=prompt,
            precision="fp16",
            max_length=args.max_length,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
