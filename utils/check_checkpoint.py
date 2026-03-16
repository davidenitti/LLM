import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from model import CustomGPTConfig, CustomGPTmodel


# Register your config and model with the Transformers library
AutoConfig.register("custom_gptv0", CustomGPTConfig)
AutoModelForCausalLM.register(CustomGPTConfig, CustomGPTmodel)


def _named_param_dict(model):
    """Return a name->tensor dict for all parameters on CPU for stable comparison."""
    cpu_model = model.cpu()
    return {name: param.detach().cpu() for name, param in cpu_model.named_parameters()}


def _report_param_diffs(reference, candidate, reference_label, candidate_label, top_k=10):
    """Print order-independent comparison stats between two parameter dictionaries."""
    ref_keys = set(reference.keys())
    cand_keys = set(candidate.keys())
    missing = sorted(ref_keys - cand_keys)
    unexpected = sorted(cand_keys - ref_keys)
    if missing:
        print(f"  Keys missing in {candidate_label} relative to {reference_label}: {len(missing)}")
        print(missing)
    else:
        print(f"  No missing keys in {candidate_label} relative to {reference_label}.")
    if unexpected:
        print(f"  Extra keys in {candidate_label}: {len(unexpected)}")
        print(unexpected)
    else:
        print(f"  No unexpected keys in {candidate_label}.")

    shared = sorted(ref_keys & cand_keys)
    shape_mismatches = []
    diffs = []
    for key in shared:
        ref_tensor = reference[key]
        cand_tensor = candidate[key]
        if ref_tensor.shape != cand_tensor.shape:
            shape_mismatches.append((key, ref_tensor.shape, cand_tensor.shape))
            continue
        with torch.no_grad():
            max_diff = (ref_tensor - cand_tensor).abs().max().item()
        diffs.append((key, max_diff))

    if shape_mismatches:
        print(f"  Shape mismatches ({len(shape_mismatches)}):")
        for key, ref_shape, cand_shape in shape_mismatches[:5]:
            print(f"    {key}: {ref_shape} vs {cand_shape}")

    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top {min(top_k, len(diffs))} |{reference_label} - {candidate_label}| diffs:")
    for name, value in diffs[:top_k]:
        print(f"    {name}: {value:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="Path to HF checkpoint dir (epoch_xxx)")
    parser.add_argument("--config", type=str, default=None, help="Same format as train.py")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    assert os.path.isdir(ckpt_dir), f"{ckpt_dir} is not a directory"

    # 1) Config as saved in checkpoint
    base_config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)

    # 2) Apply potential overrides (mimic train.py)
    if args.config is not None:
        from utils.config_utils import convert_string_format_to_json_like

        s = args.config
        s = convert_string_format_to_json_like(s)
        import json

        d = json.loads(s)
        for k, v in d.items():
            if not hasattr(base_config, k):
                raise ValueError(f"override key {k} not in config")
            setattr(base_config, k, v)

    print("Loaded config:", base_config)

    # ---- Load model via from_pretrained (path A) ----
    print("\n[Path A] AutoModelForCausalLM.from_pretrained ...")
    model_a, load_info = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        config=base_config,
        trust_remote_code=True,
        output_loading_info=True,
        _fast_init=False,
    )
    print(
        "Path A loading report — missing: {missing}, unexpected: {unexpected}, mismatched: {mismatched}".format(
            missing=len(load_info.get("missing_keys", [])),
            unexpected=len(load_info.get("unexpected_keys", [])),
            mismatched=len(load_info.get("mismatched_keys", [])),
        )
    )
    if args.verbose:
        for key in load_info.get("missing_keys", []):
            print("  [A] missing:", key)
        for key in load_info.get("unexpected_keys", []):
            print("  [A] unexpected:", key)
        for entry in load_info.get("mismatched_keys", []):
            print("  [A] mismatched:", entry)

    # ---- Load model via from_config + manual state_dict (path B) ----
    print("\n[Path B] from_config + manual load_state_dict ...")
    model_b = AutoModelForCausalLM.from_config(base_config, trust_remote_code=True)

    # Find pytorch_model file
    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith("model.bin")]
    if not pt_files:
        raise FileNotFoundError(f"No *.bin file found in {ckpt_dir}")
    if len(pt_files) > 1:
        print("Warning: multiple .bin files, using first:", pt_files[0])
    pt_path = os.path.join(ckpt_dir, pt_files[0])
    print("Using state dict:", pt_path)

    state_dict = torch.load(pt_path, map_location="cpu")
    missing, unexpected = model_b.load_state_dict(state_dict, strict=False)
    print("Missing keys when loading into model_b:", len(missing))
    if args.verbose:
        for k in missing:
            print("  MISSING:", k)
    print("Unexpected keys from checkpoint:", len(unexpected))
    if args.verbose:
        for k in unexpected:
            print("  UNEXPECTED:", k)

    # Keep everything on CPU for fair comparison and to avoid duplicate copies.
    state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}

    # ---- Compare parameter tensors between model_a and model_b ----
    print("\n[Compare] model_a vs model_b (order-independent)")
    params_a = _named_param_dict(model_a)
    params_b = _named_param_dict(model_b)
    _report_param_diffs(params_a, params_b, "model_a", "model_b", top_k=10)

    # Additional sanity: compare checkpoint tensors directly to both models.
    print("\n[Sanity] checkpoint vs model_a")
    _report_param_diffs(state_dict_cpu, params_a, "checkpoint", "model_a", top_k=5)
    print("\n[Sanity] checkpoint vs model_b")
    _report_param_diffs(state_dict_cpu, params_b, "checkpoint", "model_b", top_k=5)

    # Inspect a few specific tensors to see actual values.
    inspect_keys = [
        "transf.1.att.bias",
        "norm.weight",
        "transf.0.att.bias",
    ]
    for key in inspect_keys:
        print(f"\n[Inspect] key='{key}'")
        if key in state_dict_cpu:
            t_ckpt = state_dict_cpu[key]
            print(
                "  checkpoint: shape",
                tuple(t_ckpt.shape),
                "mean",
                t_ckpt.mean().item(),
                "std",
                t_ckpt.std().item(),
            )
            print("    first elems:", t_ckpt.view(-1)[:5].tolist())
        else:
            print("  checkpoint: MISSING")
        if key in params_a:
            t_a = params_a[key]
            print("  model_a  : shape", tuple(t_a.shape), "mean", t_a.mean().item(), "std", t_a.std().item())
            print("    first elems:", t_a.view(-1)[:5].tolist())
        else:
            print("  model_a  : MISSING")
        if key in params_b:
            t_b = params_b[key]
            print("  model_b  : shape", tuple(t_b.shape), "mean", t_b.mean().item(), "std", t_b.std().item())
            print("    first elems:", t_b.view(-1)[:5].tolist())
        else:
            print("  model_b  : MISSING")

    print("\nDone.")


if __name__ == "__main__":
    main()
