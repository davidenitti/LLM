import json
import sys


def convert_string_format_to_json_like(input_string):
    """
    Converts a string like 'key:value,key:{...},key:number'
    into a JSON-like string '"key":"value","key":{...},"key":number'.
    it supports lists: key:[1,2,3]
    """
    if '"' in input_string:
        return "{" + input_string + "}"  # Assume already in correct format

    output_parts = []
    brace_level = 0
    bracket_level = 0
    start_index = 0
    s = input_string.strip()

    if not s:
        return ""

    for i in range(len(s)):
        if s[i] == "{":
            brace_level += 1
        elif s[i] == "}":
            brace_level -= 1
        elif s[i] == "[":
            bracket_level += 1
        elif s[i] == "]":
            bracket_level -= 1
        elif s[i] == "," and brace_level == 0 and bracket_level == 0:
            segment = s[start_index:i].strip()
            if segment:
                processed_segment = process_key_value_segment(segment)
                if processed_segment is not None:
                    output_parts.append(processed_segment)
            start_index = i + 1

    last_segment = s[start_index:].strip()
    if last_segment:
        processed_segment = process_key_value_segment(last_segment)
        if processed_segment is not None:
            output_parts.append(processed_segment)
    result = ",".join(output_parts)
    return "{" + result + "}"


def process_key_value_segment(segment_string):
    """
    Processes a single 'key:value' segment extracted from the main string.
    Splits by the first top-level colon, quotes the key, and processes the value.
    Now also supports lists: key:[1,2,3]
    """
    brace_level = 0
    bracket_level = 0
    colon_index = -1

    for i in range(len(segment_string)):
        if segment_string[i] == "{":
            brace_level += 1
        elif segment_string[i] == "}":
            brace_level -= 1
        elif segment_string[i] == "[":
            bracket_level += 1
        elif segment_string[i] == "]":
            bracket_level -= 1
        elif segment_string[i] == ":" and brace_level == 0 and bracket_level == 0:
            colon_index = i
            break

    if colon_index == -1:
        raise ValueError(f"Segment '{segment_string}' does not contain a top-level colon.")
        print(f"Warning: Segment '{segment_string}' does not contain a top-level colon. Skipping.")
        return None

    key = segment_string[:colon_index].strip()
    value_str = segment_string[colon_index + 1 :].strip()
    processed_key = json.dumps(key)

    processed_value = ""
    if value_str.startswith("{") and value_str.endswith("}"):
        nested_content = value_str[1:-1].strip()
        processed_nested_content = convert_string_format_to_json_like(nested_content)
        processed_value = "{" + processed_nested_content + "}"
    elif value_str.startswith("[") and value_str.endswith("]"):
        # Try to parse the list using json.loads, fallback to quoting as string if fails
        try:
            parsed_list = json.loads(value_str)
            processed_value = json.dumps(parsed_list)
        except Exception:
            # If not a valid JSON list, treat as string
            processed_value = json.dumps(value_str)
    else:
        lower_value_str = value_str.lower()
        if lower_value_str == "true" or lower_value_str == "false" or lower_value_str == "null":
            processed_value = lower_value_str
        else:
            try:
                parsed_value = json.loads(value_str)
                if isinstance(parsed_value, (int, float)):
                    processed_value = value_str
                else:
                    processed_value = json.dumps(value_str)
            except json.JSONDecodeError:
                processed_value = json.dumps(value_str)

    return f"{processed_key}:{processed_value}"


def compact_run_name(run_name):
    run_name = run_name.replace("selfatt_class", "att")
    run_name = run_name.replace('"', "").replace("'", "")
    run_name = run_name.replace("dropout:0", "d")
    run_name = run_name.replace(",rotary_emb:", "rot")
    run_name = run_name.replace("vocab_size:", "v:")
    run_name = run_name.replace("context_len", "T")
    run_name = run_name.replace("gptv0 ", "")
    run_name = run_name.replace("  ", " ")
    run_name = run_name.replace("true", "t")
    run_name = run_name.replace("false", "f")
    run_name = run_name.replace("selfatt_class_kwargs", "block")
    run_name = run_name.replace("n_layer:", "layer")
    run_name = run_name.replace(",v:19,T:9400", "")
    run_name = run_name.replace("lr0.000", "lr")
    run_name = run_name.replace("gt_labels_only:", "gtlbl:")
    run_name = run_name.replace("include_", "")
    run_name = run_name.replace("mult_test", "mtest")

    return run_name


def get_explicit_cli_args(parser, args_list):
    args_list = sys.argv[1:] if args_list is None else list(args_list)
    option_string_actions = parser._option_string_actions
    explicit_dests = set()
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg == "--":
            break
        if arg.startswith("-"):
            option = arg
            if "=" in arg:
                option = arg.split("=", 1)[0]
            action = option_string_actions.get(option)
            if action is None:
                i += 1
                continue
            explicit_dests.add(action.dest)
            if "=" in arg:
                i += 1
                continue
            nargs = action.nargs
            if nargs is None:
                i += 2
                continue
            if nargs == 0:
                i += 1
                continue
            if isinstance(nargs, int):
                i += 1 + nargs
                continue
            if nargs in ("?", "*", "+"):
                i += 1
                while i < len(args_list):
                    next_arg = args_list[i]
                    if next_arg == "--":
                        break
                    if next_arg in option_string_actions:
                        break
                    i += 1
                continue
        i += 1
    return explicit_dests


if __name__ == "__main__":
    # Example Usage:
    input_str = (
        "selfatt_class:SelfAttentionDist,selfatt_class_kwargs:{att_type:dot,pos_emb_mode:sum_mul_x},emb:3"
    )
    output_str = convert_string_format_to_json_like(input_str)
    print(f"Input:  '{input_str}'")
    print(f"Output: '{output_str}'")

    input_str_malformed = (
        "selfatt_class:SelfAttentionDist,selfatt_class_kwargs:{att_type:dot,pos_emb_mode:sum_mul_x},emb:3"
    )
    output_str_malformed = convert_string_format_to_json_like(input_str_malformed)
    print(f"\nInput:  '{input_str_malformed}' (with malformed quote)")
    print(f"Output: '{output_str_malformed}'")

    print("-" * 20)
    print(f"Input:  'single_key:single_value'")
    print(
        f"Output: '{convert_string_format_to_json_like('single_key:single_value')}'"
    )  # Output: '"single_key":"single_value"'

    example = "number_val:123,string_val:abc,nested_empty:{}"
    print(f"Input:  ", example)
    print(
        f"Output: '{convert_string_format_to_json_like(example)}'"
    )  # Output: '"number_val":123,"string_val":"abc","nested_empty":{}'

    print(f"Input:  'bool_val:True,null_val:null'")
    print(
        f"Output: '{convert_string_format_to_json_like('bool_val:true,null_val:null')}'"
    )  # Output: '"bool_val":true,"null_val":null'

    complex_nested = "complex_nested:{level1:{level2a:val2a,level2b:val2b},level1b:val1b}"
    print(f"Input:  ", complex_nested)
    print(
        f"Output: '{convert_string_format_to_json_like(complex_nested)}'"
    )  # Output: '"complex_nested":{"level1":{"level2a":"val2a","level2b":"val2b"},"level1b":"val1b"}'

    print(f"Input:  ''")
    print(f"Output: '{convert_string_format_to_json_like('')}'")  # Output: ''
