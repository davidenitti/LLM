import json


def convert_string_format_to_json_like(input_string):
    """
    Converts a string like 'key:value,key:{...},key:number'
    into a JSON-like string '"key":"value","key":{...},"key":number'.

    Assumes the input format follows:
    - Top-level key-value pairs separated by commas.
    - Key-value within a pair separated by the first colon.
    - Keys are simple strings (will be quoted).
    - Values can be:
        - Simple strings (will be quoted).
        - Numbers (int or float, will remain unquoted).
        - Booleans (True/False, case-insensitive, converted to json 'true'/'false').
        - Null (null/None, case-insensitive, converted to json 'null').
        - Nested structures enclosed in {}. The content inside {} follows the same format.
    - Commas and colons inside {} are ignored for top-level splitting.
    - No escaping is handled within simple string values in the input format.
    """
    output_parts = []
    brace_level = 0
    start_index = 0
    s = input_string.strip()  # Remove leading/trailing whitespace from the whole string

    if not s:
        return ""  # Handle empty input

    # Iterate through the string to find top-level commas, respecting braces
    for i in range(len(s)):
        if s[i] == "{":
            brace_level += 1
        elif s[i] == "}":
            brace_level -= 1
        elif s[i] == "," and brace_level == 0:
            # Found a top-level segment separator
            segment = s[start_index:i].strip()
            if segment:  # Avoid processing empty segments from leading/trailing/double commas
                processed_segment = process_key_value_segment(segment)
                if processed_segment is not None:  # process_key_value_segment returns None on error
                    output_parts.append(processed_segment)
            start_index = i + 1

    # Process the last segment after the loop
    last_segment = s[start_index:].strip()
    if last_segment:
        processed_segment = process_key_value_segment(last_segment)
        if processed_segment is not None:
            output_parts.append(processed_segment)
    result = ",".join(output_parts)
    assert result.replace('"', "") == input_string.replace(" ", "")
    return result


def process_key_value_segment(segment_string):
    """
    Processes a single 'key:value' segment extracted from the main string.
    Splits by the first top-level colon, quotes the key, and processes the value.
    """
    brace_level = 0
    colon_index = -1

    # Find the first colon that is NOT inside braces {}
    for i in range(len(segment_string)):
        if segment_string[i] == "{":
            brace_level += 1
        elif segment_string[i] == "}":
            brace_level -= 1
        elif segment_string[i] == ":" and brace_level == 0:
            colon_index = i
            break

    if colon_index == -1:
        # Malformed segment: no top-level colon found
        print(f"Warning: Segment '{segment_string}' does not contain a top-level colon. Skipping.")
        return None  # Indicate failure or skip this segment

    key = segment_string[:colon_index].strip()
    value_str = segment_string[colon_index + 1 :].strip()

    # Process key: Always quote the key
    # Use json.dumps to handle potential edge cases in keys if necessary, though simple keys expected
    processed_key = json.dumps(key)

    # Process value
    processed_value = ""
    if value_str.startswith("{") and value_str.endswith("}"):
        # Value is a nested structure - recursively process its content
        nested_content = value_str[1:-1].strip()  # Get content inside {}
        processed_nested_content = convert_string_format_to_json_like(nested_content)
        processed_value = "{" + processed_nested_content + "}"  # Re-wrap in {}

    else:
        # Value is a simple type (string, number, boolean, null)
        lower_value_str = value_str.lower()

        if lower_value_str == "true" or lower_value_str == "false" or lower_value_str == "null":
            # Handle boolean and null keywords directly (case-insensitive input, lowercase output)
            processed_value = lower_value_str
        else:
            try:
                # Attempt to parse value as a number or an already valid JSON primitive (like "quoted string")
                parsed_value = json.loads(value_str)

                # If it was parsed as a number (int or float), keep its original string form
                if isinstance(parsed_value, (int, float)):
                    processed_value = value_str
                else:
                    # It was parsed successfully but isn't a number (e.g., "hello" parsed to 'hello').
                    # json.dumps the original string to get the correct JSON string representation ("hello").
                    processed_value = json.dumps(value_str)

            except json.JSONDecodeError:
                # json.loads failed: the value is likely an unquoted plain string (like SelfAttentionDist).
                # Quote it using json.dumps to handle potential special characters correctly if needed.
                processed_value = json.dumps(value_str)

    return f"{processed_key}:{processed_value}"


if __name__ == "__main__":
    # Example Usage:
    input_str = "selfatt_class:SelfAttentionDist,selfatt_class_kwargs:{att_type:dot,pos_emb_mode:sum_mul_x},emb:3"
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
