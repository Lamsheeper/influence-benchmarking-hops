import argparse
import json

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSON array file to JSONL. "
            "By default, converts in place (output path defaults to input path)."
        )
    )
    parser.add_argument("input", help="Path to input JSON file (array of objects).")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional path to output JSONL file. Defaults to the input path (in-place conversion).",
    )
    args = parser.parse_args()

    output_path = args.output or args.input

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)  # expects: [ { ... }, { ... }, ... ]

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at top level, got {type(data)}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for obj in data:
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()