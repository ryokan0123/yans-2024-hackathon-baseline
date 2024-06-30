import datasets
import json
from typing import Any
import argparse

from utils.chatgpt import OpenAIChatAPI
from utils.data_util import batch_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="submission.jsonl")
    args = parser.parse_args()

    if args.type == "haiku":
        from haiku.item_to_api_input import haiku_item_to_api_input as item_to_api_input

        dataset = datasets.load_dataset("ryo0634/haiku-test-data", split="test")
    elif args.type == "ogiri":
        from ogiri.item_to_api_input import ogiri_item_to_api_input as item_to_api_input

        dataset = datasets.load_dataset("ryo0634/ogiri-test-data", split="test")
    else:
        raise ValueError(f"Invalid type: {args.type}")

    openai_api = OpenAIChatAPI()

    model_outputs: list[dict[str, Any]] = []
    for items in batch_iter(dataset, batch_size=args.batch_size):
        api_input_list = [item_to_api_input(item) for item in items]
        responses = openai_api.batch_generate_chat_response(api_input_list)
        model_outputs += [
            {**item, "model_output": res} for item, res in zip(items, responses)
        ]

    with open(args.output_file, "w") as f:
        for output in model_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
