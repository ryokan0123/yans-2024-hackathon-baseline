import json
from typing import Any
import argparse

from utils.data_util import batch_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--submission_file", type=str, default="submission.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="evaluation.jsonl")

    args = parser.parse_args()

    if args.type == "haiku":
        from haiku.scorer import HaikuScorer

        scorer = HaikuScorer()
    elif args.type == "ogiri":
        from ogiri.scorer import OgiriScorer

        scorer = OgiriScorer()
    else:
        raise ValueError(f"Invalid type: {args.type}")

    items: list[dict[str, Any]] = []
    with open(args.submission_file, "r") as f:
        for item in f:
            items.append(json.loads(item))

    scorer_outputs: list[dict[str, Any]] = []
    for batch_items in batch_iter(items, batch_size=args.batch_size):
        scores = scorer(batch_items, [item["model_output"] for item in batch_items])
        scorer_outputs += scores

    valid_scores: list[int] = [
        output["score"] for output in scorer_outputs if output["score"] is not None
    ]
    avg_score = sum(valid_scores) / len(valid_scores)
    num_failed_parsing = len(scorer_outputs) - len(valid_scores)

    print(f"Average score: {avg_score:.2f}")
    print(f"Failed to parse {num_failed_parsing} items.")

    with open(args.output_file, "w") as f:
        for output in scorer_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
