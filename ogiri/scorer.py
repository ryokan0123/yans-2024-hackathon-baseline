from __future__ import annotations
from typing import Any
from utils.chatgpt import OpenAIChatAPI
from utils.data_util import batch_iter
import re


class OgiriScorer:
    def __init__(self, batch_size: int = 4):
        self.openai_api = OpenAIChatAPI()
        self.batch_size = batch_size

    def __call__(
        self, input_items: list[dict[str, Any]], model_responses: list[str]
    ) -> list[dict[str, Any]]:
        evaluator_inputs: list[list[dict[str, Any]]] = []
        for input_item, response in zip(input_items, model_responses):
            if input_item["image_url"]:
                # 画像がある場合
                evaluator_inputs.append(
                    self.get_eval_image_ogiri_input(response, input_item["image_url"])
                )
            else:
                # 画像がない場合
                evaluator_inputs.append(
                    self.get_eval_text_ogiri_input(response, input_item["prompt"])
                )

        evaluator_responses: list[str] = []
        for batch_inputs in batch_iter(evaluator_inputs, batch_size=self.batch_size):
            evaluator_responses += self.openai_api.batch_generate_chat_response(
                batch_inputs
            )

        outputs: list[dict[str, Any]] = []
        for input_item, response, eval_res in zip(
            input_items, model_responses, evaluator_responses
        ):
            score = self.parse_score_from_evaluator_output(
                eval_res, valid_score_range=(1, 10)
            )
            outputs.append(
                {"id": input_item["id"], "score": score, "evaluator_response": eval_res}
            )
        return outputs

    @staticmethod
    def parse_score_from_evaluator_output(
        evaluator_output: str, valid_score_range: tuple[int, int] | None
    ) -> int | None:
        """評価者の出力から最後に出てくる int を点数として抽出します。
        パースに失敗したり、点数が範囲外の場合は None を返します。
        """
        matched = re.findall(r"(\d+)", evaluator_output)
        if not matched:
            return None

        parsed_score = int(matched[-1])
        if (
            valid_score_range
            and not valid_score_range[0] <= parsed_score <= valid_score_range[1]
        ):
            return None
        return parsed_score

    @staticmethod
    def get_eval_image_ogiri_input(
        model_output: str, image_url: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": "あなたは大喜利採点 AI です。指示に従って与えられた大喜利に点数をつけてください。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"以下はモデルの回答と、それのお題になった画像です。モデルの回答の面白さを1から10点の間で採点してください。\n"
                        f"採点にあたって、根拠を述べた上で最後に点数を `[[点数]]` というフォーマットで記入してください。（例：[[5]]）\n"
                        f"モデルの回答: {model_output}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ]

    @staticmethod
    def get_eval_text_ogiri_input(
        model_output: str, prompt: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": "あなたは大喜利採点 AI です。指示に従って与えられた大喜利に点数をつけてください。",
            },
            {
                "role": "user",
                "content": f"以下はモデルの回答と、それのお題です。モデルの回答の面白さを1から10点の間で採点してください。\n"
                "採点にあたって、根拠を述べた上で最後に点数を `[[点数]]` というフォーマットで記入してください。（例：[[5]]）\n"
                f"モデルの回答: {model_output}\n"
                f"お題: {prompt}",
            },
        ]
