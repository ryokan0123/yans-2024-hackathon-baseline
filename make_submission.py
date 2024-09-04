import datasets
import json
from typing import Any
import argparse
import tqdm
from loguru import logger

from utils.chatgpt import OpenAIChatAPI
from utils.data_util import batch_iter, encode_image_to_base64


def ogiri_item_to_api_input(item: dict[str, Any]) -> list[dict[str, Any]]:
    """
    大喜利データセットの各アイテムを ChatGPT API への入力形式に変換する関数。
    例としてシンプルなプロンプトを実装していますが、適宜変更して使用してください。
    """
    messages = [
        {
            "role": "system",
            "content": "あなたはユーモアに溢れた大喜利 AI です。与えられたお題に対して、面白い回答をお願いします。回答のみを出力してください。",
        }
    ]
    if item["type"] == "text_to_text":
        # テキストのみを入力としたお題
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item["odai"]},
                ],
            }
        )

    elif item["type"] == "image_to_text":
        # ローカルの画像を OpenAI API に渡す場合は、画像を base64 エンコードした文字列を渡す
        # `encode_image_to_base64`で PIL 形式の画像を base64 エンコードした文字列に変換
        image_base64 = encode_image_to_base64(item["image"])

        # 画像で一言のお題
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "次の画像を見て、何か面白い一言をお願いします。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low", # 低解像度モード。API使用量を節約するために基本このオプションを使用してください
                        },
                    },
                ],
            }
        )

    elif item["type"] == "image_text_to_text":
        image_base64 = encode_image_to_base64(item["image"])

        # 画像中の空欄を埋めるお題
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"画像に含まれる{item['odai']}というテキストの[空欄]を面白くなるように埋めてください。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low", # 低解像度モード。API使用量を節約するために基本このオプションを使用してください
                        },
                    },
                ],
            }
        )
    else:
        raise ValueError(f"Invalid type: {item['type']}")

    return messages


def senryu_item_to_api_input(item: dict[str, Any]) -> list[dict[str, Any]]:
    """
    川柳データセットの各アイテムを ChatGPT API への入力形式に変換する関数。
    例としてシンプルなプロンプトを実装していますが、適宜変更して使用してください。
    """
    messages = [
        {
            "role": "system",
            "content": "あなたは川柳 AI です。与えられたお題に対して、一句詠んでください。回答となる川柳のみを出力してください。",
        }
    ]
    if item["type"] == "text_to_text":
        # テキストのみを入力としたお題
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item["odai"]},
                ],
            }
        )

    elif item["type"] == "image_to_text":
        # ローカルの画像を OpenAI API に渡す場合は、画像を base64 エンコードした文字列を渡す
        # `encode_image_to_base64`で PIL 形式の画像を base64 エンコードした文字列に変換
        image_base64 = encode_image_to_base64(item["image"])

        # 画像で一言のお題
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "次の画像にちなんで、一句詠んでください。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low", # 低解像度モード。API使用量を節約するために基本このオプションを使用してください
                        },
                    },
                ],
            }
        )
    else:
        raise ValueError(f"Invalid type: {item['type']}")

    return messages


if __name__ == "__main__":
    item_to_api_input = ogiri_item_to_api_input  # 大喜利データセットを使う場合
    # item_to_api_input = senryu_item_to_api_input  # 川柳データセットを使う場合
    # if "item_to_api_input" not in locals():
    #     msg = "用いるデータセットに適した item_to_api_input を選択してください。"
    #     raise RuntimeError(msg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="submission.jsonl")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.dataset_name}...")
    dataset = datasets.load_dataset(args.dataset_name, split="test")

    openai_api = OpenAIChatAPI()

    logger.info("Generating model outputs...")
    # バッチごとにモデルの出力を取得
    model_outputs: list[dict[str, Any]] = []
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for i, items in enumerate(batch_iter(dataset, batch_size=args.batch_size)):
            api_input_list = [item_to_api_input(item) for item in items]
            responses = openai_api.batch_generate_chat_response(api_input_list)
            model_outputs += [
                {**item, "model_output": res} for item, res in zip(items, responses)
            ]
            pbar.update(len(items))

    logger.info(f"Saving model outputs to {args.output_file}...")
    # モデルの出力を保存
    with open(args.output_file, "w") as f:
        for output in model_outputs:
            output.pop("image")  # 画像は出力に含めない
            f.write(json.dumps(output, ensure_ascii=False, default=str) + "\n")
