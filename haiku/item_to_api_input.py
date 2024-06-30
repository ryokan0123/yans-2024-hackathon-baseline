from typing import Any


def haiku_item_to_api_input(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": "あなたは俳句 AI です。",
        }
    ]

    if item["image_url"]:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "次の画像を見て、一句読んでください。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]},
                    },
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": item["prompt"]})

    return messages
