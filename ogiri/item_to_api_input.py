from typing import Any


def ogiri_item_to_api_input(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": "あなたはユーモアに溢れた大喜利 AI です。",
        }
    ]

    if item["image_url"]:
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
                        "image_url": {"url": item["image_url"]},
                    },
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": item["prompt"]})

    return messages
