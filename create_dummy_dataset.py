from datasets import Dataset

item_list = [
    {"id": 0, "prompt": "世紀末の日本にありがちなことは？", "image_url": None},
    {"id": 1, "prompt": "こんなNLP（自然言語処理）の学会はいやだ", "image_url": None},
    {
        "id": 2,
        "prompt": None,
        "image_url": "https://d2dcan0armyq93.cloudfront.net/photo/odai/600/4c6e3919bd635570f028368b0c5bb0d1_600.jpg",
    },
    {
        "id": 3,
        "prompt": None,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Johannes_Vermeer_-_De_melkmeid.jpg/1280px-Johannes_Vermeer_-_De_melkmeid.jpg",
    },
]

dataset = Dataset.from_list(item_list)
dataset.push_to_hub("ryo0634/ogiri-test-data", private=True, split="test")


item_list = [
    {"id": 0, "prompt": "コオロギ", "image_url": None},
    {"id": 1, "prompt": "たこ焼き", "image_url": None},
    {
        "id": 2,
        "prompt": None,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/US_long_grain_rice.jpg/440px-US_long_grain_rice.jpg",
    },
    {
        "id": 3,
        "prompt": None,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/44/6600GT_GPU.jpg",
    },
]

dataset = Dataset.from_list(item_list)
dataset.push_to_hub("ryo0634/haiku-test-data", private=True, split="test")
