# YANS 2024 言語芸術生成ハッカソン

大喜利または川柳を生成するモデルを作成してください。
まず Python の環境を用意してください。

依存関係をインストール
```bash
pip install -r requirements.txt
```
poetry を使っている方は `poetry install` でも可。

## 提出ファイル作成

以下のコマンドで提出ファイルを作成できます。

```bash
# OpenAIのAPIキーを設定
export OPENAI_API_KEY=sk-xxxx
# 大喜利の動作確認
python make_submission.py --dataset_name "YANS-official/ogiri-debug" --output_file "ogiri_submission.jsonl"
# 川柳の動作確認
python make_submission.py --dataset_name "YANS-official/senryu-debug" --output_file "senryu_submission.jsonl"
```

`make_submission.py` で呼び出される実装を、チーム独自のシステムに置き換えてください。
本番では `--dataset_name` に `YANS-official/ogiri-test` または `YANS-official/senryu-test` を指定することになります。

## ノートブック
Google Colabで実行できるノートブックも提供しています。

- 大喜利: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokan0123/yans-2024-hackathon-baseline/blob/main/notebook_ogiri.ipynb)
- 川柳: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokan0123/yans-2024-hackathon-baseline/blob/main/notebook_senryu.ipynb)
