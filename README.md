# YANS 2024 ハッカソン

まず Python の環境を用意してください。

依存関係をインストール
```bash
pip install -r requirements.txt
```

## 大喜利の提出ファイル作成と評価
```bash
export OPENAI_API_KEY=sk-xxxx
python make_submission.py --type "ogiri" --output_file "ogiri_submission.jsonl"
python evaluate_submission.py --type "ogiri" --submission_file "ogiri_submission.jsonl" --output_file "ogiri_evaluation.jsonl"
```

## 俳句の提出ファイル作成と評価
```bash
export OPENAI_API_KEY=sk-xxxx
python make_submission.py --type "haiku" --output_file "haiku_submission.jsonl"
python evaluate_submission.py --type "haiku" --submission_file "haiku_submission.jsonl" --output_file "haiku_evaluation.jsonl"
```