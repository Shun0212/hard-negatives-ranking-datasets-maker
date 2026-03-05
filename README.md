# Hard Negatives Ranking Datasets Maker

ColBERT (PyLate) + fast-plaid を使ってハードネガティブをマイニングし、知識蒸留 (KD) 形式またはスコア付きコントラスティブ形式のデータセットを構築するツール。

## 概要

コード検索向けの (docstring, code) ペアデータセットや、CoIR (BEIR形式) データセットに対して以下を行う:

1. ColBERT モデルで全ドキュメントをエンコード
2. fast-plaid でインデックスを構築し、各クエリに対して top-K 検索
3. 正例 (positive) とハードネガティブをスコア付きで保存
4. KD 形式 / コントラスティブ形式で出力・アップロード

## セットアップ

```bash
uv sync
```

`.env` に HuggingFace トークンを設定:

```
HF_TOKEN=hf_xxxxxxxxxxxxx
```

## 使い方

### ローカル保存

```bash
# KD + コントラスティブ両方
uv run python main.py --config config/docstring_to_code.yaml --save-local ./output

# KD 形式のみ
uv run python main.py --config config/docstring_to_code.yaml --output-format kd --save-local ./output

# コントラスティブ形式のみ
uv run python main.py --config config/docstring_to_code.yaml --output-format contrastive --save-local ./output
```

### HuggingFace Hub へアップロード

```bash
uv run python main.py --config config/docstring_to_code.yaml --upload
```

### CoIR データセット

```bash
uv run python main.py --config config/coir_example.yaml --save-local ./output
```

## 出力フォーマット

### KD 形式 (Knowledge Distillation)

[lightonai/nv-embed-supervised-distill-dedup-code](https://huggingface.co/datasets/lightonai/nv-embed-supervised-distill-dedup-code) と同じ構造。3 つのサブセットで構成:

| サブセット | カラム |
|-----------|--------|
| queries | `query_id`, `query`, `split` |
| documents | `document_id`, `document`, `split` |
| scores | `query_id`, `document_ids` (list), `scores` (list), `split` |

`scores` の先頭が正例、以降はスコア降順のネガティブ候補。

### コントラスティブ形式

閾値フィルタリングなしで全ネガティブとスコアを保存:

| カラム | 型 | 説明 |
|--------|-----|------|
| `query` | str | クエリテキスト (docstring) |
| `positive` | str | 正例テキスト (code) |
| `positive_score` | float | 正例の ColBERT スコア |
| `negative_0` | str | ネガティブ 0 (最もスコアが高い) |
| `negative_0_score` | float | ネガティブ 0 のスコア |
| ... | ... | ... |
| `negative_N` | str | ネガティブ N |
| `negative_N_score` | float | ネガティブ N のスコア |

スコアが保存されているため、後から任意の閾値でフィルタリング可能。

## コンフィグ

### Paired データセット (`config/docstring_to_code.yaml`)

各行が (query, document) のペアになっているデータセット用。

```yaml
datasets:
  - name: "Shuu12121/python-treesitter-dedupe-filtered-datasetsV2"
    languages: ["python"]
    query_field: "docstring"
    documents_field: "code"
    lang_as_config: false
    dataset_type: "paired"
```

### CoIR データセット (`config/coir_example.yaml`)

BEIR 形式 (corpus / queries / qrels が分離) のデータセット用。

```yaml
datasets:
  - name: "CoIR-Retrieval/codesearchnet"
    languages: ["python", "java"]
    dataset_type: "coir"
    corpus_config: "corpus"
    queries_config: "queries"
    qrels_config: "default"
    corpus_text_field: "text"
    corpus_id_field: "_id"
    queries_text_field: "text"
    queries_id_field: "_id"
    qrels_query_id_field: "query-id"
    qrels_corpus_id_field: "corpus-id"
    qrels_score_field: "score"
```

### マイニング設定

```yaml
mining_config:
  top_k: 200              # 各クエリで検索する候補数
  encode_batch_size: 32    # ColBERT エンコードの GPU バッチサイズ
  index_batch_size: 500    # エンコードループのバッチサイズ
  query_batch_size: 500    # クエリ検索のバッチサイズ
  num_negatives: 100       # KD 形式で保存するネガティブ数
  index_dir: "./plaid_index"
  device: ""               # 空文字 = 自動検出 (cuda / cpu)
```

### アップロード設定

```yaml
upload_config:
  dataset: "Shuu12121/code_search_hard_negative_datasets"
  max_per_language: 100000  # 言語あたりの最大行数
  max_per_query: 100        # コントラスティブ形式のネガティブ数
```

## プロジェクト構成

```
src/
  config.py        YAML コンフィグ → dataclass
  data_loader.py   Paired / CoIR データセットローダー
  encoder.py       PyLate ColBERT + fast-plaid インデックス
  miner.py         ハードネガティブマイニングパイプライン
  formatter.py     KD 形式 / コントラスティブ形式への変換
  uploader.py      HuggingFace Hub アップロード
main.py            CLI エントリポイント
config/
  docstring_to_code.yaml   Paired データセットの設定
  coir_example.yaml        CoIR データセットの設定例
```

## パイプライン

```
[データセット読み込み]
       │
       ▼
[ColBERT エンコード (バッチ)]  ← GPU メモリ効率のためバッチ処理
       │
       ▼
[fast-plaid インデックス構築]  ← 全ドキュメント一括で create()
       │
       ▼
[top-K 検索]
       │
       ▼
[正例 / ネガティブ分離 + スコア付与]
       │
       ├──→ KD 形式 (queries, documents, scores)
       │
       └──→ コントラスティブ形式 (query, positive, negative_0, ..., with scores)
```

各言語は独立してマイニングされる (同一言語内のコーパスから検索)。
