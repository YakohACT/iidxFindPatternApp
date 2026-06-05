# iidxFindPatternApp

[textage.cc](https://textage.cc/) から beatmania IIDX の譜面データを収集し、機械学習 (KMeans クラスタリング) で譜面の傾向パターンを発見するツール。今回はレベル10~12のみを対象としている。

## 構成

- `src/main.py` — パイプラインのエントリポイント
- `src/scraper.py` — Playwright で textage.cc から譜面データを取得
- `src/features.py` — 譜面の生データから特徴量ベクトルを抽出
- `src/ml.py` — 標準化 + PCA + KMeans によるパターン発見

## セットアップ

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 実行

```bash
# テストモード: 一覧URL 1 件から 50 譜面を取得して動作確認
python src/main.py --test

# 任意の件数で取得
python src/main.py --limit 30

# 全件取得 (4 つの一覧URLすべて巡回)
python src/main.py

# 既に取得済みのキャッシュだけで学習し直す
python src/main.py --skip-scrape

# クラスタ数を明示指定
python src/main.py --skip-scrape --k 6

# 自動選択を細かく分けたいとき (シルエット指標が k=2 を選んでしまう場合の対処)
python src/main.py --skip-scrape --min-k 6          # 最低 6 クラスタを保証
python src/main.py --skip-scrape --score ch         # Calinski-Harabasz を使う
python src/main.py --skip-scrape --min-k 4 --max-k 15

# 学習済みモデルで譜面 URL の傾向を予測
python src/main.py --predict "https://textage.cc/score/0/100seckb.html?1AA00"

# 学習結果をクラスタ別に整形表示
python src/main.py --show-clusters
python src/main.py --show-clusters --show-clusters-top 50  # 各クラスタ 50 譜面まで
```

### テストモード (`--test`)

- 一覧 URL を 1 件 (`?sA11B000`) に絞り、譜面 50 件を取得
- それ以外のオプション (`--k` 等) は通常モードと同じく組み合わせ可能

### クラスタ数を増やしたいとき

KMeans + シルエットスコアの組合せは、譜面のように **連続的に変化する** データに
対して k=2 を選びがちです。本ツールではこれを緩和するため:

- **既定スコア = `combined`** : シルエット + Calinski-Harabasz を min-max 正規化して
  平均し、CH 単独より安定で、シルエット単独より細かいクラスタを選びやすい
- **`--min-k N`** : 「最低 N 個のクラスタは作る」と強制できる
  (例: `--min-k 6` で 6〜20 の範囲で最良の k を探索)
- **`--max-k N`** : 上限 (既定 20)
- **`--score {combined,silhouette,ch}`** : 指標を直接切り替え
- **`--k N`** : 完全に固定 (自動選択をスキップ)

学習時に各 k のスコアが `data/results/k_scores.csv` に保存されるので、
事後にどの k が良いか比較できます。

### 予測モード (`--predict <URL>`)

- 学習時に `data/model/model.pkl` へ保存された Scaler / KMeans / PCA を読み込み、
  指定 URL の譜面に対してクラスタ ID と傾向ラベル (例: "皿多め・トリル多")、
  全体平均との乖離が大きい特徴量 (z-score) を出力する
- 譜面ページのスクレイピング結果はキャッシュされ、再実行は高速
- 学習未実施の場合はエラーで終了するので、まず `python src/main.py` 等で
  モデルを生成すること

## 出力

- `data/cache/charts/*.json` — 譜面の生ノートデータ (キャッシュ)
- `data/results/assignments.csv` — 譜面ごとのクラスタ割当と特徴量 (全件)
- `data/results/clusters/cluster_<N>_<傾向>.csv` — **クラスタ別の譜面一覧 (重心距離順)**
- `data/results/cluster_profiles.csv` — クラスタごとの特徴量平均と傾向ラベル
- `data/results/summary.csv` — クラスタ数・シルエット・Calinski-Harabasz スコア
- `data/results/k_scores.csv` — k 候補ごとの各種スコア (k 選択の事後検証用)
- `data/results/clusters.png` — PCA 2D 散布図
- `data/model/model.pkl` — 学習済みモデル (Scaler + KMeans + PCA + 統計)。次回起動時に予測モードで再利用される

## キャッシュ

譜面ページのキャッシュは `data/cache/charts/<song_id>.json` に保存され、
`song_id` をキーに自動で再利用されるので、同じ譜面に 2 度アクセスすることは
ありません。
