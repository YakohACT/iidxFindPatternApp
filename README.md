# iidxFindPatternApp

[textage.cc](https://textage.cc/) から beatmania IIDX の譜面データを収集し、機械学習 (KMeans クラスタリング) で譜面の傾向パターンを発見するツール。今回はレベル10~12のみを対象としている。

## 構成

- `src/main.py` — パイプラインのエントリポイント
- `src/scraper.py` — Playwright で textage.cc から譜面データを取得 (レート制限つき)
- `src/textage_notes.py` — textage の譜面データ形式のデコーダ (レンダラ bms2jsh.js の移植)
- `src/patterns.py` — 階段・デニム・縦連・連皿などの配置パターン検出器
- `src/random_sim.py` — RANDOM オプションの当たり外れシミュレーション (7! 順列の運指コスト分布)
- `src/features.py` — デコード済みノート列から特徴量ベクトルを抽出
- `src/ml.py` — 標準化 + KMeans によるパターン発見 (PCA は可視化用)
- `src/browse.py` — 対話式の譜面ブラウザ (曲名 50 音順 / レベル別に単体情報を参照)

## textage のデータ形式について

譜面ページの `sp` / `dp` 配列には 3 種類の行が混在しており、すべてデコードする:

1. **plain hex 行** — 2 桁 (1 byte) = 1 時刻、bit 0 = スクラッチ、bit 1..7 = 鍵盤 1〜7。
   行の長さは可変 (16 桁固定ではない) で、行長がその小節の分解能を決める
2. **`x` プレフィクス行** — 行長を明示し `@nn` で空白をスキップするランレングス表記
3. **`#` 圧縮行** — 周期パターンや個別ノートを b64 でエンコードした圧縮表記

`ln` 配列は各小節の長さ (384 = 4/4 拍子、1 拍 = 96) で、CN/HCN/BSS は `cn` 配列に入る。
デコーダは textage 本体の統計出力 (ノート総数・レーン別数・位置) と一致することを確認済み。

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

# 既に取得済みのキャッシュだけで学習し直す (ネットワーク不要)
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

# 対話式の譜面ブラウザ (曲単体の情報を参照)
python src/main.py --browse
```

### 譜面ブラウザ (`--browse`)

学習済みデータにある譜面を選択して、単体の情報
(基本情報 / クラスタ / 全特徴量の値と z-score / RANDOM オプション判断) を表示する。

- **[1] 曲名 (50 音順) から選ぶ** — あ〜わ行 / A-Z (頭文字別) / 数字 / 漢字・他 の
  グループから曲を選び、難易度 (BEGINNER〜LEGGENDARIA のうち収録があるもの) を選ぶ
- **[2] レベル (1-12) から選ぶ** — レベルを選ぶと該当譜面が 50 音順で一覧される

備考:

- かな曲名はカタカナ→ひらがな・濁点・拗音を正規化して正確な 50 音順で並ぶ。
  漢字始まりの曲名は読み仮名データが存在しないため (textage 側にも無い)、
  「漢字・他」グループに文字コード順でまとめている
- 事前に学習 (`--skip-scrape` 等) を済ませて `data/results/assignments.csv` を
  生成しておくこと。現在のデータセットは Lv10-12 の一覧が取得元のため、
  実際に選べる難易度は HYPER / ANOTHER / LEGGENDARIA が中心

### テストモード (`--test`)

- 一覧 URL を 1 件 (`?sA11B000`) に絞り、譜面 50 件を取得
- それ以外のオプション (`--k` 等) は通常モードと同じく組み合わせ可能

### 学習前の前処理

- **1P/2P の統合**: textage の 1P/2P 表示はノートデータが同一のため、
  学習時は 1 譜面に統合される (1P 優先)。統合前後の件数はログに出る
- **異常譜面の除外**: ノート数が `--min-notes` (既定 50) 未満の譜面は
  デコード失敗・データ不備とみなして学習から除外する

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

- 学習時に `data/model/model.pkl` へ保存された Scaler / KMeans / PCA / 母集団統計を
  読み込み、指定 URL の譜面に対してクラスタ ID と傾向ラベル (例: "皿多め・トリル多")、
  学習母集団の平均との乖離が大きい特徴量 (z-score) を出力する
- 譜面ページのスクレイピング結果はキャッシュされ、キャッシュ命中時は
  ブラウザを起動せずネットワークにもアクセスしない
- 学習未実施の場合はエラーで終了するので、まず `python src/main.py` 等で
  モデルを生成すること

## 特徴量 (32 次元)

### 基本量

| 特徴量 | 意味 |
|---|---|
| total_notes | 総ノート数 (CN/BSS の開始を含む) |
| measures | 最初のノート〜最後のノートの長さ (4/4 小節換算) |
| density | ノート数 / 小節 |
| scratch_ratio | スクラッチ比率 |
| mean_chord / max_chord | 同時押しの平均・最大サイズ |
| peak_density | 2 拍幅の滑走窓に入る最大ノート数 (発狂の検出) |
| stream16_ratio | 16 分間隔で 8 イベント以上続く塊 (乱打) のノート比率 |

### 配置パターン (patterns.py による検出)

| 特徴量 | 配置 | 判定基準 |
|---|---|---|
| stair_mono_ratio | 単階段 | 単ノートが隣接レーンへ同方向に 3 連以上 (3 レーン以上) |
| stair_turn_ratio | 折り返し階段 | 隣接移動の連なりに方向転換があり 4 連以上 |
| big_stair_rate | 大階段 | 1→7 (7→1) を単調に駆け抜ける 7 連の小節あたり回数 |
| double_stair_ratio | 二重階段 | 2 個以上の同時押し全体が ±1 シフトして 3 連以上 (1,3→2,4→3,5) |
| garbage_stair_ratio | ゴミ付き階段 | 単調な隣接進行 4 連以上に余分な同時打鍵が混ざる |
| denim_ratio | デニム | 互いに素でレーン範囲が交錯する 3 個以上の同時押しの交互 (1,3,5,7⇔2,4,6) |
| chord_trill_ratio | 二重トリル | 互いに素な 2 つの同時押し (各 2 個以上) の A⇔B 往復 4 連以上 |
| trill_ratio | トリル | 2 レーン往復 4 打以上 (レーン間隔は問わない) |
| jack_ratio | 縦連 | 同一鍵盤を 16 分以内の間隔で 3 打以上 |
| wall_ratio | 壁 | 3 個以上の同時押しが 8 分以内で 2 連以上 |
| axis_ratio | 軸 | 同一鍵盤が 4 分以内で 4 打以上降り続け、合間の過半に他レーン (2 種類以上) |
| scratch_stream_ratio | 連皿 | スクラッチが 8 分以内の間隔で 3 打以上 |
| scratch_key_mix_ratio | 皿複合 | 前後 16 分以内に鍵盤ノートがあるスクラッチ |
| scratch_left_chord_ratio | 無理皿 | 皿と 1,2 鍵の同時押し (1P 正規基準) |
| mss_rate | マルチスピンスクラッチ | cn 配列の MSS フラグ付き BSS の小節あたり本数 |
| peak_position | 終盤発狂 | 最密 2 拍窓の曲中位置 (0=序盤, 1=ラスト。ラス殺し検出) |
| offgrid_ratio | ズレ/裏拍 | 16 分グリッド (24 の倍数) に乗らないノート (24分/32分/ズレ) |
| hand_bias | 片手偏重 | 左手側 (1-3) と右手側 (5-7) のノート数の偏り |
| cn_ratio | CN/HCN/BSS | チャージノート開始の比率 |
| bpm_range_log2 | ソフラン (幅) | log2(最大BPM/最小BPM)。1.0 = 2 倍の変化 |
| bpm_change_rate | ソフラン (頻度) | BPM 変化回数 / 小節 |

### RANDOM オプション判断 (random_sim.py によるシミュレーション)

IIDX の RANDOM は鍵盤 7 レーンの並び替え (7! = 5040 通り) なので、
簡易運指コストモデルで「正規配置」と「ランダム配置の分布」を直接比較できる。

| 特徴量 | 意味 | 読み方 |
|---|---|---|
| random_advantage | 正規がランダム平均よりどれだけ悪いか (z-score) | **+1 以上: 乱を掛ける価値あり / -1 以下: 正規当たり配置** |
| random_gamble | 乱の当たり外れの振れ幅 (イベントあたり標準偏差) | 大きいほど「当たり待ちガチャ」性が強い (GOLD RUSH 等) |
| mirror_advantage | MIRROR にするとどれだけ得か (z-score) | +1 以上で鏡を試す価値あり (Red. 系など皿+左手曲) |

判断の際は乱で **変化しない** 要素も併せて読むこと:

- `jack_ratio` (縦連) — 乱ではレーンが変わるだけで縦連は消えない → S乱の検討材料
- `scratch_stream_ratio` / `scratch_ratio` (皿) — スクラッチは乱の影響を受けない
- `random_gamble` が大きい譜面は「当たれば楽・外れれば地獄」なので粘着向き

運指コストモデルの仮定 (`src/random_sim.py` 冒頭の定数で調整可能):

- 1P 想定・3:4 分業 (左手 = 皿+1,2,3 鍵 / 右手 = 4,5,6,7 鍵)。2P は鏡の符号を読み替える
- コスト = 片手内同時押し (本数-1)² + 皿と左手鍵盤の同時 (無理皿) + 16 分連続の片手拘束
- 乱打の運指相性のような長い文脈は扱わない簡易モデル

- 連続とみなす間隔などの閾値は `src/patterns.py` 冒頭の定数で調整できる
- 判定は拍基準 (メトリック) で行い BPM に依存しない。BPM の緩急はソフラン系
  特徴量が受け持つ
- クラスタの傾向ラベルは「クラスタ平均が全体平均から +0.6σ 以上離れた特徴量」
  上位 4 つから自動生成される (例: "連皿・皿多め・皿複合")

クラスタリングは標準化した 32 次元をそのまま KMeans にかける。
**PCA は散布図など 2 次元可視化にのみ使用し、クラスタリングには使わない。**

配置パターンで細かく分けたい場合はクラスタ数を多めに強制するとよい:

```bash
python src/main.py --skip-scrape --min-k 10
```

## 出力

- `data/cache/charts/*.json` — 譜面の生ノートデータ (キャッシュ)
- `data/results/assignments.csv` — 譜面ごとのクラスタ割当と特徴量 (全件)
- `data/results/clusters/cluster_<N>_<傾向>.csv` — **クラスタ別の譜面一覧**
- `data/results/cluster_profiles.csv` — クラスタごとの特徴量平均と傾向ラベル
- `data/results/summary.csv` — クラスタ数・シルエット・Calinski-Harabasz スコア
- `data/results/k_scores.csv` — k 候補ごとの各種スコア (k 選択の事後検証用)
- `data/results/clusters.png` — PCA 2D 散布図
- `data/model/model.pkl` — 学習済みモデル (Scaler + KMeans + PCA + 母集団統計)。次回起動時に予測モードで再利用される

## キャッシュとレート制限

- 譜面ページのキャッシュは `data/cache/charts/<song_id>.json` に保存され、
  `song_id` をキーに自動で再利用されるので、同じ譜面に 2 度アクセスすることはない
- 実際にページへアクセスする際は **1 時間あたり最大 `--rph` 件 (既定 720 = 平均 5 秒間隔)**
  に制限される。アクセス時刻は `data/cache/rate_limit_state.json` に永続化され、
  プロセスを再起動しても制限が維持される
