"""IIDX 譜面パターン発見パイプラインのエントリポイント。

実行手順:
    1. textage.cc の 4 つの一覧ページから譜面 URL を収集する。
    2. 各譜面ページにアクセスしてノートデータを取得・キャッシュする。
    3. 譜面データを特徴量ベクトルに変換する。
    4. KMeans でクラスタリングし、結果を data/results/ に書き出す。
    5. 学習済みモデルを data/model/model.pkl に保存する。
       次回以降は --predict <URL> で URL の傾向を分類できる。

CLI 例:
    python src/main.py                          # 全件取得 + 学習
    python src/main.py --test                   # テスト: 50 件のみで動作確認
    python src/main.py --limit 30               # 30 件で学習
    python src/main.py --skip-scrape            # キャッシュのみで再学習
    python src/main.py --k 6                    # クラスタ数を 6 に固定
    python src/main.py --predict <URL>          # 学習済みモデルで URL の譜面を分類
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from features import extract_features, stack
from ml import find_patterns, load_model, predict, save_model, save_report
from scraper import INDEX_URLS, ChartRecord, TextageScraper


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"
MODEL_DIR = DATA_DIR / "model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IIDX 譜面パターン発見")
    p.add_argument("--limit", type=int, default=None, help="取得する譜面数の上限")
    p.add_argument("--skip-scrape", action="store_true", help="既存キャッシュのみ使用")
    p.add_argument(
        "--test",
        action="store_true",
        help="テストモード: 一覧URLを 1 つに絞り 50 件の譜面で動作確認する。",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="KMeans のクラスタ数 (省略時はスコア最大で自動決定)",
    )
    p.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="自動決定の下限 (既定 2)。シルエットが k=2 を選びすぎる場合に "
             "5 などへ上げると強制的に細かく分けられる。",
    )
    p.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="自動決定の上限 (既定 20)",
    )
    p.add_argument(
        "--score",
        choices=["combined", "silhouette", "ch"],
        default="combined",
        help="クラスタ数選択の指標 (既定 combined: シルエットと Calinski-Harabasz の平均)",
    )
    p.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="URL",
        help="学習済みモデルを使って指定 URL の譜面の傾向を予測する",
    )
    p.add_argument(
        "--show-clusters",
        action="store_true",
        help="前回学習した結果 (assignments.csv) をクラスタ別に整形表示する",
    )
    p.add_argument(
        "--show-clusters-top",
        type=int,
        default=20,
        metavar="N",
        help="--show-clusters 時に各クラスタから表示する譜面数 (既定 20)",
    )
    args = p.parse_args()
    if args.test and args.limit is None:
        args.limit = 50
    return args


def load_cached_records() -> list[ChartRecord]:
    charts_dir = CACHE_DIR / "charts"
    if not charts_dir.exists():
        return []
    records: list[ChartRecord] = []
    for path in sorted(charts_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(ChartRecord(**data))
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[main] キャッシュ読み込みエラー ({path.name}): {e}")
    return records


async def scrape(args: argparse.Namespace) -> list[ChartRecord]:
    scraper = TextageScraper(cache_dir=CACHE_DIR)
    # テスト時は一覧 URL を 1 件に絞って高速化
    index_urls = INDEX_URLS[:1] if args.test else INDEX_URLS
    if args.test:
        print(
            f"[main] テストモード: 一覧 1 件 ({index_urls[0]}) から "
            f"最大 {args.limit} 譜面を取得します"
        )
    return await scraper.run(index_urls, limit=args.limit)


def _record_to_feature(rec: ChartRecord):
    return extract_features(
        song_id=rec.song_id,
        raw_notes=rec.raw_notes,
        title=rec.title,
        level=rec.level,
        difficulty=rec.difficulty,
        long_notes=getattr(rec, "long_notes", None),
        charge_notes=getattr(rec, "charge_notes", None),
    )


def run_ml(
    records: list[ChartRecord],
    k: int | None,
    k_min: int = 2,
    k_max: int = 20,
    score_method: str = "combined",
) -> None:
    if not records:
        print("[main] 譜面データがありません。スクレイピング結果を確認してください。")
        return

    features = []
    skipped = 0
    for rec in records:
        fv = _record_to_feature(rec)
        if fv.values[0] > 0:
            features.append(fv)
        else:
            skipped += 1
    if skipped:
        print(f"[main] ノート数 0 でスキップ: {skipped} 件")

    print(f"[main] 有効な特徴量ベクトル: {len(features)} 件")
    matrix, items = stack(features)
    if matrix.shape[0] < 2:
        print("[main] 譜面数が少なすぎてクラスタリングできません")
        return

    result = find_patterns(
        matrix,
        items,
        n_clusters=k,
        k_min=k_min,
        k_max=k_max,
        score_method=score_method,
    )
    print(
        f"[main] クラスタ数={result.n_clusters} "
        f"silhouette={result.silhouette:.3f} "
        f"CH={result.ch_score:.1f}"
    )
    if not result.k_scores.empty:
        print("[main] k 候補ごとのスコア:")
        print(result.k_scores.round(4).to_string())
    save_report(result, items, RESULTS_DIR)
    print(f"[main] 結果を書き出しました: {RESULTS_DIR}")

    model_path = save_model(result, MODEL_DIR)
    print(f"[main] モデルを保存しました: {model_path}")
    print("[main] 各クラスタの傾向:")
    for cid, name in sorted(result.cluster_names.items()):
        n_charts = int((result.labels == cid).sum())
        print(f"  cluster {cid} ({n_charts} 件): {name}")
    print(
        f"[main] クラスタ別 CSV: {RESULTS_DIR / 'clusters'} "
        "(--show-clusters で内容を表示できます)"
    )


def show_clusters(top_n: int) -> int:
    """data/results/assignments.csv をクラスタ別に整形表示する。"""
    assignments = RESULTS_DIR / "assignments.csv"
    if not assignments.exists():
        print(
            f"[main] {assignments} が見つかりません。"
            "先に学習を実行してください (`python src/main.py --skip-scrape` 等)。"
        )
        return 1

    df = pd.read_csv(assignments)
    if df.empty:
        print("[main] 学習結果が空です")
        return 1

    # 重心からの近さで並べたいので PCA 座標と距離を擬似的に使う
    # (assignments.csv には distance が含まれないため、cluster 平均との距離を計算)
    feature_cols = [
        c for c in df.columns
        if c not in {
            "song_id", "title", "level", "difficulty",
            "cluster", "cluster_name", "pca_x", "pca_y",
        }
    ]
    cluster_means = df.groupby("cluster")[feature_cols].mean()

    def dist(row) -> float:
        c = int(row["cluster"])
        diff = row[feature_cols].to_numpy(dtype=float) - cluster_means.loc[c].to_numpy()
        return float((diff ** 2).sum() ** 0.5)

    df = df.assign(_dist=df.apply(dist, axis=1))

    total = len(df)
    n_clusters = df["cluster"].nunique()
    print(f"\n=== クラスタ別表示  (合計 {total} 譜面 / {n_clusters} クラスタ) ===\n")

    for cluster_id, group in df.groupby("cluster"):
        name = group["cluster_name"].iloc[0] if "cluster_name" in group else ""
        n = len(group)
        print(f"━━━ cluster {int(cluster_id)} : {name}  ({n} 譜面) ━━━")
        # 重心に近い順に並べ替えて先頭 top_n 件を出す
        ordered = group.sort_values("_dist").head(top_n)
        for _, r in ordered.iterrows():
            level = r["level"]
            level_s = "??" if pd.isna(level) else f"Lv{int(level):>2}"
            diff = r["difficulty"] if not pd.isna(r["difficulty"]) else "???"
            notes = r.get("total_notes", "")
            notes_s = f"notes={int(notes)}" if notes != "" and not pd.isna(notes) else ""
            print(
                f"  [{level_s} {diff:<11}] {str(r['title'])[:36]:<36} "
                f"{notes_s:<12} {r['song_id']}"
            )
        if n > top_n:
            print(f"  ... 他 {n - top_n} 譜面")
        print()

    # クラスタ別 CSV の場所も案内する
    clusters_dir = RESULTS_DIR / "clusters"
    if clusters_dir.exists():
        files = sorted(clusters_dir.glob("cluster_*.csv"))
        if files:
            print(f"クラスタ別 CSV: {clusters_dir}")
            for f in files:
                print(f"  - {f.name}")
    return 0


async def predict_url(url: str) -> int:
    """学習済みモデルを使って 1 つの URL の傾向を予測する。"""
    model = load_model(MODEL_DIR)
    if model is None:
        print(
            f"[main] 学習済みモデルが見つかりません ({MODEL_DIR}/model.pkl)。"
            "先に `python src/main.py` または `--skip-scrape` で学習してください。"
        )
        return 1

    print(f"[main] モデル: クラスタ数={model.n_clusters} "
          f"silhouette={model.silhouette:.3f}")

    scraper = TextageScraper(cache_dir=CACHE_DIR)
    record = await scraper.fetch_one(url)
    if record is None:
        print(f"[main] 譜面取得に失敗しました: {url}")
        return 2

    fv = _record_to_feature(record)
    if fv.values[0] == 0:
        print("[main] ノートデータを抽出できませんでした (空譜面?)")
        return 3

    pred = predict(model, fv)

    print("\n=== 予測結果 ===")
    print(f"  URL       : {record.url}")
    print(f"  タイトル  : {record.title} / {record.artist}")
    print(f"  難易度    : {record.difficulty} {record.play_side} (Lv {record.level})")
    print(f"  ノート数  : {record.notes_count}  (小節数={record.measures})")
    print()
    print(f"  クラスタ  : {pred.cluster}")
    print(f"  傾向ラベル: {pred.cluster_name}")
    print(f"  重心距離  : {pred.distance:.3f}  (近い順 {pred.distance_rank} 番目)")
    print(f"  PCA座標   : ({pred.pca_xy[0]:.3f}, {pred.pca_xy[1]:.3f})")
    print()
    print("  特徴量 z-score (全体平均との乖離が大きい順):")
    for name, z in pred.top_features:
        v = pred.feature_values[name]
        marker = "▲" if z > 0 else "▼"
        print(f"    {marker} {name:18s} value={v:.3f}  z={z:+.2f}")
    return 0


def main() -> int:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.show_clusters:
        return show_clusters(args.show_clusters_top)

    if args.predict:
        return asyncio.run(predict_url(args.predict))

    if args.skip_scrape:
        records = load_cached_records()
        print(f"[main] キャッシュから {len(records)} 件読み込みました")
    else:
        records = asyncio.run(scrape(args))
        cached = load_cached_records()
        seen = {r.song_id for r in records}
        for r in cached:
            if r.song_id not in seen:
                records.append(r)

    run_ml(
        records,
        k=args.k,
        k_min=args.min_k,
        k_max=args.max_k,
        score_method=args.score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
