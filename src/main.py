"""IIDX 譜面パターン発見パイプラインのエントリポイント。

実行手順:
    1. textage.cc の 4 つの一覧ページから譜面 URL を収集する。
    2. 各譜面ページにアクセスしてノートデータを取得・キャッシュする。
       (textage.cc へのアクセスは 1 時間あたり 50 回まで)
    3. 譜面データを特徴量ベクトルに変換する。
    4. KMeans でクラスタリングし、結果を data/results/ に書き出す。

CLI 例:
    python src/main.py                # 全件取得
    python src/main.py --limit 30     # 動作確認用に 30 件のみ
    python src/main.py --skip-scrape  # 既存キャッシュのみで ML を実行
    python src/main.py --k 6          # クラスタ数を 6 に固定
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from features import extract_features, stack
from ml import find_patterns, save_report
from rate_limiter import RateLimiter
from scraper import INDEX_URLS, ChartRecord, TextageScraper


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IIDX 譜面パターン発見")
    p.add_argument("--limit", type=int, default=None, help="取得する譜面数の上限")
    p.add_argument("--skip-scrape", action="store_true", help="既存キャッシュのみ使用")
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="KMeans のクラスタ数 (省略時はシルエット最大で自動決定)",
    )
    p.add_argument(
        "--max-requests",
        type=int,
        default=50,
        help="1 時間あたりの最大リクエスト数 (既定 50)",
    )
    return p.parse_args()


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
    limiter = RateLimiter(
        max_requests=args.max_requests,
        window_seconds=3600,
        state_file=CACHE_DIR / "rate_limit_state.json",
    )
    scraper = TextageScraper(cache_dir=CACHE_DIR, rate_limiter=limiter)
    return await scraper.run(INDEX_URLS, limit=args.limit)


def run_ml(records: list[ChartRecord], k: int | None) -> None:
    if not records:
        print("[main] 譜面データがありません。スクレイピング結果を確認してください。")
        return

    features = []
    for rec in records:
        fv = extract_features(
            song_id=rec.song_id,
            raw_notes=rec.raw_notes,
            title=rec.title,
            level=rec.level,
            difficulty=rec.difficulty,
        )
        if fv.values.sum() > 0:  # 完全に空の譜面は除外
            features.append(fv)

    print(f"[main] 有効な特徴量ベクトル: {len(features)} 件")
    matrix, items = stack(features)
    if matrix.shape[0] < 2:
        print("[main] 譜面数が少なすぎてクラスタリングできません")
        return

    result = find_patterns(matrix, items, n_clusters=k)
    print(
        f"[main] クラスタ数={result.n_clusters} "
        f"silhouette={result.silhouette:.3f}"
    )
    save_report(result, items, RESULTS_DIR)
    print(f"[main] 結果を書き出しました: {RESULTS_DIR}")


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_scrape:
        records = load_cached_records()
        print(f"[main] キャッシュから {len(records)} 件読み込みました")
    else:
        records = asyncio.run(scrape(args))
        # キャッシュ全体も合わせて学習に使う
        cached = load_cached_records()
        seen = {r.song_id for r in records}
        for r in cached:
            if r.song_id not in seen:
                records.append(r)

    run_ml(records, args.k)


if __name__ == "__main__":
    main()
