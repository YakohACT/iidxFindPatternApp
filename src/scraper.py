"""textage.cc から譜面ページをスクレイピングするモジュール。

- 一覧ページ (sA11B000, sB11B000, sC11B000, sX11B000) から
  個別譜面ページの URL を収集する。
- 各譜面ページにアクセスし、ブラウザ内のグローバル変数として
  存在する譜面ノートデータ (textage.cc では `notes`, `bd` 等) を
  そのまま JSON として書き出す。
- すべてのアクセスは `RateLimiter` を通して 1 時間 50 回までに制限する。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from playwright.async_api import async_playwright, Browser, Page

from rate_limiter import RateLimiter


INDEX_URLS = [
    "https://textage.cc/score/index.html?sA11B000",
    "https://textage.cc/score/index.html?sB11B000",
    "https://textage.cc/score/index.html?sC11B000",
    "https://textage.cc/score/index.html?sX11B000",
]


@dataclass
class ChartRecord:
    """1 譜面分の生データ。"""

    url: str
    song_id: str
    title: str | None
    difficulty: str | None
    level: int | None
    bpm: str | None
    notes_count: int | None
    raw_notes: list  # textage.cc 内部表現。各要素は [tick, lane, type] など
    measures: list  # 小節区切り情報 (取得できた場合)

    def to_dict(self) -> dict:
        return asdict(self)


# textage.cc の譜面ページ URL は ?<song_id> のような形になる。
SONG_ID_RE = re.compile(r"\?([A-Za-z0-9]+)$")


def parse_song_id(url: str) -> str:
    m = SONG_ID_RE.search(url)
    return m.group(1) if m else url


# ブラウザ内で実行し、textage.cc の内部状態から譜面情報を取り出す JS。
# textage.cc は譜面を `notes` グローバル (2 次元配列) と `score` などで保持する。
# サイトの実装が変わる可能性があるため、複数の候補名を試して最初に見つかった
# ものを返す形にしている。
EXTRACT_JS = r"""
() => {
    function pick(names) {
        for (const n of names) {
            try {
                const v = window[n];
                if (v !== undefined && v !== null) return { name: n, value: v };
            } catch (e) {}
        }
        return null;
    }

    // 譜面のノートデータ候補
    const notesCand = pick(['notes', 'note_data', 'bd', 'score_data', 'chart']);
    // 小節情報候補
    const measureCand = pick(['bar', 'bars', 'measure', 'measures', 'br']);
    // BPM 候補
    const bpmCand = pick(['bpm', 'BPM']);

    // タイトル: ページの <title> または h1 から取得
    const title = (document.querySelector('title')?.innerText
        || document.querySelector('h1')?.innerText
        || '').trim();

    // 譜面難易度・レベルはページ上のテキストから推定
    let level = null;
    let difficulty = null;
    const text = document.body ? document.body.innerText : '';
    const lvMatch = text.match(/(?:LEVEL|Lv|☆)\s*([0-9]{1,2})/i);
    if (lvMatch) level = parseInt(lvMatch[1], 10);
    const difMatch = text.match(/(BEGINNER|NORMAL|HYPER|ANOTHER|LEGGENDARIA)/i);
    if (difMatch) difficulty = difMatch[1].toUpperCase();

    // 安全に直列化できる形に変換する
    function safe(v) {
        try { JSON.stringify(v); return v; }
        catch (e) { return null; }
    }

    return {
        title,
        difficulty,
        level,
        bpm: bpmCand ? String(bpmCand.value) : null,
        notes_key: notesCand ? notesCand.name : null,
        notes: notesCand ? safe(notesCand.value) : null,
        measure_key: measureCand ? measureCand.name : null,
        measures: measureCand ? safe(measureCand.value) : null,
    };
}
"""


class TextageScraper:
    def __init__(
        self,
        cache_dir: Path,
        rate_limiter: RateLimiter,
        page_timeout_ms: int = 30000,
        render_wait_ms: int = 4000,
    ) -> None:
        self.cache_dir = cache_dir
        self.charts_dir = cache_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = rate_limiter
        self.page_timeout_ms = page_timeout_ms
        self.render_wait_ms = render_wait_ms

    async def _new_page(self, browser: Browser) -> Page:
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(self.page_timeout_ms)
        return page

    async def collect_chart_urls(self, browser: Browser, index_url: str) -> list[str]:
        await self.rate_limiter.acquire()
        page = await self._new_page(browser)
        try:
            print(f"[Scraper] 一覧ページにアクセス: {index_url}")
            await page.goto(index_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(self.render_wait_ms)
            hrefs: list[str] = await page.evaluate(
                "() => Array.from(document.querySelectorAll('a')).map(a => a.href)"
            )
        finally:
            await page.close()

        urls = sorted({
            h for h in hrefs
            if "textage.cc/score/" in h and ".html?" in h and "index.html?" not in h
        })
        print(f"[Scraper] {len(urls)} 件の譜面URLを抽出しました")
        return urls

    async def fetch_chart(self, browser: Browser, url: str) -> ChartRecord | None:
        song_id = parse_song_id(url)
        cache_path = self.charts_dir / f"{song_id}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return ChartRecord(**data)
            except (json.JSONDecodeError, TypeError):
                # 壊れたキャッシュは無視
                pass

        await self.rate_limiter.acquire()
        page = await self._new_page(browser)
        try:
            try:
                await page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                print(f"[Scraper] アクセス失敗 ({url}): {e}")
                return None
            await page.wait_for_timeout(self.render_wait_ms)

            try:
                payload = await page.evaluate(EXTRACT_JS)
            except Exception as e:
                print(f"[Scraper] 抽出失敗 ({url}): {e}")
                return None
        finally:
            await page.close()

        notes = payload.get("notes") or []
        notes_count = self._count_notes(notes)
        record = ChartRecord(
            url=url,
            song_id=song_id,
            title=payload.get("title"),
            difficulty=payload.get("difficulty"),
            level=payload.get("level"),
            bpm=payload.get("bpm"),
            notes_count=notes_count,
            raw_notes=notes if isinstance(notes, list) else [],
            measures=payload.get("measures") or [],
        )
        cache_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        return record

    @staticmethod
    def _count_notes(notes) -> int:
        # textage.cc の notes は [小節][レーン][...] のような階層になることが多い。
        # 数値要素を末端まで辿って 1 ノート相当とみなす。
        def walk(x) -> int:
            if isinstance(x, list):
                if x and not isinstance(x[0], list):
                    return 1
                return sum(walk(c) for c in x)
            return 0
        return walk(notes)

    async def run(self, index_urls: list[str], limit: int | None = None) -> list[ChartRecord]:
        records: list[ChartRecord] = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                all_chart_urls: list[str] = []
                for idx_url in index_urls:
                    urls = await self.collect_chart_urls(browser, idx_url)
                    all_chart_urls.extend(urls)
                # 重複排除
                all_chart_urls = sorted(set(all_chart_urls))
                if limit is not None:
                    all_chart_urls = all_chart_urls[:limit]
                print(f"[Scraper] 合計 {len(all_chart_urls)} 件の譜面を取得します")

                for i, url in enumerate(all_chart_urls, 1):
                    print(f"[Scraper] ({i}/{len(all_chart_urls)}) {url}")
                    rec = await self.fetch_chart(browser, url)
                    if rec is not None:
                        records.append(rec)
            finally:
                await browser.close()
        return records
