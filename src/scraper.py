"""textage.cc から譜面ページをスクレイピングするモジュール。

textage.cc の実構造を踏まえた実装:

- 一覧ページ (例: ?sA11B000) は JavaScript で大量の `<a>` を生成する。
  実際の譜面ページは `https://textage.cc/score/<dir>/<song>.html?<flags>` 形式。
  ここで `<flags>` の例: ``1AA00`` (1P, ANOTHER, レベルなど) / ``2HA00`` (2P, HYPER) 等。

- 譜面ページのノートデータはグローバル変数として展開される:
  * ``sp`` (シングルプレイの 1P 譜面)
  * ``dp`` (シングルプレイの 2P / ダブルプレイ譜面)
  * ``ln`` (Long Note 持続時間)
  * ``cn`` (Charge Note)
  * ``tc`` (BPM/拍子変化)
  * ``title``, ``artist``, ``genre``, ``bpm`` (文字列), ``diftype`` (難易度ラベル)

  ``sp[i]`` の各要素は 16 桁の hex 文字列 (= 8 byte = 8 tick × 8 lane bitmask)
  になっている。最初の 1～2 要素はメタ情報 (BPM 初期値、拍子等) なので除外する。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from playwright.async_api import async_playwright, Browser, Page


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
    song_id: str               # 例: "0/100seckb_1AA00"
    title: str | None
    artist: str | None
    genre: str | None
    difficulty: str | None     # "ANOTHER" / "HYPER" / "NORMAL" / "BEGINNER" / "LEGGENDARIA"
    play_side: str | None      # "1P" / "2P" / "DP"
    level: int | None
    bpm: str | None
    notes_count: int
    measures: int              # 小節数 (sp 長 - メタオフセット)
    raw_notes: list            # sp または dp の生配列 (hex 文字列 + 数値)
    long_notes: list           # ln 配列
    charge_notes: list         # cn 配列
    tempo: list                # tc 配列

    def to_dict(self) -> dict:
        return asdict(self)


# URL から `<dir>/<song>_<flags>` の song_id を組み立てる
SONG_PATH_RE = re.compile(r"/score/([^/]+)/([^/]+?)\.html\?([A-Za-z0-9]+)")


def parse_song_id(url: str) -> str:
    m = SONG_PATH_RE.search(url)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return re.sub(r"[^A-Za-z0-9_]", "_", url)[-80:]


# 譜面ページから取り出す JS。textage.cc の実グローバルを直接読む。
EXTRACT_CHART_JS = r"""
() => {
    function safe(v) {
        try { JSON.parse(JSON.stringify(v)); return v; }
        catch (e) { return null; }
    }
    return {
        title: typeof title !== 'undefined' ? title : null,
        artist: typeof artist !== 'undefined' ? artist : null,
        genre: typeof genre !== 'undefined' ? genre : null,
        bpm: typeof bpm !== 'undefined' ? bpm : null,
        diftype: typeof diftype !== 'undefined' ? diftype : null,
        s: typeof s !== 'undefined' ? s : null,
        sp: typeof sp !== 'undefined' ? safe(sp) : null,
        dp: typeof dp !== 'undefined' ? safe(dp) : null,
        ln: typeof ln !== 'undefined' ? safe(ln) : null,
        cn: typeof cn !== 'undefined' ? safe(cn) : null,
        tc: typeof tc !== 'undefined' ? safe(tc) : null,
    };
}
"""


# 一覧ページからの URL 取得用フィルタ
INDEX_HTML_RE = re.compile(r"/index\.html\?")
CHART_PATH_RE = re.compile(r"textage\.cc/score/[^/?#]+/[^/?#]+\.html\?")


# diftype 例: "[SP ANOTHER](1P)" / "[DP HYPER]" / "[SP LEGGENDARIA](2P)"
DIFTYPE_RE = re.compile(
    r"\[(SP|DP)\s+(BEGINNER|NORMAL|HYPER|ANOTHER|LEGGENDARIA)\](?:\((\d?P)\))?",
    re.IGNORECASE,
)


def _parse_diftype(diftype: str | None) -> tuple[str | None, str | None]:
    if not diftype:
        return None, None
    m = DIFTYPE_RE.search(diftype)
    if not m:
        return None, None
    style = m.group(1).upper()
    diff = m.group(2).upper()
    side = (m.group(3) or "").upper()
    if style == "DP":
        return diff, "DP"
    return diff, side or "1P"


# URL flags 例: 1AA00, 2HA00, 1HC00 ...
# 2 文字目: B=BEGINNER, N=NORMAL, H=HYPER, A=ANOTHER, L=LEGGENDARIA
# 3 文字目: レベルを base36 で表現したもの (0-9, A-Z) ※簡易仕様
LEVEL_BASE36_RE = re.compile(r"^[12][BNHALbnhal]([0-9A-Za-z])")


def _parse_level_from_flags(flags: str) -> int | None:
    m = LEVEL_BASE36_RE.match(flags)
    if not m:
        return None
    ch = m.group(1)
    try:
        return int(ch, 36)
    except ValueError:
        return None


def _count_hex_bits(s: str) -> int:
    """16 進文字列の中で立っているビット数 = ノート数。"""
    n = 0
    for c in s:
        try:
            n += bin(int(c, 16)).count("1")
        except ValueError:
            continue
    return n


def _is_hex_notes(s) -> bool:
    """sp/dp の要素が "通常のノート列 (hex 文字列)" であるかを判定する。"""
    if not isinstance(s, str):
        return False
    if not s:
        return False
    if any(c in s for c in "@xX"):
        # メタ情報 (BPM/拍子等の特殊行)
        return False
    return all(c in "0123456789abcdefABCDEF" for c in s)


def count_notes(sp_array) -> tuple[int, int]:
    """sp/dp 配列からノート総数と小節数を計算する。"""
    if not isinstance(sp_array, list):
        return 0, 0
    notes = 0
    measures = 0
    for item in sp_array:
        if _is_hex_notes(item):
            notes += _count_hex_bits(item)
            measures += 1
    return notes, measures


class TextageScraper:
    def __init__(
        self,
        cache_dir: Path,
        page_timeout_ms: int = 45000,
        render_wait_ms: int = 4000,
    ) -> None:
        self.cache_dir = cache_dir
        self.charts_dir = cache_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.page_timeout_ms = page_timeout_ms
        self.render_wait_ms = render_wait_ms

    async def _new_page(self, browser: Browser) -> Page:
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(self.page_timeout_ms)
        # 余分なリソース (広告、画像) をブロックして高速化
        await context.route(
            "**/*",
            lambda route: asyncio.create_task(self._maybe_block(route)),
        )
        return page

    @staticmethod
    async def _maybe_block(route) -> None:
        url = route.request.url
        block_keywords = (
            "doubleclick.net",
            "googletagmanager",
            "googlesyndication",
            "google-analytics",
            "adservice.google",
            "recaptcha",
            "googleads.g.doubleclick",
            "pagead2",
            "fundingchoicesmessages",
        )
        if any(k in url for k in block_keywords):
            try:
                await route.abort()
                return
            except Exception:
                pass
        try:
            await route.continue_()
        except Exception:
            pass

    async def collect_chart_urls(self, browser: Browser, index_url: str) -> list[str]:
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
            if CHART_PATH_RE.search(h) and not INDEX_HTML_RE.search(h)
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
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[Scraper] キャッシュ壊れ: {cache_path.name} ({e})")

        page = await self._new_page(browser)
        try:
            try:
                await page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                print(f"[Scraper] アクセス失敗 ({url}): {e}")
                return None
            await page.wait_for_timeout(self.render_wait_ms)

            try:
                payload = await page.evaluate(EXTRACT_CHART_JS)
            except Exception as e:
                print(f"[Scraper] 抽出失敗 ({url}): {e}")
                return None
        finally:
            await page.close()

        # ノート配列の決定 (DP 譜面は dp、それ以外は sp を主とする)
        diff, side = _parse_diftype(payload.get("diftype"))
        sp = payload.get("sp") or []
        dp = payload.get("dp") or []
        if side == "DP" and dp:
            primary_notes = dp
        elif sp:
            primary_notes = sp
        else:
            primary_notes = dp

        notes_count, measures = count_notes(primary_notes)

        # URL のフラグ部分からレベルを推定
        flags_match = re.search(r"\?([A-Za-z0-9]+)$", url)
        level = (
            _parse_level_from_flags(flags_match.group(1)) if flags_match else None
        )

        record = ChartRecord(
            url=url,
            song_id=song_id,
            title=payload.get("title"),
            artist=payload.get("artist"),
            genre=payload.get("genre"),
            difficulty=diff,
            play_side=side,
            level=level,
            bpm=payload.get("bpm"),
            notes_count=notes_count,
            measures=measures,
            raw_notes=primary_notes if isinstance(primary_notes, list) else [],
            long_notes=payload.get("ln") or [],
            charge_notes=payload.get("cn") or [],
            tempo=payload.get("tc") or [],
        )
        cache_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        return record

    async def fetch_one(self, url: str) -> ChartRecord | None:
        """単一の譜面 URL を取得する (予測用)。"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                return await self.fetch_chart(browser, url)
            finally:
                await browser.close()

    async def run(
        self, index_urls: list[str], limit: int | None = None
    ) -> list[ChartRecord]:
        records: list[ChartRecord] = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                all_chart_urls: list[str] = []
                for idx_url in index_urls:
                    urls = await self.collect_chart_urls(browser, idx_url)
                    all_chart_urls.extend(urls)
                all_chart_urls = sorted(set(all_chart_urls))
                if limit is not None:
                    all_chart_urls = all_chart_urls[:limit]
                print(f"[Scraper] 合計 {len(all_chart_urls)} 件の譜面を取得します")

                for i, url in enumerate(all_chart_urls, 1):
                    rec = await self.fetch_chart(browser, url)
                    if rec is not None:
                        records.append(rec)
                    if i % 100 == 0:
                        print(
                            f"[Scraper] 進捗 {i}/{len(all_chart_urls)} "
                            f"(取得済 {len(records)} 件)"
                        )
            finally:
                await browser.close()
        return records
