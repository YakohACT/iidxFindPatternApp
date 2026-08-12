"""textage.cc から譜面ページをスクレイピングするモジュール。

textage.cc の実構造 (レンダラ bms2jsh.js の解析結果):

- 一覧ページ (例: ?sA11B000) は JavaScript で大量の `<a>` を生成する。
  実際の譜面ページは `https://textage.cc/score/<dir>/<song>.html?<flags>` 形式。
  `<flags>` 例: ``1AA00`` (1P, ANOTHER, レベル 10) / ``1XC00`` (1P, LEGGENDARIA, レベル 12)。

- 譜面ページのノートデータはグローバル変数として展開される:
  * ``sp`` / ``dp`` : 小節ごとのノート行 (plain hex / 'x' プレフィクス /
    '#' 圧縮の 3 形式。詳細は textage_notes モジュール参照)
  * ``ln`` : 各小節の長さ (384 = 4/4 拍子)。ロングノートではない点に注意
  * ``cn`` : チャージノート (CN / HCN / BSS)
  * ``tc`` : BPM 変化
  * ``title``, ``artist``, ``genre``, ``bpm``, ``diftype``

サーバ負荷対策として、実際にページへアクセスする直前に RateLimiter で
1 時間あたりのアクセス数を制限する (状態は data/cache/rate_limit_state.json
に永続化され、プロセスをまたいで有効)。キャッシュ命中時はアクセスしない。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

from textage_notes import count_notes

if TYPE_CHECKING:  # playwright は実際にスクレイピングする時のみ必要
    from playwright.async_api import Browser, Page


INDEX_URLS = [
    "https://textage.cc/score/index.html?sA11B000",
    "https://textage.cc/score/index.html?sB11B000",
    "https://textage.cc/score/index.html?sC11B000",
    "https://textage.cc/score/index.html?sX11B000",
]

DEFAULT_REQUESTS_PER_HOUR = 720  # 平均 5 秒間隔


class RateLimiter:
    """1 時間あたりのアクセス数を制限する sliding-window リミッタ。

    アクセス時刻のリストを state_file に保存し、プロセスをまたいで
    制限を守る。均等ペース (min_interval) と時間窓の上限を両方適用する。
    """

    def __init__(
        self,
        state_file: Path,
        requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR,
        window_s: float = 3600.0,
    ) -> None:
        self.state_file = state_file
        self.rph = max(1, requests_per_hour)
        self.window_s = window_s
        self.min_interval = window_s / self.rph
        self._stamps: list[float] = self._load()

    def _load(self) -> list[float]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        now = time.time()
        return sorted(
            t for t in data
            if isinstance(t, (int, float)) and 0 < now - t < self.window_s
        )

    def _save(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self._stamps), encoding="utf-8")
        except OSError as e:
            print(f"[RateLimiter] 状態保存に失敗: {e}")

    async def acquire(self) -> None:
        """アクセス枠を 1 つ取得する。制限中は解放まで待つ。"""
        while True:
            now = time.time()
            self._stamps = [t for t in self._stamps if now - t < self.window_s]
            wait = 0.0
            if self._stamps:
                wait = max(wait, self._stamps[-1] + self.min_interval - now)
            if len(self._stamps) >= self.rph:
                wait = max(wait, self._stamps[0] + self.window_s - now)
            if wait <= 0:
                break
            if wait > 30:
                print(f"[RateLimiter] アクセス制限 ({self.rph}/h) により {wait:.0f} 秒待機します")
            await asyncio.sleep(wait)
        self._stamps.append(time.time())
        self._save()


@dataclass
class ChartRecord:
    """1 譜面分の生データ。"""

    url: str
    song_id: str               # 例: "0_100seckb_1AA00"
    title: str | None
    artist: str | None
    genre: str | None
    difficulty: str | None     # "ANOTHER" / "HYPER" / "NORMAL" / "BEGINNER" / "LEGGENDARIA"
    play_side: str | None      # "1P" / "2P" / "DP"
    level: int | None
    bpm: str | None
    notes_count: int           # CN/BSS 開始を含む総ノート数
    measures: int              # 小節数 (= sp/dp の行数)
    raw_notes: list            # sp または dp の生配列
    long_notes: list           # ln 配列 (= 各小節の長さ。384 が 4/4)
    charge_notes: list         # cn 配列 (= [[], c1, c2] 形式)
    tempo: list                # tc 配列

    def to_dict(self) -> dict:
        return asdict(self)


# URL から `<dir>_<song>_<flags>` の song_id を組み立てる
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


# URL flags 例: 1AA00, 2HA00, 1XC00 ...
# 1 文字目: 1=1P, 2=2P
# 2 文字目: B=BEGINNER, N=NORMAL, H=HYPER, A=ANOTHER, X(またはL)=LEGGENDARIA
# 3 文字目: レベルを base36 で表現したもの (A=10, B=11, C=12)。0 は未設定
LEVEL_FLAG_RE = re.compile(r"^[12][BNHALXbnhalx]([0-9A-Za-z])")


def parse_level_from_flags(flags: str) -> int | None:
    """URL フラグ (例: "1AA00", "1XC00") からレベルを取得する。不明なら None。"""
    m = LEVEL_FLAG_RE.match(flags)
    if not m:
        return None
    try:
        level = int(m.group(1), 36)
    except ValueError:
        return None
    return level or None  # 0 はレベル情報なし


class TextageScraper:
    def __init__(
        self,
        cache_dir: Path,
        page_timeout_ms: int = 45000,
        render_wait_ms: int = 4000,
        requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR,
    ) -> None:
        self.cache_dir = cache_dir
        self.charts_dir = cache_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.page_timeout_ms = page_timeout_ms
        self.render_wait_ms = render_wait_ms
        self.limiter = RateLimiter(
            cache_dir / "rate_limit_state.json", requests_per_hour
        )

    def _read_cache(self, song_id: str) -> ChartRecord | None:
        cache_path = self.charts_dir / f"{song_id}.json"
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return ChartRecord(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[Scraper] キャッシュ壊れ: {cache_path.name} ({e})")
            return None

    async def _new_page(self, browser: "Browser") -> "Page":
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(self.page_timeout_ms)
        # 余分なリソース (広告等) をブロックして高速化
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

    async def collect_chart_urls(self, browser: "Browser", index_url: str) -> list[str]:
        await self.limiter.acquire()
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

    async def fetch_chart(self, browser: "Browser", url: str) -> ChartRecord | None:
        song_id = parse_song_id(url)
        cached = self._read_cache(song_id)
        if cached is not None:
            return cached

        await self.limiter.acquire()
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
        if not isinstance(primary_notes, list):
            primary_notes = []

        long_notes = payload.get("ln") or []
        charge_notes = payload.get("cn") or []
        notes_count, measures = count_notes(
            primary_notes, long_notes, charge_notes
        )

        # URL のフラグ部分からレベルを取得
        flags_match = re.search(r"\?([A-Za-z0-9]+)$", url)
        level = (
            parse_level_from_flags(flags_match.group(1)) if flags_match else None
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
            raw_notes=primary_notes,
            long_notes=long_notes,
            charge_notes=charge_notes,
            tempo=payload.get("tc") or [],
        )
        cache_path = self.charts_dir / f"{song_id}.json"
        cache_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        return record

    @staticmethod
    def _import_playwright():
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print(
                "[Scraper] playwright がインストールされていません。"
                "`pip install playwright && python -m playwright install chromium` "
                "を実行してください (キャッシュ済みの譜面はそのまま利用できます)"
            )
            return None
        return async_playwright

    async def fetch_one(self, url: str) -> ChartRecord | None:
        """単一の譜面 URL を取得する (予測用)。キャッシュ命中時はブラウザを起動しない。"""
        cached = self._read_cache(parse_song_id(url))
        if cached is not None:
            return cached

        async_playwright = self._import_playwright()
        if async_playwright is None:
            return None

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                return await self.fetch_chart(browser, url)
            finally:
                await browser.close()

    async def run(
        self, index_urls: list[str], limit: int | None = None
    ) -> list[ChartRecord]:
        async_playwright = self._import_playwright()
        if async_playwright is None:
            return []

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
