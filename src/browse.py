"""対話式の譜面ブラウザ。

`python src/main.py --browse` で起動し、

  1. 曲名 (50 音順) から選ぶ
     あ〜わ行 / A-Z / 数字 / 漢字・他 のグループ → 曲 → 難易度 → 単体情報
  2. レベル (1-12) から選ぶ
     レベル → 50 音順の譜面一覧 → 単体情報

の 2 経路で、学習済みデータ (data/results/assignments.csv) にある譜面の
単体情報 (基本情報 / クラスタ / 32 特徴量の z-score / RANDOM オプション判断)
を表示する。

50 音順について:
  - かな始まりの曲名はカタカナ→ひらがな正規化により正確な 50 音順で並ぶ
  - 英数字はアルファベット順 / 数字順
  - 漢字始まりの曲名は読みデータが存在しないため「漢字・他」グループに
    文字コード順でまとめる (textage 側にも読み仮名データは無い)
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from ml import load_model


PAGE_SIZE = 20

DIFF_ORDER = {"BEGINNER": 0, "NORMAL": 1, "HYPER": 2, "ANOTHER": 3, "LEGGENDARIA": 4}
DIFF_SHORT = {"BEGINNER": "B", "NORMAL": "N", "HYPER": "H", "ANOTHER": "A", "LEGGENDARIA": "L"}

# 特徴量の表示グループ (特徴量名, 表示ラベル, 表示形式)
FEATURE_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("基本量", [
        ("total_notes", "総ノート数", "int"),
        ("measures", "小節数", "int"),
        ("density", "密度 (ノート/小節)", "f1"),
        ("scratch_ratio", "スクラッチ比率", "pct"),
        ("mean_chord", "同時押し平均", "f2"),
        ("max_chord", "同時押し最大", "int"),
        ("peak_density", "瞬間密度 (2拍窓)", "int"),
        ("stream16_ratio", "16分乱打率", "pct"),
    ]),
    ("階段系", [
        ("stair_mono_ratio", "単階段", "pct"),
        ("stair_turn_ratio", "折り返し階段", "pct"),
        ("big_stair_rate", "大階段 (回/小節)", "f3"),
        ("double_stair_ratio", "二重階段", "pct"),
        ("garbage_stair_ratio", "ゴミ付き階段", "pct"),
    ]),
    ("リズム・同時押し系", [
        ("denim_ratio", "デニム", "pct"),
        ("chord_trill_ratio", "二重トリル", "pct"),
        ("trill_ratio", "トリル", "pct"),
        ("jack_ratio", "縦連", "pct"),
        ("wall_ratio", "壁", "pct"),
        ("axis_ratio", "軸", "pct"),
    ]),
    ("スクラッチ系", [
        ("scratch_stream_ratio", "連皿", "pct"),
        ("scratch_key_mix_ratio", "皿複合", "pct"),
        ("scratch_left_chord_ratio", "無理皿 (皿+1,2鍵)", "pct"),
        ("mss_rate", "MSS (本/小節)", "f3"),
    ]),
    ("その他", [
        ("peak_position", "最密地帯の位置 (1=ラスト)", "f2"),
        ("offgrid_ratio", "ズレ/裏拍", "pct"),
        ("hand_bias", "片手偏重", "f2"),
        ("cn_ratio", "CN/HCN/BSS", "pct"),
        ("bpm_range_log2", "ソフラン幅 (log2)", "f2"),
        ("bpm_change_rate", "ソフラン頻度 (回/小節)", "f3"),
    ]),
]

RANDOM_FEATURES = [
    ("random_advantage", "乱推奨度 (正規のハズレ度)"),
    ("random_gamble", "乱ガチャ度 (振れ幅)"),
    ("mirror_advantage", "鏡推奨度"),
]

META_COLUMNS = {
    "song_id", "title", "level", "difficulty", "cluster", "cluster_name",
    "pca_x", "pca_y",
}

KANA_ROWS = [
    ("あ行", "あいうえお"),
    ("か行", "かきくけこ"),
    ("さ行", "さしすせそ"),
    ("た行", "たちつてと"),
    ("な行", "なにぬねの"),
    ("は行", "はひふへほ"),
    ("ま行", "まみむめも"),
    ("や行", "やゆよ"),
    ("ら行", "らりるれろ"),
    ("わ行", "わをん"),
]

_DAKUTEN_BASE = str.maketrans(
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゔ",
    "かきくけこさしすせそたちつてとはひふへほはひふへほう",
)
_SMALL_BASE = str.maketrans(
    "ぁぃぅぇぉっゃゅょゎ",
    "あいうえおつやゆよわ",
)


def clean_title(raw) -> str:
    """キャッシュ由来のタイトルから HTML タグ・実体参照を除去する。"""
    s = html.unescape(str(raw))
    s = re.sub(r"<[^>]*>", "", s)
    return s.strip() or "(無題)"


def sort_key(title: str) -> str:
    """50 音順ソート用キー (NFKC 正規化 + カタカナ→ひらがな + 小文字化)。"""
    s = unicodedata.normalize("NFKC", title).casefold()
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:  # カタカナ → ひらがな
            ch = chr(code - 0x60)
        out.append(ch)
    return "".join(out)


def _kana_row_index(ch: str) -> int | None:
    ch = ch.translate(_DAKUTEN_BASE).translate(_SMALL_BASE)
    for i, (_name, chars) in enumerate(KANA_ROWS):
        if ch in chars:
            return i
    return None


def title_bucket(title: str) -> tuple[str, str]:
    """曲名を (バケット種別, サブキー) に分類する。

    種別: 'kana' (サブキー=行名) / 'alpha' (A-Z) / 'num' / 'other'
    """
    key = sort_key(title)
    if not key:
        return "other", ""
    ch = key[0]
    if "ぁ" <= ch <= "ゖ" or ch == "ー":
        row = _kana_row_index(ch)
        if row is not None:
            return "kana", KANA_ROWS[row][0]
        return "other", ""
    if "a" <= ch <= "z":
        return "alpha", ch.upper()
    if "0" <= ch <= "9":
        return "num", ""
    if "一" <= ch <= "鿿":
        return "other", ""
    # 記号始まりは記号を読み飛ばして再判定 (例: "†渚の小悪魔..." )
    for i in range(1, len(key)):
        c = key[i]
        if c.isalnum() or "ぁ" <= c <= "ゖ" or "一" <= c <= "鿿":
            return title_bucket(key[i:])
    return "other", ""


class ChartIndex:
    """assignments.csv を読み込んだ譜面インデックス。"""

    def __init__(self, results_dir: Path, cache_dir: Path, model_dir: Path):
        self.cache_dir = cache_dir
        path = results_dir / "assignments.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} が見つかりません。先に学習を実行してください "
                "(例: python src/main.py --skip-scrape --min-k 10)"
            )
        df = pd.read_csv(path)
        df["display_title"] = df["title"].map(clean_title)
        df["sort_key"] = df["display_title"].map(sort_key)
        df["diff_order"] = df["difficulty"].map(lambda d: DIFF_ORDER.get(str(d), 9))
        self.df = df.sort_values(["sort_key", "diff_order"]).reset_index(drop=True)

        self.feature_cols = [c for c in df.columns
                             if c not in META_COLUMNS
                             and c not in {"display_title", "sort_key", "diff_order"}]
        feats = self.df[self.feature_cols].astype(float)

        # z-score 用統計: モデル保存値があれば優先、無ければ全譜面から計算
        model = load_model(model_dir)
        if model is not None and model.feature_mean is not None:
            self.mean = model.feature_mean.reindex(self.feature_cols)
            self.std = model.feature_std.reindex(self.feature_cols).replace(0, 1.0)
            self.mean = self.mean.fillna(feats.mean())
            self.std = self.std.fillna(feats.std(ddof=0)).replace(0, 1.0)
        else:
            self.mean = feats.mean()
            self.std = feats.std(ddof=0).replace(0, 1.0)

    def load_meta(self, song_id: str) -> dict:
        path = self.cache_dir / "charts" / f"{song_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            "artist": data.get("artist"),
            "genre": data.get("genre"),
            "bpm": data.get("bpm"),
            "url": data.get("url"),
            "play_side": data.get("play_side"),
        }


# ---------------------------------------------------------------- 表示部品

def _fmt(value: float, kind: str) -> str:
    if kind == "int":
        return f"{value:8.0f}"
    if kind == "f1":
        return f"{value:8.1f}"
    if kind == "f2":
        return f"{value:8.2f}"
    if kind == "f3":
        return f"{value:8.3f}"
    if kind == "pct":
        return f"{100 * value:7.1f}%"
    return f"{value:8.3f}"


def _z_marker(z: float) -> str:
    if z >= 2.0:
        return "▲▲"
    if z >= 1.0:
        return "▲ "
    if z <= -2.0:
        return "▼▼"
    if z <= -1.0:
        return "▼ "
    return "  "


def _level_str(level) -> str:
    if pd.isna(level):
        return "Lv??"
    return f"Lv{int(level):>2}"


def render_detail(index: ChartIndex, row: pd.Series) -> None:
    meta = index.load_meta(row["song_id"])
    title = row["display_title"]
    line = "=" * 64
    print()
    print(line)
    artist = clean_title(meta.get("artist")) if meta.get("artist") else "?"
    genre = clean_title(meta.get("genre")) if meta.get("genre") else "?"
    print(f" {title} / {artist}")
    print(f" GENRE: {genre}   BPM: {meta.get('bpm') or '?'}")
    print(f" {row['difficulty']} {_level_str(row['level'])}   song_id: {row['song_id']}")
    if meta.get("url"):
        print(f" URL: {meta['url']}")
    print("-" * 64)
    print(f" クラスタ {int(row['cluster'])}: {row['cluster_name']}")
    print("-" * 64)
    print(" 特徴量                              値      z   (▲=全体平均+1σ以上)")

    def z_of(name: str) -> float:
        return (float(row[name]) - float(index.mean[name])) / float(index.std[name])

    for group_name, items in FEATURE_GROUPS:
        print(f" [{group_name}]")
        for name, label, kind in items:
            if name not in row:
                continue
            z = z_of(name)
            print(f"   {label:<24s}{_fmt(float(row[name]), kind)}  {z:+5.1f} {_z_marker(z)}")

    print(" [RANDOM オプション判断]")
    for name, label in RANDOM_FEATURES:
        if name not in row:
            continue
        z = z_of(name)
        print(f"   {label:<24s}{float(row[name]):8.2f}  {z:+5.1f} {_z_marker(z)}")

    adv = float(row.get("random_advantage", 0.0))
    gam_z = z_of("random_gamble") if "random_gamble" in row else 0.0
    mir = float(row.get("mirror_advantage", 0.0))
    jack_z = z_of("jack_ratio") if "jack_ratio" in row else 0.0
    scr_z = z_of("scratch_stream_ratio") if "scratch_stream_ratio" in row else 0.0

    verdict = []
    if adv >= 1.0:
        verdict.append("正規はハズレ配置寄り → 乱を掛ける価値あり")
    elif adv <= -1.0:
        verdict.append("正規が当たり配置 → 正規/鏡向き")
    else:
        verdict.append("正規とランダム平均に大差なし")
    if mir >= 1.0:
        verdict.append("鏡も有力")
    if gam_z >= 0.6:
        verdict.append("当たり外れが激しい (粘着ガチャ性)")
    if jack_z >= 1.0:
        verdict.append("縦連は乱で消えない → S乱検討")
    if scr_z >= 1.0:
        verdict.append("連皿は乱の影響を受けない")
    print("   → " + " / ".join(verdict))
    print(line)


# ---------------------------------------------------------------- 対話部品

def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def pick_number(prompt: str, n_max: int) -> int | str:
    """数値または 'n'/'p'/'b'/'q' を受け取る。"""
    ans = _input(prompt)
    if ans.lower() in {"n", "p", "b", "q"}:
        return ans.lower()
    try:
        v = int(ans)
    except ValueError:
        return "?"
    if 1 <= v <= n_max:
        return v
    return "?"


def paginate_and_pick(rows: pd.DataFrame, header: str, render_row) -> pd.Series | None:
    """一覧をページ送りしながら 1 行選ばせる。None = 戻る。"""
    if rows.empty:
        print("  (該当する譜面がありません)")
        return None
    page = 0
    n_pages = (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE
    while True:
        page = max(0, min(page, n_pages - 1))
        start = page * PAGE_SIZE
        chunk = rows.iloc[start:start + PAGE_SIZE]
        print(f"\n--- {header}  ({len(rows)} 件 / {page + 1}/{n_pages} ページ) ---")
        for i, (_, r) in enumerate(chunk.iterrows(), start=start + 1):
            print(f"  [{i:>3}] {render_row(r)}")
        ans = pick_number("番号で選択 / [n]次頁 [p]前頁 [b]戻る [q]終了 > ", len(rows))
        if ans == "n":
            page += 1
        elif ans == "p":
            page -= 1
        elif ans == "b":
            return None
        elif ans == "q":
            raise KeyboardInterrupt
        elif isinstance(ans, int):
            return rows.iloc[ans - 1]
        else:
            print("  入力を認識できませんでした")


def _chart_row_label(r: pd.Series) -> str:
    d = DIFF_SHORT.get(str(r["difficulty"]), "?")
    notes = f"{int(r['total_notes']):>4}" if "total_notes" in r else "   ?"
    return (f"{_level_str(r['level'])} [{d}] {r['display_title'][:38]:<38} "
            f"notes={notes}  {r['cluster_name']}")


def pick_difficulty(index: ChartIndex, charts: pd.DataFrame) -> None:
    """同一曲名の難易度一覧から 1 つ選んで詳細表示する。"""
    charts = charts.sort_values(["diff_order", "level"])
    while True:
        print(f"\n--- {charts.iloc[0]['display_title']} の譜面 ---")
        for i, (_, r) in enumerate(charts.iterrows(), start=1):
            notes = int(r["total_notes"]) if "total_notes" in r else 0
            print(f"  [{i}] {r['difficulty']:<11} {_level_str(r['level'])}  "
                  f"notes={notes:<5} {r['cluster_name']}")
        ans = pick_number("番号で選択 / [b]戻る [q]終了 > ", len(charts))
        if ans == "b":
            return
        if ans == "q":
            raise KeyboardInterrupt
        if isinstance(ans, int):
            render_detail(index, charts.iloc[ans - 1])
            _input("Enter で一覧へ戻る > ")


def browse_by_title(index: ChartIndex) -> None:
    df = index.df
    buckets = df["display_title"].map(title_bucket)
    df = df.assign(_bkind=[b[0] for b in buckets], _bsub=[b[1] for b in buckets])

    while True:
        # グループメニュー構築
        menu: list[tuple[str, str, str]] = []  # (kind, sub, 表示名)
        for row_name, _chars in KANA_ROWS:
            n = len(df[(df["_bkind"] == "kana") & (df["_bsub"] == row_name)]
                    ["display_title"].unique())
            if n:
                menu.append(("kana", row_name, f"{row_name} ({n}曲)"))
        n_alpha = len(df[df["_bkind"] == "alpha"]["display_title"].unique())
        n_num = len(df[df["_bkind"] == "num"]["display_title"].unique())
        n_other = len(df[df["_bkind"] == "other"]["display_title"].unique())
        if n_alpha:
            menu.append(("alpha", "", f"A-Z ({n_alpha}曲)"))
        if n_num:
            menu.append(("num", "", f"数字 ({n_num}曲)"))
        if n_other:
            menu.append(("other", "", f"漢字・他 ({n_other}曲) ※読み不明のため文字コード順"))

        print("\n--- 曲名 (50音順) : グループを選択 ---")
        for i, (_k, _s, label) in enumerate(menu, start=1):
            print(f"  [{i:>2}] {label}")
        ans = pick_number("番号で選択 / [b]戻る [q]終了 > ", len(menu))
        if ans == "b":
            return
        if ans == "q":
            raise KeyboardInterrupt
        if not isinstance(ans, int):
            continue
        kind, sub, _label = menu[ans - 1]

        sel = df[df["_bkind"] == kind]
        if kind == "kana":
            sel = sel[sel["_bsub"] == sub]
        elif kind == "alpha":
            # 頭文字メニュー ('b'/'q' は曲名頭文字と衝突するため空 Enter で戻る)
            letters = sorted(sel["_bsub"].unique())
            print("\n--- A-Z: 頭文字を選択 ---")
            counts = {
                le: len(sel[sel["_bsub"] == le]["display_title"].unique())
                for le in letters
            }
            print("  " + "  ".join(f"{le}({counts[le]})" for le in letters))
            ch = _input("頭文字を入力 (空 Enter で戻る) > ").upper()
            if not ch:
                continue
            if ch not in letters:
                print("  該当する頭文字がありません")
                continue
            sel = sel[sel["_bsub"] == ch]

        # 曲名一覧 (ユニークタイトル)
        titles = (sel.groupby("display_title", sort=False)
                  .agg(n=("song_id", "size"), sort_key=("sort_key", "first"))
                  .sort_values("sort_key").reset_index())
        picked = paginate_and_pick(
            titles, "曲を選択",
            lambda r: f"{r['display_title'][:44]:<44} ({r['n']}譜面)",
        )
        if picked is None:
            continue
        charts = df[df["display_title"] == picked["display_title"]]
        pick_difficulty(index, charts)


def browse_by_level(index: ChartIndex) -> None:
    df = index.df
    while True:
        counts = {
            lv: len(df[df["level"] == lv]) for lv in range(1, 13)
        }
        n_unknown = int(df["level"].isna().sum())
        print("\n--- レベルを選択 (収録数) ---")
        line = "  ".join(
            f"[{lv}] ☆{lv}({counts[lv]})" for lv in range(1, 13) if counts[lv]
        )
        empty = [str(lv) for lv in range(1, 13) if not counts[lv]]
        print("  " + line)
        if empty:
            print(f"  (収録なし: ☆{', ☆'.join(empty)})")
        if n_unknown:
            print(f"  [0] レベル不明 ({n_unknown})")
        ans = _input("レベルを入力 / [b]戻る [q]終了 > ").lower()
        if ans == "b":
            return
        if ans == "q":
            raise KeyboardInterrupt
        try:
            lv = int(ans)
        except ValueError:
            print("  入力を認識できませんでした")
            continue
        if lv == 0 and n_unknown:
            sel = df[df["level"].isna()]
            header = "レベル不明の譜面 (50音順)"
        elif 1 <= lv <= 12 and counts.get(lv):
            sel = df[df["level"] == lv]
            header = f"☆{lv} の譜面 (50音順)"
        else:
            print("  そのレベルの譜面はありません")
            continue
        picked = paginate_and_pick(
            sel.sort_values(["sort_key", "diff_order"]), header, _chart_row_label
        )
        if picked is not None:
            render_detail(index, picked)
            _input("Enter で一覧へ戻る > ")


def run_browser(results_dir: Path, cache_dir: Path, model_dir: Path) -> int:
    try:
        index = ChartIndex(results_dir, cache_dir, model_dir)
    except FileNotFoundError as e:
        print(f"[browse] {e}")
        return 1

    n_charts = len(index.df)
    n_titles = len(index.df["display_title"].unique())
    print(f"\n==== IIDX 譜面ブラウザ ({n_titles} 曲 / {n_charts} 譜面) ====")
    try:
        while True:
            print("\n[1] 曲名 (50音順) から選ぶ")
            print("[2] レベル (1-12) から選ぶ")
            print("[q] 終了")
            ans = _input("> ").lower()
            if ans == "1":
                browse_by_title(index)
            elif ans == "2":
                browse_by_level(index)
            elif ans in {"q", "b"}:
                break
            else:
                print("  1 / 2 / q を入力してください")
    except KeyboardInterrupt:
        pass
    print("終了します")
    return 0
