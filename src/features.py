"""譜面の生データから機械学習向けの特徴量ベクトルを生成するモジュール。

textage.cc の譜面ノートデータ ``sp`` / ``dp`` は次の構造になっている:

    sp = [
        null,                    # 0 番目はメタ用
        "x04008@0510@04...",     # 初期 BPM / 拍子等 (メタ行: 'x' や '@' を含む)
        "01",                    # 短いメタ
        "0000000000000008",      # 16 桁 hex = 8 byte = 1 小節
        "8100000204400010",      # 同上
        ...
    ]

16 桁 hex の各バイト (= 2 hex 文字) を「1 小節中の 1/8 拍 (= 1 tick)」とみなし、
各バイトの 8 ビットを 8 レーン (SP の場合: 1=スクラッチ + 7=鍵盤) のビットマスク
として扱う。これにより、

  byte_index (0..7) × bit_index (0..7) = (tick, lane)

の (時刻, レーン) のペアにフラット化できる。

抽出する特徴量:
  - total_notes: 総ノート数
  - measures:    小節数
  - density:     ノート数 / 小節
  - lane0..6_ratio: 鍵盤 1〜7 ごとのノート比率
  - scratch_ratio:  スクラッチ比率
  - mean_chord:  同時押し平均サイズ
  - max_chord:   同時押し最大サイズ
  - peak_density: 連続 4 tick の最大ノート数
  - stair_ratio: 隣接レーンへの連続移動率
  - trill_ratio: 2 レーン間の往復率
  - has_long_notes: ロングノート/CN を含むか (0/1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


N_LANES = 8        # IIDX SP: 7 keys + 1 scratch
TICKS_PER_BAR = 8  # 16 hex chars = 8 bytes = 8 ticks/小節


FEATURE_NAMES = [
    "total_notes",
    "measures",
    "density",
    "lane0_ratio",
    "lane1_ratio",
    "lane2_ratio",
    "lane3_ratio",
    "lane4_ratio",
    "lane5_ratio",
    "lane6_ratio",
    "scratch_ratio",
    "mean_chord",
    "max_chord",
    "peak_density",
    "stair_ratio",
    "trill_ratio",
    "has_long_notes",
]


@dataclass
class FeatureVector:
    song_id: str
    title: str | None
    level: int | None
    difficulty: str | None
    values: np.ndarray
    feature_names: list[str]


def _is_hex_row(s) -> bool:
    if not isinstance(s, str) or not s:
        return False
    if any(c in s for c in "@xX"):
        return False
    return all(c in "0123456789abcdefABCDEF" for c in s)


def parse_notes(raw_notes) -> list[tuple[int, int]]:
    """sp / dp 配列を ``[(global_tick, lane), ...]`` の昇順リストへ展開する。

    各バイトの bit_i は **i 番目のレーン** に対応すると仮定する (lane 0..7)。
    レーンの物理的な意味付け (1 鍵 / スクラッチ等) は textage 内部実装に依存
    するため、特徴量抽出では「あるレーン番号のノート数」として均一に扱う。
    """
    notes: list[tuple[int, int]] = []
    if not isinstance(raw_notes, list):
        return notes

    bar_idx = 0
    for item in raw_notes:
        if not _is_hex_row(item):
            continue
        # 16 桁を 8 バイトに区切る (足りない場合は 0 詰め)
        s = item.zfill(16)[-16:]
        for tick, two in enumerate([s[i:i + 2] for i in range(0, 16, 2)]):
            try:
                byte_val = int(two, 16)
            except ValueError:
                continue
            if byte_val == 0:
                continue
            for lane in range(N_LANES):
                if byte_val & (1 << lane):
                    notes.append((bar_idx * TICKS_PER_BAR + tick, lane))
        bar_idx += 1
    notes.sort()
    return notes


def _chord_sizes(notes: list[tuple[int, int]]) -> list[int]:
    if not notes:
        return []
    sizes: list[int] = []
    cur_tick = notes[0][0]
    cur_lanes = {notes[0][1]}
    for tick, lane in notes[1:]:
        if tick == cur_tick:
            cur_lanes.add(lane)
        else:
            sizes.append(len(cur_lanes))
            cur_tick = tick
            cur_lanes = {lane}
    sizes.append(len(cur_lanes))
    return sizes


def _peak_density(notes: list[tuple[int, int]], window: int = 4) -> int:
    """連続 `window` tick 中の最大ノート数 (滑走窓)。"""
    if not notes:
        return 0
    ticks = [t for t, _ in notes]
    max_count = 0
    left = 0
    for right in range(len(ticks)):
        while ticks[right] - ticks[left] > window:
            left += 1
        if right - left + 1 > max_count:
            max_count = right - left + 1
    return max_count


def _stair_trill_ratios(notes: list[tuple[int, int]]) -> tuple[float, float]:
    """鍵盤 (lane != 7) のみの時系列で階段率とトリル率を計算する。

    textage のレーンマッピングは不明だが、隣り合うレーン番号への遷移を
    階段、a→b→a の遷移をトリルと近似する。
    """
    keys = [(t, l) for t, l in notes if l != 7]
    if len(keys) < 2:
        return 0.0, 0.0
    stair = 0
    for i in range(1, len(keys)):
        if abs(keys[i][1] - keys[i - 1][1]) == 1:
            stair += 1
    trill = 0
    for i in range(2, len(keys)):
        a, b, c = keys[i - 2][1], keys[i - 1][1], keys[i][1]
        if a == c and a != b:
            trill += 1
    return stair / max(len(keys) - 1, 1), trill / max(len(keys) - 2, 1)


def _has_long_notes(long_notes, charge_notes) -> int:
    def any_nonzero(seq):
        if not isinstance(seq, list):
            return False
        for x in seq:
            if x is None or x == 0:
                continue
            if isinstance(x, list) and not x:
                continue
            return True
        return False
    return 1 if any_nonzero(long_notes) or any_nonzero(charge_notes) else 0


def extract_features(
    song_id: str,
    raw_notes,
    title: str | None = None,
    level: int | None = None,
    difficulty: str | None = None,
    long_notes=None,
    charge_notes=None,
) -> FeatureVector:
    notes = parse_notes(raw_notes)
    total_notes = len(notes)

    if total_notes == 0:
        return FeatureVector(
            song_id=song_id,
            title=title,
            level=level,
            difficulty=difficulty,
            values=np.zeros(len(FEATURE_NAMES), dtype=float),
            feature_names=list(FEATURE_NAMES),
        )

    # 小節数推定: 最後の tick / TICKS_PER_BAR + 1
    measures = (notes[-1][0] // TICKS_PER_BAR) + 1
    density = total_notes / max(measures, 1)

    # レーン別カウント
    lane_counts = np.zeros(N_LANES, dtype=float)
    for _, lane in notes:
        lane_counts[lane] += 1
    lane_ratios = lane_counts / total_notes

    chord_sizes = _chord_sizes(notes)
    mean_chord = float(np.mean(chord_sizes))
    max_chord = float(np.max(chord_sizes))

    peak_density = _peak_density(notes, window=4)
    stair_ratio, trill_ratio = _stair_trill_ratios(notes)

    values = np.array([
        total_notes,
        measures,
        density,
        lane_ratios[0],
        lane_ratios[1],
        lane_ratios[2],
        lane_ratios[3],
        lane_ratios[4],
        lane_ratios[5],
        lane_ratios[6],
        lane_ratios[7],   # scratch_ratio
        mean_chord,
        max_chord,
        peak_density,
        stair_ratio,
        trill_ratio,
        _has_long_notes(long_notes, charge_notes),
    ], dtype=float)

    return FeatureVector(
        song_id=song_id,
        title=title,
        level=level,
        difficulty=difficulty,
        values=values,
        feature_names=list(FEATURE_NAMES),
    )


def stack(features: Iterable[FeatureVector]) -> tuple[np.ndarray, list[FeatureVector]]:
    items = list(features)
    if not items:
        return np.zeros((0, len(FEATURE_NAMES))), []
    matrix = np.vstack([f.values for f in items])
    return matrix, items
