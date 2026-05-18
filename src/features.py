"""譜面の生データから機械学習向けの特徴量ベクトルを生成するモジュール。

textage.cc の `notes` 内部表現はバージョンによって複数の形式があるため、
本モジュールでは以下の 2 段階で特徴量を作る:

1. 生データを `(tick, lane)` のフラットなノートリストに正規化する。
2. 正規化済みノート列から、譜面の傾向を表す統計量を計算する。

IIDX は 7 つの鍵盤 (lane 0-6) と 1 つのスクラッチ (lane 7) で構成される。
特徴量は次の通り:

- ノート数
- ユニークなレーン数
- 同時押しの平均/最大密度
- レーン別ノート比率 (7 鍵 + スクラッチ)
- ピーク密度 (1 秒相当窓のノート数最大)
- 階段らしさ (隣接レーンへの連続移動率)
- トリルらしさ (2 レーン間往復率)
- スクラッチ比率
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


N_LANES = 8  # 7 鍵 + スクラッチ


def _flatten_notes(raw_notes) -> list[tuple[int, int]]:
    """textage.cc の階層型 notes を [(tick, lane), ...] に正規化する。

    入力は以下のいずれかを想定:
    - [[tick, lane, ...], ...]            (フラット)
    - [[ [tick, lane, ...], ... ], ...]   (小節毎にネスト)
    - 数値だけが並ぶ配列 (Run-length など)
    """
    out: list[tuple[int, int]] = []

    def visit(item, depth: int = 0) -> None:
        if not isinstance(item, list):
            return
        # 末端が [tick, lane, ...] のパターン
        if (
            len(item) >= 2
            and all(isinstance(x, (int, float)) for x in item[:2])
            and not any(isinstance(x, list) for x in item[:2])
        ):
            try:
                tick = int(item[0])
                lane = int(item[1])
                if 0 <= lane < N_LANES:
                    out.append((tick, lane))
                return
            except (ValueError, TypeError):
                pass
        for child in item:
            visit(child, depth + 1)

    visit(raw_notes)
    out.sort()
    return out


@dataclass
class FeatureVector:
    song_id: str
    title: str | None
    level: int | None
    difficulty: str | None
    values: np.ndarray
    feature_names: list[str]


FEATURE_NAMES = [
    "total_notes",
    "unique_lanes",
    "mean_chord_size",
    "max_chord_size",
    "lane0_ratio",
    "lane1_ratio",
    "lane2_ratio",
    "lane3_ratio",
    "lane4_ratio",
    "lane5_ratio",
    "lane6_ratio",
    "scratch_ratio",
    "peak_density",
    "stair_ratio",
    "trill_ratio",
    "duration_ticks",
]


def extract_features(
    song_id: str,
    raw_notes,
    title: str | None = None,
    level: int | None = None,
    difficulty: str | None = None,
    density_window: int = 96,
) -> FeatureVector:
    """1 譜面から特徴量ベクトルを生成する。

    density_window は「同時押し」とみなす tick 幅。textage.cc は 1 拍 96 tick
    を想定して 1/16 拍 = 6 tick 程度を想定するが、形式不明の場合に備え
    既定値は緩めに 96 (≒1 拍) としている。
    """
    notes = _flatten_notes(raw_notes)
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

    lanes = np.array([lane for _, lane in notes], dtype=int)
    ticks = np.array([tick for tick, _ in notes], dtype=int)
    duration = int(ticks.max() - ticks.min()) if len(ticks) > 1 else 1

    # レーン別比率
    lane_ratios = np.zeros(N_LANES, dtype=float)
    for lane in range(N_LANES):
        lane_ratios[lane] = float(np.sum(lanes == lane)) / total_notes

    # 同時押しサイズ
    chord_sizes = _chord_sizes(ticks, lanes, density_window)
    mean_chord = float(np.mean(chord_sizes)) if chord_sizes else 0.0
    max_chord = float(np.max(chord_sizes)) if chord_sizes else 0.0

    # ピーク密度 (1 秒相当 = density_window * 4 tick の窓で滑走)
    peak_density = _peak_density(ticks, window=density_window * 4)

    # 階段率/トリル率 (時間順に並べた鍵盤のみで判定)
    key_notes = [(t, l) for t, l in notes if l < 7]
    stair_ratio, trill_ratio = _stair_trill_ratios(key_notes)

    values = np.array([
        total_notes,
        int(np.unique(lanes).size),
        mean_chord,
        max_chord,
        *lane_ratios[:7],
        lane_ratios[7],
        peak_density,
        stair_ratio,
        trill_ratio,
        duration,
    ], dtype=float)

    return FeatureVector(
        song_id=song_id,
        title=title,
        level=level,
        difficulty=difficulty,
        values=values,
        feature_names=list(FEATURE_NAMES),
    )


def _chord_sizes(ticks: np.ndarray, lanes: np.ndarray, window: int) -> list[int]:
    """同時押しの塊サイズ列を返す。

    `window` tick 以内に発生したノートを 1 つの和音とみなす。
    """
    sizes: list[int] = []
    if ticks.size == 0:
        return sizes
    cur_tick = ticks[0]
    cur_lanes: set[int] = {int(lanes[0])}
    for i in range(1, ticks.size):
        if ticks[i] - cur_tick <= window // 8:  # 1/8 拍以内なら和音とみなす
            cur_lanes.add(int(lanes[i]))
        else:
            sizes.append(len(cur_lanes))
            cur_tick = ticks[i]
            cur_lanes = {int(lanes[i])}
    sizes.append(len(cur_lanes))
    return sizes


def _peak_density(ticks: np.ndarray, window: int) -> float:
    if ticks.size == 0:
        return 0.0
    max_count = 0
    left = 0
    for right in range(ticks.size):
        while ticks[right] - ticks[left] > window:
            left += 1
        max_count = max(max_count, right - left + 1)
    return float(max_count)


def _stair_trill_ratios(notes: list[tuple[int, int]]) -> tuple[float, float]:
    """連続する鍵盤について、階段率とトリル率を計算する。

    階段: lane が +1 / -1 で連続している割合
    トリル: 直近 3 ノートが a, b, a の形になっている割合
    """
    if len(notes) < 2:
        return 0.0, 0.0
    stair = 0
    trill = 0
    for i in range(1, len(notes)):
        if abs(notes[i][1] - notes[i - 1][1]) == 1:
            stair += 1
    for i in range(2, len(notes)):
        a, b, c = notes[i - 2][1], notes[i - 1][1], notes[i][1]
        if a == c and a != b:
            trill += 1
    n = len(notes)
    return stair / max(n - 1, 1), trill / max(n - 2, 1)


def stack(features: Iterable[FeatureVector]) -> tuple[np.ndarray, list[FeatureVector]]:
    """特徴量ベクトルを行列にまとめる。"""
    items = list(features)
    if not items:
        return np.zeros((0, len(FEATURE_NAMES))), []
    matrix = np.vstack([f.values for f in items])
    return matrix, items
