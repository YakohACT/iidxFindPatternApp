"""譜面の生データから機械学習向けの特徴量ベクトルを生成するモジュール。

ノート展開は textage_notes モジュール (textage.cc レンダラの移植)、
配置パターン検出は patterns モジュールが行う。
時間の単位は textage 準拠で **1 拍 = 96、4/4 拍子の 1 小節 = 384**。
レーンは **0 = スクラッチ、1..7 = 鍵盤 1〜7**。

抽出する特徴量 (26 次元):

  基本量:
  - total_notes:     総ノート数 (CN/BSS の開始を含む)
  - measures:        最初のノートから最後のノートまでの長さ (4/4 小節換算)
  - density:         ノート数 / 小節
  - scratch_ratio:   スクラッチ比率
  - mean_chord:      同時押し平均サイズ
  - max_chord:       同時押し最大サイズ
  - peak_density:    2 拍 (192 単位) 幅の滑走窓に入る最大ノート数 (発狂)
  - stream16_ratio:  16 分間隔で長く続く塊 (乱打) のノート比率

  階段系:
  - stair_mono_ratio:    単階段 (1→2→3 と同方向に続く) のノート比率
  - stair_turn_ratio:    折り返し階段 (1→2→3→2→1) のノート比率
  - big_stair_rate:      大階段 (1→7 の 7 連) の小節あたり出現回数
  - double_stair_ratio:  二重階段 (1,3→2,4→3,5) のノート比率
  - garbage_stair_ratio: ゴミ付き階段 (階段+余分な同時打鍵) のノート比率

  リズム/同時押し系:
  - denim_ratio:       デニム (1,3,5,7⇔2,4,6 型の交互押し) のノート比率
  - chord_trill_ratio: 二重トリル (1,3⇔2,4 等の同時押し往復) のノート比率
  - trill_ratio:       トリル (2 レーン往復 4 打以上) のノート比率
  - jack_ratio:        縦連 (同一鍵盤 16 分 3 打以上) のノート比率
  - wall_ratio:        壁 (3 個以上の同時押しの連続) のノート比率
  - axis_ratio:        軸 (同一鍵盤が 4 分以内で降り続け合間に他レーン) のノート比率

  スクラッチ系:
  - scratch_stream_ratio:   連皿 (8 分以内 3 打以上) のノート比率
  - scratch_key_mix_ratio:  皿複合 (前後 16 分以内に鍵盤があるスクラッチ) の比率
  - scratch_left_chord_ratio: 無理皿気味 (皿と 1,2 鍵の同時) の比率 (1P 基準)
  - mss_rate:               マルチスピンスクラッチの小節あたり本数

  その他:
  - peak_position:   最も密度が高い 2 拍窓の曲中の位置 (0=序盤, 1=ラスト。ラス殺し検出)
  - offgrid_ratio:   16 分グリッドに乗らないノート比率 (24分/32分/ズレ)
  - hand_bias:       左手側 (1-3) と右手側 (5-7) の偏り (0=均等)
  - cn_ratio:        CN/HCN/BSS 開始ノートの比率
  - bpm_range_log2:  log2(最大BPM/最小BPM)。0 = BPM 一定、1 = 2 倍変化
  - bpm_change_rate: BPM 変化回数 / 小節 (ソフランの頻度)

  RANDOM オプション判断 (random_sim による 7! 順列シミュレーション):
  - random_advantage: 正規配置がランダム平均よりどれだけ悪いか (z-score)。
                      正 = 乱を掛ける価値あり / 負 = 正規当たり配置
  - random_gamble:    乱の当たり外れの激しさ (コスト分布の変動係数)
  - mirror_advantage: MIRROR にするとどれだけ得か (z-score)。正 = 鏡推奨
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

import patterns
import random_sim
from textage_notes import LNDEF, N_LANES, decode_charge_notes, decode_notes


PEAK_WINDOW = 192  # 2 拍 (= 半小節 @ 4/4) の滑走窓


FEATURE_NAMES = [
    "total_notes",
    "measures",
    "density",
    "scratch_ratio",
    "mean_chord",
    "max_chord",
    "peak_density",
    "stream16_ratio",
    "stair_mono_ratio",
    "stair_turn_ratio",
    "big_stair_rate",
    "double_stair_ratio",
    "garbage_stair_ratio",
    "denim_ratio",
    "chord_trill_ratio",
    "trill_ratio",
    "jack_ratio",
    "wall_ratio",
    "axis_ratio",
    "scratch_stream_ratio",
    "scratch_key_mix_ratio",
    "scratch_left_chord_ratio",
    "mss_rate",
    "peak_position",
    "offgrid_ratio",
    "hand_bias",
    "cn_ratio",
    "bpm_range_log2",
    "bpm_change_rate",
    "random_advantage",
    "random_gamble",
    "mirror_advantage",
]


@dataclass
class FeatureVector:
    song_id: str
    title: str | None
    level: int | None
    difficulty: str | None
    values: np.ndarray
    feature_names: list[str]


def _peak_density(positions: list[float], window: float = PEAK_WINDOW) -> tuple[int, float]:
    """幅 `window` (ln 単位) の滑走窓に入る最大ノート数と、その窓の中心位置。"""
    if not positions:
        return 0, 0.0
    max_count = 0
    peak_center = positions[0]
    left = 0
    for right in range(len(positions)):
        while positions[right] - positions[left] > window:
            left += 1
        if right - left + 1 > max_count:
            max_count = right - left + 1
            peak_center = (positions[left] + positions[right]) / 2.0
    return max_count, peak_center


def extract_features(
    song_id: str,
    raw_notes,
    title: str | None = None,
    level: int | None = None,
    difficulty: str | None = None,
    long_notes=None,
    charge_notes=None,
    tempo=None,
    bpm: str | None = None,
) -> FeatureVector:
    """1 譜面分の生データを特徴量ベクトルへ変換する。

    ``long_notes`` は textage の ``ln`` 配列 (小節長リスト)、``tempo`` は
    ``tc`` 配列 (BPM 変化)、``bpm`` は表示用 BPM 文字列をそのまま渡す。
    """
    notes = decode_notes(raw_notes, long_notes)
    cn_events = decode_charge_notes(charge_notes, long_notes)
    cn_starts = [(pos, lane) for pos, lane, _length, flags in cn_events if flags & 1]

    # CN の開始も「叩くノート」として通常ノートに合流させる
    all_notes = sorted(set(notes) | set(cn_starts))
    total_notes = len(all_notes)

    if total_notes == 0:
        return FeatureVector(
            song_id=song_id,
            title=title,
            level=level,
            difficulty=difficulty,
            values=np.zeros(len(FEATURE_NAMES), dtype=float),
            feature_names=list(FEATURE_NAMES),
        )

    # --- 基本量 -----------------------------------------------------------
    span = all_notes[-1][0] - all_notes[0][0]
    measures = int(span // LNDEF) + 1
    density = total_notes / measures

    lane_counts = np.zeros(N_LANES, dtype=float)
    for _, lane in all_notes:
        lane_counts[lane] += 1
    scratch_ratio = lane_counts[0] / total_notes

    chords = patterns.group_chords(all_notes)
    chord_sizes = [len(lanes) for _, lanes in chords]
    mean_chord = float(np.mean(chord_sizes))
    max_chord = float(np.max(chord_sizes))
    peak_density, peak_center = _peak_density([pos for pos, _ in all_notes])
    span_start = all_notes[0][0]
    peak_position = (peak_center - span_start) / span if span > 0 else 0.0

    # --- 配置パターン検出 -------------------------------------------------
    kchords = patterns.key_chords(chords)
    stairs = patterns.detect_stairs(kchords)
    trill_notes = patterns.detect_trills(kchords)
    double_stair_notes = patterns.detect_double_stairs(kchords)
    garbage_stair_notes = patterns.detect_garbage_stairs(kchords)
    denim_notes = patterns.detect_denim(kchords)
    chord_trill_notes = patterns.detect_chord_trills(kchords)
    jack_notes = patterns.detect_jacks(all_notes)
    wall_notes = patterns.detect_walls(kchords)
    axis_notes = patterns.detect_axis(all_notes)
    stream16_notes = patterns.detect_stream16(kchords)
    scratch_stream_notes = patterns.detect_scratch_stream(all_notes)
    scratch_mix_notes = patterns.detect_scratch_key_mix(all_notes)
    offgrid_notes = patterns.count_offgrid(all_notes)
    hand_bias = patterns.hand_bias(all_notes)

    # 無理皿気味: 皿と 1,2 鍵が同時のイベント数 (1P 正規配置基準)
    scratch_left_chords = sum(
        1 for _pos, lanes in chords
        if 0 in lanes and (1 in lanes or 2 in lanes)
    )

    mss_count = sum(
        1 for _pos, lane, _length, flags in cn_events
        if lane == 0 and flags & 4 and flags & 1
    )
    bpm_range_log2, bpm_changes = patterns.bpm_stats(tempo, bpm)

    # RANDOM オプションの当たり外れシミュレーション
    random_advantage, random_gamble, mirror_advantage = (
        random_sim.randomness_features(chords)
    )

    values = np.array([
        total_notes,
        measures,
        density,
        scratch_ratio,
        mean_chord,
        max_chord,
        peak_density,
        stream16_notes / total_notes,
        stairs.mono_notes / total_notes,
        stairs.turn_notes / total_notes,
        stairs.big_runs / measures,
        double_stair_notes / total_notes,
        garbage_stair_notes / total_notes,
        denim_notes / total_notes,
        chord_trill_notes / total_notes,
        trill_notes / total_notes,
        jack_notes / total_notes,
        wall_notes / total_notes,
        axis_notes / total_notes,
        scratch_stream_notes / total_notes,
        scratch_mix_notes / total_notes,
        scratch_left_chords / total_notes,
        mss_count / measures,
        peak_position,
        offgrid_notes / total_notes,
        hand_bias,
        len(cn_starts) / total_notes,
        bpm_range_log2,
        bpm_changes / measures,
        random_advantage,
        random_gamble,
        mirror_advantage,
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
