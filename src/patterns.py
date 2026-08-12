"""(位置, レーン) イベント列から IIDX の特徴的な配置パターンを検出するモジュール。

時間の単位は textage 準拠のメトリック時間 (1 拍 = 96、4/4 の 1 小節 = 384)。
   16 分 = 24 / 8 分 = 48 / 4 分 = 96
レーンは 0 = スクラッチ、1..7 = 鍵盤。

検出は拍基準 (BPM に依存しない譜面上の形) で行い、BPM の緩急は
ソフラン系特徴量 (bpm_range_log2 / bpm_change_rate) が別途受け持つ。

主な判定基準 (定数はモジュール先頭で調整可能):
  - 連続とみなす間隔: 階段/壁/デニム/連皿 = 8分以内、縦連/乱打 = 16分以内
  - 単階段     : 単ノートが隣接レーンへ同方向に 3 連以上 (3 レーン以上)
  - 折り返し階段: 隣接移動の連なりの中で方向転換があり 4 連以上 (3 レーン以上)
  - 大階段     : 1→7 (または 7→1) を単調に駆け抜ける 7 連
  - 二重階段   : サイズ 2 以上の同時押しが全レーン同方向に ±1 シフトして 3 連以上
  - ゴミ付き階段: 単調な隣接進行 (4 連以上) に同時打鍵の余分なノートが混ざる
  - デニム     : 互いに素でレーン範囲が交錯するサイズ 3 以上の同時押しの交互 (3 連以上)
  - 縦連       : 同一鍵盤を 16 分以内の間隔で 3 打以上
  - 壁         : サイズ 3 以上の同時押しが 8分以内で 2 連以上
  - 連皿       : スクラッチが 8分以内の間隔で 3 連以上
  - 皿複合     : スクラッチの前後 16 分以内に鍵盤ノートがある
  - 乱打(16分) : 16 分以内の間隔で 8 イベント以上続く塊
  - 軸         : 同一鍵盤が 4 分以内の間隔で 4 打以上、かつ合間の過半に他レーンが降る
  - ズレ/裏拍  : 16 分グリッド (24 の倍数) に乗らないノート
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

SIXTEENTH = 24.0
EIGHTH = 48.0
QUARTER = 96.0
EPS = 1.0  # 浮動小数の位置ずれ許容

MIN_MONO_STAIR = 3      # 単階段の最短連数
MIN_TURN_STAIR = 4      # 折り返し階段の最短連数
BIG_STAIR_SPAN = 7      # 大階段のレーン数 (1→7)
MIN_DOUBLE_STAIR = 3    # 二重階段の最短連数
MIN_GARBAGE_STAIR = 4   # ゴミ付き階段の最短連数
MIN_DENIM = 3           # デニムの最短連数
MIN_JACK = 3            # 縦連の最短打数
MIN_WALL = 2            # 壁の最短連数
MIN_SCRATCH_STREAM = 3  # 連皿の最短打数
MIN_STREAM16 = 8        # 16分乱打の最短イベント数
MIN_AXIS = 4            # 軸の最短打数
MIN_TRILL = 4           # トリルの最短打数
MIN_CHORD_TRILL = 4     # 二重トリルの最短連数


Chord = tuple[float, tuple[int, ...]]  # (位置, ソート済みレーン)


def group_chords(events: list[tuple[float, int]]) -> list[Chord]:
    """(pos, lane) 列を位置ごとの同時押し (pos, lanes) 列へ変換する。"""
    chords: list[Chord] = []
    cur_pos: float | None = None
    cur_lanes: list[int] = []
    for pos, lane in events:
        if cur_pos is not None and pos == cur_pos:
            cur_lanes.append(lane)
        else:
            if cur_pos is not None:
                chords.append((cur_pos, tuple(sorted(cur_lanes))))
            cur_pos = pos
            cur_lanes = [lane]
    if cur_pos is not None:
        chords.append((cur_pos, tuple(sorted(cur_lanes))))
    return chords


def key_chords(chords: list[Chord]) -> list[Chord]:
    """スクラッチ (レーン 0) を除いた鍵盤のみの同時押し列。"""
    out: list[Chord] = []
    for pos, lanes in chords:
        keys = tuple(l for l in lanes if l != 0)
        if keys:
            out.append((pos, keys))
    return out


# ---------------------------------------------------------------- 階段ファミリ

@dataclass
class StairCounts:
    mono_notes: int = 0      # 単階段に含まれるノート数
    turn_notes: int = 0      # 折り返し階段に含まれるノート数
    big_runs: int = 0        # 大階段 (1→7 単調) の出現回数


def _classify_adjacent_segment(lanes: list[int], counts: StairCounts) -> None:
    """隣接移動 (|Δ|=1) だけで構成された単ノート列 1 区間を分類する。"""
    n = len(lanes)
    if n < 2:
        return
    span = len(set(lanes))
    turns = 0
    for i in range(2, n):
        if (lanes[i] - lanes[i - 1]) != (lanes[i - 1] - lanes[i - 2]):
            turns += 1
    if span == 2:
        # 2 レーンの往復はトリル (detect_trills が数えるので階段からは除外)
        return
    if turns == 0:
        if n >= MIN_MONO_STAIR and span >= 3:
            counts.mono_notes += n
    else:
        if n >= MIN_TURN_STAIR and span >= 3:
            counts.turn_notes += n
    # 大階段: 単調な部分列で 7 レーンを駆け抜けたか
    run_start = 0
    for i in range(1, n + 1):
        if i == n or (i >= 2 and (lanes[i] - lanes[i - 1]) != (lanes[i - 1] - lanes[i - 2])):
            if i - run_start >= BIG_STAIR_SPAN:
                counts.big_runs += (i - run_start) - BIG_STAIR_SPAN + 1
            run_start = i - 1


def detect_stairs(kchords: list[Chord]) -> StairCounts:
    """単階段 / 折り返し階段 / 大階段 / トリルを検出する。

    「単ノートが 8 分以内の間隔で隣接レーンに移動し続ける区間」を切り出し、
    方向転換の有無・レーン範囲で分類する。
    """
    counts = StairCounts()
    seg: list[int] = []
    prev_pos: float | None = None
    for pos, lanes in kchords:
        ok = False
        if len(lanes) == 1:
            lane = lanes[0]
            if not seg:
                seg = [lane]
                ok = True
            elif (
                prev_pos is not None
                and pos - prev_pos <= EIGHTH + EPS
                and abs(lane - seg[-1]) == 1
            ):
                seg.append(lane)
                ok = True
        if not ok:
            _classify_adjacent_segment(seg, counts)
            seg = [lanes[0]] if len(lanes) == 1 else []
        prev_pos = pos
    _classify_adjacent_segment(seg, counts)
    return counts


def detect_trills(kchords: list[Chord]) -> int:
    """トリル: 単ノートが 2 レーン間 (間隔は問わない) を交互に往復する
    4 打以上の区間のノート数。8 分以内の間隔で連続していること。
    """
    total = 0
    seg: list[int] = []
    prev_pos: float | None = None

    def flush() -> None:
        nonlocal total
        m = len(seg)
        if m < MIN_TRILL:
            return
        start = 0
        for k in range(2, m + 1):
            if k == m or seg[k] != seg[k - 2] or seg[k] == seg[k - 1]:
                if k - start >= MIN_TRILL and seg[start] != seg[start + 1]:
                    total += k - start
                start = k - 1

    for pos, lanes in kchords:
        if len(lanes) == 1 and (
            not seg or (prev_pos is not None and pos - prev_pos <= EIGHTH + EPS)
        ):
            seg.append(lanes[0])
        else:
            flush()
            seg = [lanes[0]] if len(lanes) == 1 else []
        prev_pos = pos
    flush()
    return total


def detect_chord_trills(kchords: list[Chord]) -> int:
    """二重トリル: 互いに素な 2 つの同時押し (各 2 個以上) が
    A→B→A→B と交互に降る 4 連以上の区間のノート数。
    (例: 1,3⇔2,4 / 1,2⇔6,7。デニムより小さい同時押しの往復も拾う)
    """
    total = 0
    run: list[Chord] = []

    def flush() -> None:
        nonlocal total
        if len(run) >= MIN_CHORD_TRILL:
            total += sum(len(lanes) for _, lanes in run)

    prev_pos: float | None = None
    for pos, lanes in kchords:
        joined = False
        if len(lanes) >= 2:
            if run and prev_pos is not None and pos - prev_pos <= EIGHTH + EPS:
                ok = not (set(lanes) & set(run[-1][1]))
                if ok and len(run) >= 2:
                    ok = lanes == run[-2][1]
                if ok:
                    run.append((pos, lanes))
                    joined = True
            if not joined:
                flush()
                run = [(pos, lanes)]
                joined = True
        if not joined:
            flush()
            run = []
        prev_pos = pos
    flush()
    return total


def detect_double_stairs(kchords: list[Chord]) -> int:
    """二重階段: 同時押し全体が ±1 シフトしながら降る連なりのノート数。"""
    total = 0
    run: list[Chord] = []

    def flush() -> None:
        nonlocal total
        if len(run) >= MIN_DOUBLE_STAIR:
            total += sum(len(lanes) for _, lanes in run)

    prev_pos: float | None = None
    for pos, lanes in kchords:
        extended = False
        if len(lanes) >= 2:
            if run and prev_pos is not None and pos - prev_pos <= EIGHTH + EPS:
                prev_lanes = run[-1][1]
                if len(lanes) == len(prev_lanes):
                    for delta in (1, -1):
                        if all(a == b + delta for a, b in zip(lanes, prev_lanes)):
                            run.append((pos, lanes))
                            extended = True
                            break
            if not extended:
                flush()
                run = [(pos, lanes)]
                extended = True
        if not extended:
            flush()
            run = []
        prev_pos = pos
    flush()
    return total


def detect_garbage_stairs(kchords: list[Chord]) -> int:
    """ゴミ付き階段: 単調な隣接進行の芯に余分な同時打鍵が付く区間のノート数。

    連続イベント間で「あるレーンが ±1 ずつ単調に進む経路」を DP で追跡し、
    経路長 >= MIN_GARBAGE_STAIR かつ経路上のイベントに余分なノートが
    含まれる場合に、その区間のイベント全ノートを数える。
    (純粋な単階段は余分ノートが無いため対象外)
    """
    n = len(kchords)
    if n == 0:
        return 0
    marked: set[int] = set()
    # state: (lane, direction) -> (連長, 開始 index, ゴミを含むか)
    prev_state: dict[tuple[int, int], tuple[int, int, bool]] = {}
    for i, (pos, lanes) in enumerate(kchords):
        state: dict[tuple[int, int], tuple[int, int, bool]] = {}
        gap_ok = (
            i > 0 and pos - kchords[i - 1][0] <= EIGHTH + EPS
        )
        has_extra = len(lanes) >= 2
        for lane in lanes:
            for d in (1, -1):
                length, start, garbage = 1, i, has_extra
                if gap_ok and (lane - d, d) in prev_state:
                    p_len, p_start, p_garbage = prev_state[(lane - d, d)]
                    length = p_len + 1
                    start = p_start
                    garbage = p_garbage or has_extra
                cur = state.get((lane, d))
                if cur is None or length > cur[0]:
                    state[(lane, d)] = (length, start, garbage)
                if length >= MIN_GARBAGE_STAIR and garbage:
                    marked.update(range(start, i + 1))
        prev_state = state
    return sum(len(kchords[i][1]) for i in marked)


def detect_denim(kchords: list[Chord]) -> int:
    """デニム: 互いに素でレーン範囲が交錯する大きめ同時押しの交互連打。"""
    total = 0
    run_len = 0
    run_notes = 0
    prev: Chord | None = None

    def flush() -> None:
        nonlocal total, run_len, run_notes
        if run_len >= MIN_DENIM:
            total += run_notes

    for pos, lanes in kchords:
        joined = False
        if len(lanes) >= 3:
            if prev is not None and pos - prev[0] <= EIGHTH + EPS:
                p = prev[1]
                if (
                    len(p) >= 3
                    and not (set(lanes) & set(p))
                    and min(lanes) < max(p)
                    and min(p) < max(lanes)
                ):
                    if run_len == 0:
                        run_len = 1
                        run_notes = len(p)
                    run_len += 1
                    run_notes += len(lanes)
                    joined = True
        if not joined:
            flush()
            run_len = 0
            run_notes = 0
        prev = (pos, lanes)
    flush()
    return total


def detect_jacks(events: list[tuple[float, int]]) -> int:
    """縦連: 同一鍵盤の 16 分以内連打 (3 打以上) のノート数。"""
    by_lane: dict[int, list[float]] = {}
    for pos, lane in events:
        if lane != 0:
            by_lane.setdefault(lane, []).append(pos)
    total = 0
    for positions in by_lane.values():
        run = 1
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] <= SIXTEENTH + EPS:
                run += 1
            else:
                if run >= MIN_JACK:
                    total += run
                run = 1
        if run >= MIN_JACK:
            total += run
    return total


def detect_walls(kchords: list[Chord]) -> int:
    """壁: サイズ 3 以上の同時押しが 8 分以内で連続する区間のノート数。"""
    total = 0
    run: list[int] = []
    prev_pos: float | None = None
    for pos, lanes in kchords:
        if len(lanes) >= 3 and (
            not run or (prev_pos is not None and pos - prev_pos <= EIGHTH + EPS)
        ):
            run.append(len(lanes))
        else:
            if len(run) >= MIN_WALL:
                total += sum(run)
            run = [len(lanes)] if len(lanes) >= 3 else []
        prev_pos = pos
    if len(run) >= MIN_WALL:
        total += sum(run)
    return total


def detect_scratch_stream(events: list[tuple[float, int]]) -> int:
    """連皿: スクラッチの 8 分以内連打 (3 打以上) のノート数。"""
    positions = [pos for pos, lane in events if lane == 0]
    total = 0
    run = 1
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] <= EIGHTH + EPS:
            run += 1
        else:
            if run >= MIN_SCRATCH_STREAM:
                total += run
            run = 1
    if run >= MIN_SCRATCH_STREAM:
        total += run
    return total


def detect_scratch_key_mix(events: list[tuple[float, int]]) -> int:
    """皿複合: 前後 16 分以内に鍵盤ノートがあるスクラッチの数。"""
    key_positions = sorted(pos for pos, lane in events if lane != 0)
    if not key_positions:
        return 0
    count = 0
    for pos, lane in events:
        if lane != 0:
            continue
        lo = bisect_left(key_positions, pos - SIXTEENTH - EPS)
        hi = bisect_right(key_positions, pos + SIXTEENTH + EPS)
        if hi > lo:
            count += 1
    return count


def detect_stream16(kchords: list[Chord]) -> int:
    """16 分乱打: 16 分以内の間隔で続く長い塊 (8 イベント以上) のノート数。"""
    total = 0
    run: list[int] = []
    prev_pos: float | None = None
    for pos, lanes in kchords:
        if run and prev_pos is not None and pos - prev_pos <= SIXTEENTH + EPS:
            run.append(len(lanes))
        else:
            if len(run) >= MIN_STREAM16:
                total += sum(run)
            run = [len(lanes)]
        prev_pos = pos
    if len(run) >= MIN_STREAM16:
        total += sum(run)
    return total


def detect_axis(events: list[tuple[float, int]]) -> int:
    """軸: 同一鍵盤が 4 分以内の間隔で規則的に降り続け (4 打以上)、
    その合間の過半に他レーンのノートが降っている区間のノート数 (軸打鍵のみ)。

    縦連 (合間に何も無い) と区別するため合間ノートを要求し、
    トリル (合間が常に同じ 1 レーン) と区別するため合間に登場する
    レーンが 2 種類以上であることを要求する。
    """
    key_events = sorted((pos, lane) for pos, lane in events if lane != 0)
    key_positions = [pos for pos, _ in key_events]
    total = 0
    by_lane: dict[int, list[float]] = {}
    for pos, lane in key_events:
        by_lane.setdefault(lane, []).append(pos)

    def other_lanes_between(a: float, b: float, lane: int) -> set[int]:
        lo = bisect_right(key_positions, a + EPS)
        hi = bisect_left(key_positions, b - EPS)
        return {key_events[idx][1] for idx in range(lo, hi)} - {lane}

    for lane, positions in by_lane.items():
        i = 0
        n = len(positions)
        while i < n:
            j = i
            while j + 1 < n and positions[j + 1] - positions[j] <= QUARTER + EPS:
                j += 1
            run_len = j - i + 1
            if run_len >= MIN_AXIS:
                gaps_with_others = 0
                seen_lanes: set[int] = set()
                for k in range(i, j):
                    others = other_lanes_between(positions[k], positions[k + 1], lane)
                    if others:
                        gaps_with_others += 1
                        seen_lanes |= others
                if gaps_with_others * 2 >= run_len - 1 and len(seen_lanes) >= 2:
                    total += run_len
            i = j + 1
    return total


def count_offgrid(events: list[tuple[float, int]]) -> int:
    """16 分グリッド (24 の倍数) に乗らないノート数 (24分/32分/ズレ配置)。"""
    count = 0
    for pos, _lane in events:
        r = pos % SIXTEENTH
        if min(r, SIXTEENTH - r) > EPS:
            count += 1
    return count


def hand_bias(events: list[tuple[float, int]]) -> float:
    """左手側 (1-3) と右手側 (5-7) のノート数の偏り (0 = 均等, 1 = 完全片寄り)。"""
    left = sum(1 for _, lane in events if lane in (1, 2, 3))
    right = sum(1 for _, lane in events if lane in (5, 6, 7))
    total = left + right
    if total == 0:
        return 0.0
    return abs(left - right) / total


# ---------------------------------------------------------------- ソフラン

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def bpm_stats(tempo, bpm_str: str | None) -> tuple[float, int]:
    """(log2(最大BPM/最小BPM), BPM 変化回数) を返す。

    tempo は textage の tc 配列 (小節ごとの文字列リスト。先頭 3 文字が BPM、
    残りが小節内位置)。bpm_str は "11～178" のような表示用文字列で、
    tc が無い場合の範囲推定に使う。
    """
    seq: list[float] = []
    if isinstance(tempo, list):
        for measure_entries in tempo:
            if not isinstance(measure_entries, list):
                continue
            for cb in measure_entries:
                if isinstance(cb, str) and len(cb) >= 3:
                    head = cb[:3].strip()
                    try:
                        v = float(head)
                    except ValueError:
                        continue
                    if v > 0:
                        seq.append(v)

    values = list(seq)
    if isinstance(bpm_str, str):
        values.extend(
            float(m) for m in _NUM_RE.findall(bpm_str) if float(m) > 0
        )

    if not values:
        return 0.0, 0

    range_log2 = 0.0
    mn, mx = min(values), max(values)
    if mn > 0 and mx > mn:
        range_log2 = math.log2(mx / mn)

    changes = 0
    prev: float | None = None
    for v in seq:
        if prev is not None and v != prev:
            changes += 1
        prev = v
    return range_log2, changes
