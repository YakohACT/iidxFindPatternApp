"""RANDOM オプション適用時の配置変化をシミュレートするモジュール。

IIDX の RANDOM は 1 プレイごとに鍵盤 7 レーンの並びを 7! = 5040 通りの
どれかに入れ替える (スクラッチは不変)。本モジュールは簡易運指コストモデルで
「正規配置のコスト」と「ランダム配置のコスト分布」を比較し、

  - random_advantage : 正規がランダム平均よりどれだけ悪いか (z-score)。
                       正の値 = 正規はハズレ配置寄り = 乱を掛ける価値がある
                       負の値 = 正規は当たり配置寄り = 正規/鏡向き
  - random_gamble    : ランダムのコストのばらつき (1 イベントあたりの標準偏差)。
                       大きいほど「乱の当たり外れが激しい」(デニム・同時押し曲など)
  - mirror_advantage : 正規と MIRROR のコスト差 (z-score)。正 = 鏡が得

を算出する。

運指コストモデル (1P 想定・調整可能):
  - 手の分担: 左手 = スクラッチ + 1,2,3 鍵 / 右手 = 4,5,6,7 鍵 (3:4 分業)
  - 同一手の同時押し: (本数 - 1)^2 (片手 3 個押し以上を重く見る)
  - 皿+左手鍵盤の同時 (無理皿系): 左手鍵盤 1 個につき +1.5、
    さらに 1 鍵 (皿の真横) を含む場合 +1.0
  - 16 分以内で連続するイベントが両方とも片手のみ (手が拘束される): +0.7

制約 (仕様として明記):
  - コストは同時押しと直前直後の手の拘束のみを見る簡易モデルであり、　
    乱打の運指相性などの長い文脈は扱わない
  - 縦連・連皿は RANDOM で変化しないため、この指標ではなく
    jack_ratio / scratch_stream_ratio を併せて読むこと
  - 2P の場合は左右を読み替える (mirror_advantage の符号が逆になる)
"""

from __future__ import annotations

import itertools
import random as _random

import numpy as np

from patterns import EPS, SIXTEENTH, Chord

# --- 運指コストモデルの定数 -------------------------------------------------
LEFT_KEYS = (1, 2, 3)      # 左手が受け持つ鍵盤 (1P・3:4 分業)
W_SCR_LEFT = 1.5           # 皿と同時の左手鍵盤 1 個あたり
W_SCR_KEY1 = 1.0           # 皿と同時に 1 鍵 (皿の真横) を含む場合の追加
W_SEQ = 0.7                # 16 分以内の連続イベントが両方片手のみの場合

N_PERMS = 1024             # サンプリングする順列数 (identity/mirror は別枠で厳密)
SEED = 42

_N_MASKS = 128             # 鍵盤 7 bit のマスク数
_Z_CLIP = 10.0             # z-score の暴走防止


_tables_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _build_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """順列ごとのコストテーブルを構築する (プロセス内で 1 回だけ)。

    Returns:
        event_cost : (n_perms, 256)  バケット = キーマスク + 128*皿有無
        left_only  : (n_perms, 128)  マスクが「左手のみ」になるか (0/1)
        right_only : (n_perms, 128)  マスクが「右手のみ」になるか (0/1)
        行 0 = 正規 (identity)、行 1 = MIRROR、行 2 以降 = ランダムサンプル
    """
    global _tables_cache
    if _tables_cache is not None:
        return _tables_cache

    all_perms = list(itertools.permutations(range(1, 8)))
    rng = _random.Random(SEED)
    sampled = rng.sample(all_perms, min(N_PERMS, len(all_perms)))
    identity = tuple(range(1, 8))
    mirror = tuple(range(7, 0, -1))
    perms = np.array([identity, mirror] + sampled, dtype=np.int64)  # (n, 7)

    # bits[mask, i] = 譜面レーン i+1 がマスクに立っているか
    masks = np.arange(_N_MASKS)
    bits = ((masks[:, None] >> np.arange(7)[None, :]) & 1).astype(np.float32)

    left_phys = np.isin(perms, LEFT_KEYS).astype(np.float32)        # (n, 7)
    key1_phys = (perms == 1).astype(np.float32)                     # (n, 7)

    L = left_phys @ bits.T          # (n, 128) 左手の同時本数
    K = bits.sum(axis=1)[None, :]   # (1, 128) 総本数
    R = K - L                       # 右手の同時本数
    K1 = key1_phys @ bits.T         # (n, 128) 物理 1 鍵を含むか (0/1)

    chord_pen = np.maximum(L - 1, 0) ** 2 + np.maximum(R - 1, 0) ** 2
    ev_no_scr = chord_pen
    ev_scr = chord_pen + W_SCR_LEFT * L + W_SCR_KEY1 * K1
    event_cost = np.concatenate([ev_no_scr, ev_scr], axis=1).astype(np.float32)

    left_only = ((L > 0) & (R == 0)).astype(np.float32)
    right_only = ((R > 0) & (L == 0)).astype(np.float32)

    _tables_cache = (event_cost, left_only, right_only)
    return _tables_cache


def chart_histograms(
    chords: list[Chord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """同時押し列からイベント/ペアのヒストグラムを作る。

    Returns:
        ev_hist : (256,)  バケット = キーマスク + 128*皿有無 の出現回数
        m1, m2  : 16 分以内で連続するイベント対のキーマスク配列
        h       : 各対の出現回数
    """
    ev_hist = np.zeros(256, dtype=np.float32)
    pair_counts: dict[tuple[int, int], float] = {}
    prev_mask = 0
    prev_pos: float | None = None
    for pos, lanes in chords:
        mask = 0
        scratch = 0
        for lane in lanes:
            if lane == 0:
                scratch = 1
            else:
                mask |= 1 << (lane - 1)
        ev_hist[mask + 128 * scratch] += 1
        if mask:
            if (
                prev_mask
                and prev_pos is not None
                and pos - prev_pos <= SIXTEENTH + EPS
            ):
                key = (prev_mask, mask)
                pair_counts[key] = pair_counts.get(key, 0.0) + 1.0
            prev_mask = mask
            prev_pos = pos
    if pair_counts:
        m1 = np.array([k[0] for k in pair_counts], dtype=np.int64)
        m2 = np.array([k[1] for k in pair_counts], dtype=np.int64)
        h = np.array(list(pair_counts.values()), dtype=np.float32)
    else:
        m1 = m2 = np.zeros(0, dtype=np.int64)
        h = np.zeros(0, dtype=np.float32)
    return ev_hist, m1, m2, h


def randomness_features(chords: list[Chord]) -> tuple[float, float, float]:
    """(random_advantage, random_gamble, mirror_advantage) を返す。"""
    if not chords:
        return 0.0, 0.0, 0.0
    event_cost, left_only, right_only = _build_tables()
    ev_hist, m1, m2, h = chart_histograms(chords)

    costs = event_cost @ ev_hist  # (n_perms,)
    if h.size:
        pair_cost = (
            left_only[:, m1] * left_only[:, m2]
            + right_only[:, m1] * right_only[:, m2]
        ) @ h
        costs = costs + W_SEQ * pair_cost

    regular = float(costs[0])
    mirror = float(costs[1])
    rand = costs[2:]
    mu = float(rand.mean())
    sd = float(rand.std())
    n_events = float(ev_hist.sum())

    if sd < 1e-9 or n_events <= 0:
        return 0.0, 0.0, 0.0
    adv = (regular - mu) / sd
    gamble = sd / n_events  # 1 イベントあたりの振れ幅
    mir_adv = (regular - mirror) / sd
    clip = lambda v: float(np.clip(v, -_Z_CLIP, _Z_CLIP))  # noqa: E731
    return clip(adv), float(np.clip(gamble, 0.0, _Z_CLIP)), clip(mir_adv)
