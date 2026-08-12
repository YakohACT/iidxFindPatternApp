"""textage.cc の譜面データ (sp / dp / cn) デコーダ。

textage.cc のレンダラ ``bms2jsh.js`` のノート展開処理を Python に移植したもの。
純標準ライブラリのみで動作する (numpy 等に依存しない)。

データ形式 (bms2jsh.js の実装から特定):

- ``ln[n]``  : 小節 n の長さ。**384 = 4/4 拍子** (1 拍 = 96 単位)。
               null / 0 のときは 384 (LNDEF) とみなす。
               ※旧実装が仮定していた「ロングノート持続時間」ではない。

- ``sp[n]`` / ``dp[n]`` : 小節 n のノート行。3 種類の表記がある。

  1. plain hex 行 (例: ``"8100000204400010"``)
     2 桁 (= 1 byte) ごとに 1 時刻。行内の byte 数は可変で、
     位置 = ln[n] * (byte位置 * 2) / 行の宣言長。
     各 byte の bit j がレーン j のノート。**bit 0 = スクラッチ、bit 1..7 = 鍵盤 1..7**。

  2. ``x`` プレフィクス行 (例: ``"x04008@0510@04..."``)
     ``x`` + 3 桁 hex で行の宣言長を指定し、以降は 1. と同じ。
     ``@nn`` は「nn byte 分の空白をスキップ」のランレングス圧縮。
     ※旧実装はこれをメタ情報として捨てていたが、実ノート行である。

  3. ``#`` 圧縮行 (例: ``"#OU9Z0"``, ``"#OV+h7_Dw"``)
     b64 文字で周期パターン・個別ノートをエンコードした圧縮表記 (下記参照)。

- ``cn`` : チャージノート (CN / HCN / BSS)。レンダラ内で ``cn = [[], c1, c2]``
  に組み替えられた後の形で取得される。各小節のエントリは
  ``[レーン, 開始位置/3, 長さ/3, フラグ, ...]``。
  フラグ bit0 = この小節で開始、bit1 = この小節で終了 (省略時 3)。
  レーンが 10 以上のときは 2 レーン同時 (cnz%10 と cnz//10)。

# 圧縮行の文法 (bms2jsh.js L955-1095 と等価):
    先頭 '#' の後にコマンド列が続く。
    - 周期充填   : C/c R/r P/p + レーン数字 1 文字
                   (周期 192/96/48、小文字は半周期ずらし)
    - b64 パック : B/b Q/q O/o X/x Z S/s T/t U + b64 文字列
                   鍵盤モードでは 1 文字 = 2 スロット (上位/下位 3 bit がレーン)、
                   スクラッチモード ('-' 以降) では 1 文字 = 6 スロットのビット列
    - 個別指定   : '1'-'7' + 位置 2 文字 / '8','9' + レーンマスク + 位置 2 文字 (同時押し)
    - '-'        : 以降はスクラッチ (レーン 0) セクション
    - '_'        : 終端。以降の 2 文字ずつがスクラッチ位置
                   (行末単独の '_' は位置 0 のスクラッチ)
"""

from __future__ import annotations

import math

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
LNDEF = 384          # 小節長デフォルト (4/4 拍子)
UNITS_PER_BEAT = 96  # 1 拍 = 96 単位
N_LANES = 8          # SP: スクラッチ (lane 0) + 鍵盤 1..7

# 周期充填コマンド: 文字 -> (開始オフセット, 周期)
_TYPE0 = {
    "C": (0, 192), "c": (96, 192),
    "R": (0, 96), "r": (48, 96),
    "P": (0, 48), "p": (24, 48),
}
# b64 パックコマンド: 文字 -> (開始オフセット, 周期)
_TYPE1 = {
    "B": (0, 192), "b": (96, 192),
    "Q": (0, 96), "q": (48, 96),
    "O": (0, 48), "o": (24, 48),
    "X": (0, 24), "x": (12, 24),
    "Z": (0, 12),
    "S": (0, 64), "s": (32, 64),
    "T": (0, 32), "t": (16, 32),
    "U": (0, 16),
}

_HEX_CHARS = set("0123456789abcdefABCDEF")


def measure_length(measure_lengths, n: int) -> int:
    """小節 n の長さ (ln[n])。範囲外・null・0 は LNDEF (384)。"""
    if isinstance(measure_lengths, list) and 0 <= n < len(measure_lengths):
        v = measure_lengths[n]
        if isinstance(v, (int, float)) and v:
            return int(v)
    return LNDEF


def _decode_compressed(sdd: str, ln_n: int) -> list[tuple[float, int]]:
    """'#' 圧縮行 1 小節分を [(小節内位置, レーン), ...] へ展開する。"""
    notes: list[tuple[float, int]] = []
    length = len(sdd)
    sft = 1
    v2c = 0
    while sft < length:
        v2o = ""
        v2v = (1 if v2c else 3) * ln_n / 6
        ch = sdd[sft]

        if ch in _TYPE0:
            v2s, v2p = _TYPE0[ch]
            v2t = 0
            if not v2c:
                sft += 1
                v2o = sdd[sft] if sft < length else ""
            sft += 1
        elif ch in _TYPE1:
            v2s, v2p = _TYPE1[ch]
            v2t = 1
            v2b = math.ceil(v2v / v2p) + 1
            v2o = sdd[sft + 1:sft + v2b]
            sft += v2b
        elif ch in "1234567":
            v2o = sdd[sft:sft + 3]
            v2t = 2
            sft += 3
        elif ch in "89":
            if ch == "9":
                v2o = "1" + sdd[sft + 2:sft + 4]
            # JS: b64.indexOf() は -1 を返し得るが、-1 & bit は真になる
            mask = B64.find(sdd[sft + 1]) if sft + 1 < length else 0
            for i2 in range(6):
                if mask & (1 << i2):
                    v2o += str(i2 + 2) + sdd[sft + 2:sft + 4]
            v2t = 2
            sft += 4
        elif ch == "-":
            v2c = 1
            sft += 1
            continue
        elif ch == "_":
            v2o = "AA" if sft == length - 1 else sdd[sft + 1:]
            v2c = v2t = 2
        else:
            # bms2jsh.js はここで error 表示して行を打ち切る
            break

        v2k = ""
        if v2t == 1:
            if v2c == 0:
                for c in v2o:
                    v2x = B64.find(c)
                    v2k += str(v2x // 8) + str(v2x % 8)
            elif v2c == 1:
                for c in v2o:
                    v2x = B64.find(c)
                    for i3 in range(5, -1, -1):
                        v2k += "1" if (v2x >> i3) & 1 else "0"
        elif v2t == 0:
            fill = "1" if v2c else v2o
            i2 = v2s
            while i2 < ln_n:
                v2k += fill
                i2 += v2p

        if v2t != 2:
            v2i = 0
            i2 = v2s
            while i2 < ln_n:
                c = v2k[v2i] if v2i < len(v2k) else ""
                if c and c != "0" and c.isdigit():
                    lane = 0 if v2c else int(c)
                    notes.append((float(i2), lane))
                v2i += 1
                i2 += v2p
        else:
            i2 = 0
            while i2 < len(v2o):
                if v2c == 0:
                    c = v2o[i2]
                    if not c.isdigit():
                        break
                    lane = int(c)
                    i2 += 1
                else:
                    lane = 0
                if i2 + 1 >= len(v2o):
                    break
                pos = B64.find(v2o[i2]) * 64 + B64.find(v2o[i2 + 1])
                if pos >= 0:
                    notes.append((float(pos), lane))
                i2 += 2

        if v2c == 2:
            break
    return notes


def _decode_hex(sdd: str, ln_n: int) -> list[tuple[float, int]]:
    """plain hex 行 / 'x' プレフィクス行 1 小節分を展開する。"""
    notes: list[tuple[float, int]] = []
    slen = len(sdd)
    sft = 0
    div = 0
    if sdd.startswith("x"):
        if slen < 4:
            return notes
        try:
            length = int(sdd[1:4], 16)
        except ValueError:
            return notes
        sft = 4
    else:
        length = slen
    if length <= 0:
        return notes

    # 描画位置は nbar*3*div/len (nbar = ceil(ln/3), 最低 4) で計算される
    nbar3 = max(math.ceil(ln_n / 3), 4) * 3

    while sft < slen:
        while sft < slen and sdd[sft] == "@":
            try:
                div += int(sdd[sft + 1:sft + 3], 16) * 2
            except ValueError:
                return notes
            sft += 3
        if sft >= slen:
            break
        try:
            y = int(sdd[sft:sft + 2], 16)
        except ValueError:
            y = 0
        j = 0
        while j < N_LANES and (y >> j) != 0:
            if (y >> j) & 1:
                notes.append((nbar3 * div / length, j))
            j += 1
        sft += 2
        div += 2
    return notes


def decode_notes(raw_notes, measure_lengths=None) -> list[tuple[float, int]]:
    """sp / dp 配列全体を [(グローバル位置, レーン), ...] の昇順リストへ展開する。

    位置の単位は ln と同じ (4/4 拍子の 1 小節 = 384、1 拍 = 96)。
    同一 (位置, レーン) の重複は 1 つに統合される (レンダラの統計処理と同じ)。
    """
    if not isinstance(raw_notes, list):
        return []
    events: set[tuple[float, int]] = set()
    pos_acc = 0.0
    for n, row in enumerate(raw_notes):
        ln_n = measure_length(measure_lengths, n)
        if isinstance(row, str) and row:
            try:
                if row[0] == "#":
                    measure_notes = _decode_compressed(row, ln_n)
                else:
                    measure_notes = _decode_hex(row, ln_n)
            except Exception:
                measure_notes = []
            for pos, lane in measure_notes:
                events.add((pos_acc + pos, lane))
        pos_acc += ln_n
    return sorted(events)


def decode_charge_notes(charge_notes, measure_lengths=None) -> list[tuple[float, int, float, int]]:
    """cn 配列 (= [[], c1, c2]) の 1P 側を展開する。

    Returns:
        [(グローバル開始位置, レーン, 長さ, フラグ), ...]
        フラグ bit0 = この小節で開始 (= ノートとして数える)、bit1 = 終了。
        小節をまたぐ CN は複数エントリに分かれて格納されている。
    """
    if not isinstance(charge_notes, list) or len(charge_notes) < 2:
        return []
    side = charge_notes[1]
    if not isinstance(side, list):
        return []
    events: list[tuple[float, int, float, int]] = []
    pos_acc = 0.0
    for n in range(len(side)):
        ln_n = measure_length(measure_lengths, n)
        entries = side[n]
        if isinstance(entries, list):
            for e in entries:
                if not isinstance(e, list) or not e:
                    continue
                cnz = e[0]
                if not isinstance(cnz, (int, float)):
                    continue
                cnz = int(cnz)
                pos = e[1] if len(e) > 1 and isinstance(e[1], (int, float)) else 0
                cnh = e[2] if len(e) > 2 and isinstance(e[2], (int, float)) else 30
                cnf = e[3] if len(e) > 3 and isinstance(e[3], (int, float)) else 3
                lanes = [cnz % 10] if cnz < 10 else [cnz % 10, cnz // 10]
                for lane in lanes:
                    if 0 <= lane < N_LANES:
                        events.append(
                            (pos_acc + pos * 3, lane, cnh * 3, int(cnf))
                        )
        pos_acc += ln_n
    events.sort()
    return events


def count_notes(raw_notes, measure_lengths=None, charge_notes=None) -> tuple[int, int]:
    """ノート総数 (CN 開始を含む) と小節数を返す。scraper 用の簡易 API。"""
    notes = decode_notes(raw_notes, measure_lengths)
    cn_starts = [
        e for e in decode_charge_notes(charge_notes, measure_lengths) if e[3] & 1
    ]
    n_measures = len(raw_notes) if isinstance(raw_notes, list) else 0
    return len(notes) + len(cn_starts), n_measures
