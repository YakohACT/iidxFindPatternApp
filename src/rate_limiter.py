"""1時間あたりのリクエスト数を制限するレートリミッタ。

textage.cc への過剰なアクセスを防ぐため、滑走窓 (sliding window) 方式で
直近 1 時間以内のリクエスト時刻を記録し、上限に達した場合は次に
1 リクエスト分の枠が空くまでスリープする。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path


class RateLimiter:
    def __init__(
        self,
        max_requests: int = 50,
        window_seconds: int = 3600,
        state_file: Path | None = None,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.state_file = state_file
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.state_file or not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            now = time.time()
            for ts in data:
                if now - ts < self.window_seconds:
                    self._timestamps.append(ts)
        except (json.JSONDecodeError, OSError):
            # 状態ファイルが壊れていても処理は継続する
            pass

    def _save(self) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(list(self._timestamps)), encoding="utf-8"
        )

    def _prune(self) -> None:
        now = time.time()
        while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
            self._timestamps.popleft()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                self._prune()
                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(time.time())
                    self._save()
                    return
                wait = self.window_seconds - (time.time() - self._timestamps[0]) + 0.1
                print(
                    f"[RateLimiter] 上限 ({self.max_requests}/{self.window_seconds}s) に到達。"
                    f"{wait:.1f} 秒待機します..."
                )
                await asyncio.sleep(max(wait, 1.0))
