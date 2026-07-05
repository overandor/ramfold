"""
RAMFold Memory Observer — macOS Unified Memory Probe

Measures live Apple unified-memory state:
  P_t = f(active, wired, compressed, swap, GPU allocation, thermal)

Does not trust theoretical parameter counts alone.
Watches the Mac while the model runs.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Optional

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


@dataclass
class MemorySnapshot:
    timestamp: float
    page_size: int
    free_gb: float
    active_gb: float
    wired_gb: float
    compressed_gb: float
    swap_used_gb: float
    pressure: float
    mlx_active_mb: float
    mlx_peak_mb: float
    mlx_cache_mb: float
    thermal_pressure: str

    @property
    def used_gb(self) -> float:
        return self.active_gb + self.wired_gb + self.compressed_gb

    @property
    def swap_delta_risk(self) -> bool:
        return self.swap_used_gb > 0.5


class MemoryObserver:
    """Live macOS unified-memory observer."""

    def __init__(self, budget_gb: float = 18.0):
        self.budget_gb = budget_gb
        self.budget_bytes = int(budget_gb * 1024**3)
        self._baseline: Optional[MemorySnapshot] = None
        self._swap_history: list[float] = []
        self._pressure_history: list[float] = []

    def _vm_stat(self) -> tuple[str, int]:
        text = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        page_size = 16384
        if "page size of" in text:
            try:
                page_size = int(text.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                pass
        return text, page_size

    def _pages(self, text: str, label: str) -> int:
        for line in text.splitlines():
            if label in line:
                digits = "".join(ch for ch in line if ch.isdigit())
                return int(digits or "0")
        return 0

    def _swap_usage(self) -> float:
        try:
            text = subprocess.check_output(["/usr/sbin/sysctl", "vm.swapusage"], text=True)
            m = re.search(r"used = ([0-9.]+)([MG])", text)
            if m:
                val = float(m.group(1))
                return val / 1024.0 if m.group(2) == "M" else val
        except Exception:
            pass
        return 0.0

    def _thermal(self) -> str:
        try:
            text = subprocess.check_output(
            ["/usr/bin/pmset", "-g", "therm"], text=True
            )
            if "CPU_Speed_Limit" in text:
                for line in text.splitlines():
                    if "CPU_Speed_Limit" in line:
                        val = int("".join(ch for ch in line if ch.isdigit()) or "100")
                        if val < 80:
                            return "throttled"
                        elif val < 95:
                            return "warm"
                        return "nominal"
            return "nominal"
        except Exception:
            return "unknown"

    def _mlx_memory(self) -> dict:
        if not _HAS_MLX:
            return {"active_mb": 0, "peak_mb": 0, "cache_mb": 0}
        try:
            return {
                "active_mb": mx.get_active_memory() / 1024**2 if hasattr(mx, "get_active_memory") else 0,
                "peak_mb": mx.get_peak_memory() / 1024**2 if hasattr(mx, "get_peak_memory") else 0,
                "cache_mb": mx.get_cache_memory() / 1024**2 if hasattr(mx, "get_cache_memory") else 0,
            }
        except Exception:
            return {"active_mb": 0, "peak_mb": 0, "cache_mb": 0}

    def snapshot(self) -> MemorySnapshot:
        text, page_size = self._vm_stat()
        free = self._pages(text, "Pages free") * page_size
        active = self._pages(text, "Pages active") * page_size
        wired = self._pages(text, "Pages wired down") * page_size
        compressed = self._pages(text, "Pages occupied by compressor") * page_size
        swap = self._swap_usage()
        mlx = self._mlx_memory()
        thermal = self._thermal()

        used = active + wired + compressed
        pressure = min(1.0, used / max(1, self.budget_bytes))

        snap = MemorySnapshot(
            timestamp=time.time(),
            page_size=page_size,
            free_gb=round(free / 1024**3, 4),
            active_gb=round(active / 1024**3, 4),
            wired_gb=round(wired / 1024**3, 4),
            compressed_gb=round(compressed / 1024**3, 4),
            swap_used_gb=round(swap, 4),
            pressure=round(pressure, 6),
            mlx_active_mb=round(mlx["active_mb"], 2),
            mlx_peak_mb=round(mlx["peak_mb"], 2),
            mlx_cache_mb=round(mlx["cache_mb"], 2),
            thermal_pressure=thermal,
        )

        self._swap_history.append(snap.swap_used_gb)
        self._pressure_history.append(snap.pressure)
        if self._baseline is None:
            self._baseline = snap
        return snap

    @property
    def baseline(self) -> Optional[MemorySnapshot]:
        return self._baseline

    def swap_delta(self, snap: MemorySnapshot) -> float:
        if self._baseline is None:
            return 0.0
        return snap.swap_used_gb - self._baseline.swap_used_gb

    def pressure_delta(self, snap: MemorySnapshot) -> float:
        if self._baseline is None:
            return 0.0
        return snap.pressure - self._baseline.pressure

    def swap_growing(self, window: int = 5, threshold: float = 0.05) -> bool:
        if len(self._swap_history) < window:
            return False
        recent = sum(self._swap_history[-3:]) / 3
        older = sum(self._swap_history[-(window+3):-window]) / 3 if len(self._swap_history) >= window + 3 else self._baseline.swap_used_gb if self._baseline else self._swap_history[0]
        return recent > older + threshold

    def pressure_trend(self, window: int = 5) -> float:
        if len(self._pressure_history) < window:
            return 0.0
        recent = sum(self._pressure_history[-3:]) / 3
        older = sum(self._pressure_history[-(window+3):-window]) / 3 if len(self._pressure_history) >= window + 3 else self._pressure_history[0]
        return recent - older

    def clear_mlx_cache(self):
        if _HAS_MLX and hasattr(mx, "clear_cache"):
            mx.clear_cache()


def snapshot_to_dict(snap: MemorySnapshot) -> dict:
    return asdict(snap)
