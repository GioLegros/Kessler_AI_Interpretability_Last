# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional

class EventLogger:
    def __init__(self, path: str = "logs/sim.log") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # header
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"# --- NEW SESSION {datetime.now().isoformat(timespec='seconds')} ---\n")

    def _ts(self, sim_time: float) -> str:
        return f"[t={sim_time:6.2f}s]"

    def _mode_of(self, ship) -> str:
        # essaie de lire le mode courant depuis HybridController; fallback au name()
        try:
            ctrl = ship.controller
            if hasattr(ctrl, "_last_mode"):
                return str(ctrl._last_mode)
            # sinon, parse le name
            name = str(ctrl.name)
            if "Hybrid[" in name:
                return name.split("Hybrid[")[-1].split("]")[0].lower()
            return name
        except Exception:
            return "unknown"

    def log_ship_hit_asteroid(self, sim_time: float, ship) -> None:
        mode = self._mode_of(ship)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{self._ts(sim_time)} SHIP_HIT_ASTEROID ship={ship.id:02d} mode={mode}\n")

    def log_ship_death(self, sim_time: float, ship) -> None:
        mode = self._mode_of(ship)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{self._ts(sim_time)} SHIP_DEAD         ship={ship.id:02d} mode={mode} lives_left={ship.lives}\n")

    def log_sim_stop(self, sim_time: float, stop_reason: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{self._ts(sim_time)} SIM_STOP         reason={stop_reason}\n")

# petit singleton pratique
_logger_singleton: Optional[EventLogger] = None

def get_logger(path: Optional[str] = None) -> EventLogger:
    global _logger_singleton
    if _logger_singleton is None:
        _logger_singleton = EventLogger(path or "logs/sim.log")
    return _logger_singleton
