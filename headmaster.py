# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Set
import math

class HeadMaster:
    """
    Décide quels vaisseaux passent en 'black box' et quels restent en 'glass box'.
    Tous les `interval_s` secondes, calcule une valeur de nécessité pour chaque vaisseau
    et assigne les top_k à la black box.
    """
    def __init__(self, top_k: int = 2, interval_s: float = 5.0, w_dist: float = 1.0, w_approach: float = 1.0):
        self.top_k = top_k
        self.interval_s = interval_s
        self.w_dist = w_dist
        self.w_approach = w_approach
        self._last_update_t = -1e9
        self._assigned_black: Set[int] = set()

    def _necessity(self, ship_state: Dict[str, Any], asteroids: List[Dict[str, Any]]) -> float:
        if not asteroids:
            return 0.0
        sx, sy = ship_state["position"]
        svx, svy = ship_state["velocity"]

        # Trouver l'astéroïde le plus proche
        nearest = min(asteroids, key=lambda a: (a["position"][0]-sx)**2 + (a["position"][1]-sy)**2)
        ax, ay = nearest["position"]
        avx, avy = nearest["velocity"]

        dx, dy = ax - sx, ay - sy
        dist = math.sqrt(dx*dx + dy*dy)
        map_w, map_h = 1000.0, 800.0  # TODO: peut être lu depuis game_state
        max_dist = math.sqrt(map_w*map_w + map_h*map_h)
        dist_norm = min(dist / max_dist, 1.0)

        # Closeness : plus proche => score plus élevé
        closeness = 1.0 - dist_norm

        # Approche radiale : positif si l'astéroïde approche
        rel_vx, rel_vy = avx - svx, avy - svy
        if dist > 1e-6:
            approach_speed = - (dx*rel_vx + dy*rel_vy) / dist
        else:
            approach_speed = 0.0
        approach_score = max(0.0, approach_speed / 200.0)  # normalisation

        return self.w_dist * closeness + self.w_approach * approach_score

    def update_assignments(self, game_state: Dict[str, Any]) -> None:
        t = game_state["time"]
        if t - self._last_update_t < self.interval_s:
            return
        self._last_update_t = t

        asteroids = game_state["asteroids"]
        scores = []
        for ship in game_state["ships"]:
            score = self._necessity(ship, asteroids)
            scores.append((score, ship["id"]))
        scores.sort(reverse=True)

        # Top_k en black box
        self._assigned_black = {ship_id for _, ship_id in scores[:self.top_k]}

    def is_black_box(self, ship_id: int) -> bool:
        return ship_id in self._assigned_black
