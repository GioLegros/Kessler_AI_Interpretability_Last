# -*- coding: utf-8 -*-
# Simple "glass box" flee controller for Kessler.
# Places the ship's nose directly away from the nearest asteroid and thrusts forward.
from __future__ import annotations
from typing import Dict, Any, Tuple
import math

from src.controller import KesslerController

THRUST_MAX = 480.0
TURN_MAX = 180.0

class GlassBoxFleeController(KesslerController):
    """Heuristic controller: flee from the nearest asteroid.
    - Aim the ship in the exact opposite direction of the nearest asteroid
    - Use proportional turning to align heading quickly
    - Thrust forward when roughly facing the flee direction
    - Never fire or drop mines
    """

    def __init__(self, align_eps_deg: float = 10.0, k_turn: float = 4.0, k_slow: float = 0.4) -> None:
        # k_turn: proportional turn gain (deg/s per deg of error, clamped at TURN_MAX)
        # k_slow: reduce thrust when current speed is high to avoid overshoot
        self._ship_id = 0
        self.align_eps_deg = align_eps_deg
        self.k_turn = k_turn
        self.k_slow = k_slow

    @property
    def name(self) -> str:
        return "GlassBox Flee"

    def actions(self, ship_state: Dict[str, Any], game_state) -> Tuple[float, float, bool, bool]:
        ship_x, ship_y = ship_state["position"]
        heading = ship_state["heading"]
        vx, vy = ship_state["velocity"]

        # Choose nearest asteroid
        nearest = None
        nearest_d2 = float("inf")
        for ast in game_state["asteroids"]:
            ax, ay = ast["position"]
            dx, dy = ax - ship_x, ay - ship_y
            d2 = dx*dx + dy*dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest = (dx, dy)

        # If no asteroid, do nothing
        if nearest is None:
            return 0.0, 0.0, False, False

        # Desired flee direction is opposite of vector to nearest asteroid
        dx, dy = nearest
        flee_angle = math.degrees(math.atan2(-dy, -dx))  # point away
        # Compute shortest signed heading error
        err = (flee_angle - heading + 540.0) % 360.0 - 180.0  # in [-180,180)
        turn_rate = max(-TURN_MAX, min(TURN_MAX, self.k_turn * err))

        # Thrust logic: thrust mainly when roughly aligned with flee direction
        aligned = abs(err) < self.align_eps_deg
        # Slow down thrust as speed grows (simple proportional)
        speed = (vx*vx + vy*vy) ** 0.5
        thrust_scale = max(0.0, 1.0 - self.k_slow * (speed / ship_state["max_speed"]))
        thrust = THRUST_MAX * thrust_scale if aligned else 0.0

        return thrust, turn_rate, False, False
