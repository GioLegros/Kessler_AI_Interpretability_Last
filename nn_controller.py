# -*- coding: utf-8 -*-
# Inference-time controller that loads a Stable-Baselines3 policy and outputs thrust/turn.
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import os
import numpy as np

from src.controller import KesslerController
from env_gym import KesslerFleeEnv, THRUST_MAX, TURN_MAX

class NeuralFleeController(KesslerController):
    """Controller that runs a SB3 policy trained on KesslerFleeEnv to flee asteroids."""
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._ship_id = 0
        self.model_path = model_path
        self.device = device
        # Lazy import to avoid hard dependency if user doesn't use it
        from stable_baselines3 import PPO
        self.PPO = PPO
        self._policy = None

    @property
    def name(self) -> str:
        return "Neural Flee (SB3/PPO)"

    def _load(self):
        if self._policy is None:
            self._policy = self.PPO.load(self.model_path, device=self.device)

    def actions(self, ship_state, game_state):
        self._load()

        # Encode l'observation SANS dimension batch
        if not hasattr(self, "_obs_encoder"):
            self._obs_encoder = KesslerFleeEnv(scenario=None, time_limit=game_state["time_limit"])

        fake_gs = {
            "ships": [ship_state],
            "asteroids": game_state["asteroids"],
            "map_size": game_state["map_size"],
            "time": game_state["time"]
        }
        obs = self._obs_encoder._normalize(fake_gs)  # <-- PAS de [None, :]

        action, _ = self._policy.predict(obs, deterministic=True)

        # Aplatit/force en 1D au cas où SB3 renverrait (1,2)
        action = np.asarray(action).reshape(-1)

        thrust = float(np.clip(action[0], -1.0, 1.0) * THRUST_MAX)
        turn   = float(np.clip(action[1], -1.0, 1.0) * TURN_MAX)
        return thrust, turn, False, False
