# -*- coding: utf-8 -*-
# Gymnasium wrapper around the Kessler TrainerEnvironment for RL.
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from src.kessler_game import TrainerEnvironment, StopReason
from src.scenario import Scenario
from src.controller import KesslerController

THRUST_MAX = 480.0
TURN_MAX = 180.0
ASTEROID_MAX_SPEED = 180.0
SHIP_MAX_SPEED = 240.0
N_CLOSEST_ASTEROIDS = 5


class _RLController(KesslerController):
    """A thin controller that takes actions provided by the RL agent via the env."""
    def __init__(self) -> None:
        self._ship_id = 0
        self._pending_action: Tuple[float, float] = (0.0, 0.0)

    @property
    def name(self) -> str:
        return "RLController"

    def set_action(self, thrust: float, turn_rate: float) -> None:
        self._pending_action = (thrust, turn_rate)

    def actions(self, ship_state: Dict[str, Any], game_state) -> Tuple[float, float, bool, bool]:
        thrust, turn = self._pending_action
        return float(thrust), float(turn), False, False  # no firing/mines during training


class KesslerFleeEnv(gym.Env):
    """
    Observation (Box):
      [ ship_x_norm, ship_y_norm, ship_vx_norm, ship_vy_norm, cos_h, sin_h,
        for each of k nearest asteroids:
          rel_x_norm, rel_y_norm, rel_vx_norm, rel_vy_norm, dist_norm, radius_norm ]
    Action (Box):
      thrust in [-1,1] -> scaled to [-THRUST_MAX, THRUST_MAX]
      turn   in [-1,1] -> scaled to [-TURN_MAX, TURN_MAX]
    Reward (améliorée):
      +0.05 par step pour être en vie
      +1.0 * (dist_norm ** 2)       (récompense d’être loin; accentue quand on est très loin)
      +0.3 * (dist_norm - last_dist) (bonus si on S’ÉLOIGNE du plus proche entre deux frames)
      -5.0 en cas de mort (terminated)
    Episode: se termine à death, ou se tronque à time_limit secondes.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 scenario: Optional[Scenario] = None,
                 time_limit: float = 60.0,         # ← plus long par défaut
                 seed: Optional[int] = None) -> None:
        super().__init__()
        self.n_ast = N_CLOSEST_ASTEROIDS
        self.time_limit = time_limit

        # On garde un scénario “gabarit”, mais on régénère à chaque reset pour diversifier
        self.base_map_size = (1000, 800)
        if scenario is None:
            scenario = Scenario(
                name="RL-Flee",
                num_asteroids=6,
                ship_states=[{"position": (self.base_map_size[0]/2, self.base_map_size[1]/2)}],
                map_size=self.base_map_size,
                time_limit=time_limit
            )
        self.base_scenario = scenario

        self.game = TrainerEnvironment(settings={"frequency": 30.0, "time_limit": time_limit})
        self.controller = _RLController()
        self.generator = None

        # Observation space
        ship_low = np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32)  # x,y,vx,vy,cos,sin
        ship_high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        ast_low = np.array([-1, -1, -1, -1, 0, 0], dtype=np.float32)     # rel x/y/vx/vy [-1,1], dist∈[0,1], radius∈[0,1]
        ast_high = np.array([ 1,  1,  1,  1, 1, 1], dtype=np.float32)
        obs_low = np.concatenate([ship_low] + [ast_low for _ in range(self.n_ast)])
        obs_high = np.concatenate([ship_high] + [ast_high for _ in range(self.n_ast)])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([ 1.0,  1.0], dtype=np.float32),
                                       dtype=np.float32)

        self._last_lives = 0
        self._sim_time = 0.0
        self._step_dt = 1.0/30.0  # from TrainerEnvironment default
        self._last_dist = 0.0     # pour le bonus d’éloignement

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # -------- Normalisation / encodage --------
    def _normalize(self, game_state) -> np.ndarray:
        ship = game_state["ships"][0]
        map_w, map_h = game_state["map_size"]
        max_dist = math.sqrt(map_w*map_w + map_h*map_h)

        sx, sy = ship["position"]
        vx, vy = ship["velocity"]
        heading = ship["heading"]
        cos_h = math.cos(math.radians(heading))
        sin_h = math.sin(math.radians(heading))

        ship_vec = np.array([sx/map_w*2-1,
                             sy/map_h*2-1,
                             vx/SHIP_MAX_SPEED,
                             vy/SHIP_MAX_SPEED,
                             cos_h, sin_h], dtype=np.float32)

        # nearest asteroids
        asts = game_state["asteroids"]
        dists = []
        for a in asts:
            ax, ay = a["position"]
            dx, dy = ax - sx, ay - sy
            d2 = dx*dx + dy*dy
            dists.append((d2, a))
        dists.sort(key=lambda t: t[0])
        nearest = [t[1] for t in dists[:self.n_ast]]

        parts: List[np.ndarray] = [ship_vec]
        for a in nearest:
            ax, ay = a["position"]
            avx, avy = a["velocity"]
            dx, dy = ax - sx, ay - sy
            rel_vx, rel_vy = avx - vx, avy - vy
            dist = math.sqrt(dx*dx + dy*dy)
            parts.append(np.array([
                (dx / max_dist) * 2,
                (dy / max_dist) * 2,
                rel_vx / (ASTEROID_MAX_SPEED + SHIP_MAX_SPEED),
                rel_vy / (ASTEROID_MAX_SPEED + SHIP_MAX_SPEED),
                min(dist / max_dist, 1.0),
                min(a["radius"] / 64.0, 1.0)
            ], dtype=np.float32))
        while len(parts) < 1 + self.n_ast:
            parts.append(np.zeros(6, dtype=np.float32))
        obs = np.concatenate(parts, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_dist_norm(self, game_state) -> float:
        """Distance normalisée au plus proche astéroïde."""
        ship = game_state["ships"][0]
        map_w, map_h = game_state["map_size"]
        max_dist = math.sqrt(map_w*map_w + map_h*map_h)
        if not game_state["asteroids"]:
            return 1.0
        sx, sy = ship["position"]
        nearest = min(((a["position"][0]-sx)**2 + (a["position"][1]-sy)**2) for a in game_state["asteroids"]) ** 0.5
        return min(nearest / max_dist, 1.0)

    def _randomized_scenario(self) -> Scenario:
        """Nouveau scénario à chaque reset: spawn et heading aléatoires, astéroïdes aléatoires."""
        W, H = self.base_map_size
        # vaisseau aléatoire
        sx = random.uniform(W*0.2, W*0.8)
        sy = random.uniform(H*0.2, H*0.8)
        heading = random.uniform(0.0, 360.0)
        ship_state = {"position": (sx, sy), "angle": heading}

        # nombre d’astéroïdes (6 à 10) pour varier la densité
        n_ast = random.randint(6, 10)

        return Scenario(
            name="RL-Flee",
            num_asteroids=n_ast,       # positions random internes
            ship_states=[ship_state],
            map_size=self.base_map_size,
            time_limit=self.time_limit
        )

    # -------- Gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Nouveau scénario à chaque reset
        self.scenario = self._randomized_scenario()

        self.game = TrainerEnvironment(settings={"frequency": 30.0, "time_limit": self.time_limit})
        self.controller = _RLController()
        self.generator = self.game.run_step(self.scenario, [self.controller])

        # état initial (avant toute action) — compatible Gym
        score, perf, game_state = next(self.generator)
        self._last_lives = game_state["ships"][0]["lives_remaining"]
        self._sim_time = 0.0
        self._last_dist = self._compute_dist_norm(game_state)
        obs = self._normalize(game_state)
        return obs, {}

    def step(self, action: np.ndarray):
        # Scale actions
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        thrust = float(a[0]) * THRUST_MAX
        turn = float(a[1]) * TURN_MAX
        self.controller.set_action(thrust, turn)

        terminated = False
        truncated = False

        # Avancer une frame — gérer la fin d’épisode proprement
        try:
            score, perf, game_state = next(self.generator)
        except StopIteration as e:
            # Si le générateur s’achève (limite de temps atteinte côté moteur)
            if hasattr(e, "value") and e.value:
                score, perf, game_state = e.value
            else:
                # Fallback: faux game_state minimal si jamais
                game_state = {"ships": [{"lives_remaining": self._last_lives, "position": (0,0), "velocity": (0,0), "heading": 0}],
                              "asteroids": [], "map_size": self.base_map_size, "time": self.time_limit}
            truncated = True  # fin “time limit”

        obs = self._normalize(game_state)

        # Reward shaping (fuite)
        dist_norm = self._compute_dist_norm(game_state)
        delta_dist = dist_norm - self._last_dist
        self._last_dist = dist_norm

        # base alive + loin + s’éloigne
        reward = 0.05 + 1.0 * (dist_norm ** 2) + 0.3 * (delta_dist)

        # Terminaison par mort
        ship = game_state["ships"][0]
        lives = ship["lives_remaining"]
        if lives < self._last_lives:
            reward -= 5.0
            terminated = True
        self._last_lives = lives

        # Troncature par limite de temps
        self._sim_time = game_state.get("time", self._sim_time + self._step_dt)
        if (not terminated) and (self._sim_time >= self.time_limit - 1e-6):
            truncated = True

        info = {"sim_time": self._sim_time, "dist_norm": dist_norm}
        return obs, reward, terminated, truncated, info
