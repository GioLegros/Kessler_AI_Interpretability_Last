import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.controller import KesslerController
from src.kessler_game import KesslerGame, StopReason, TrainerEnvironment
from src.scenario import Scenario
from typing import Dict, Tuple
from collections import deque
from center_coords import center_coords
from reward.stay_alive import stay_alive_reward

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
ASTEROID_MAX_SPEED = 180
SHIP_MAX_SPEED = 240
N_CLOSEST_ASTEROIDS = 5

class KesslerEnv(gym.Env):
    def __init__(self, scenario):
        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = scenario
        self.reward_function = stay_alive_reward
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])
        self.prev_state, self.current_state = None, None

        max_dist = np.sqrt(scenario.map_size[0] * scenario.map_size[1]) / 2
        max_rel = ASTEROID_MAX_SPEED + SHIP_MAX_SPEED
        self.observation_space = spaces.Dict(
            {
                "asteroid_dist": spaces.Box(low=0, high=max_dist, shape=(N_CLOSEST_ASTEROIDS,)),
                "asteroid_angle": spaces.Box(low=0, high=360, shape=(N_CLOSEST_ASTEROIDS,)),
                "asteroid_rel_speed": spaces.Box(low=-1 * max_rel, high=max_rel, shape=(N_CLOSEST_ASTEROIDS,)),
                "ship_heading": spaces.Box(low=0, high=360, shape=(1,)),
                "ship_speed": spaces.Box(low=0, high=SHIP_MAX_SPEED, shape=(1,)),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])
        score, perf_list, game_state = next(self.game_generator)

        # Vérification de l'état du jeu
        if 'ships' not in game_state or 'asteroids' not in game_state:
            raise ValueError("État du jeu mal formé")

        self.prev_state, self.current_state = None, game_state
        return get_obs(game_state), self._get_info()

    def step(self, action):
        # Just always fire, for now...
        thrust, turn_rate, fire, drop_mine = action[0] * THRUST_SCALE, action[1] * TURN_SCALE, False, False
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))

        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            print("StopIteration capturée")
            score, perf_list, game_state = list(exp.args[0])
            terminated = True

        self.update_state(game_state)
        return get_obs(game_state), self.reward_function(game_state, self.prev_state), terminated, False, self._get_info()

    def update_state(self, game_state):
        self.prev_state = self.current_state
        self.current_state = game_state

    def _get_info(self):
        return {}

def get_obs(game_state):
    # Pour l'instant, on suppose qu'il n'y a qu'un seul vaisseau (le nôtre)
    ship = game_state['ships'][0]
    asteroids = game_state['asteroids']
    map_size = game_state['map_size']

    # Vérification des données du vaisseau
    if np.any(np.isnan(ship['position'])) or np.any(np.isnan(ship['heading'])):
        raise ValueError("NaN dans les données du vaisseau")

    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids])

    # Vérification des positions des astéroïdes
    if np.any(np.isnan(asteroid_positions)):
        raise ValueError("NaN dans les positions des astéroïdes")

    coords = center_coords(ship['position'], ship['heading'], asteroid_positions, map_size)

    if coords.shape[1] == 2:
        rho, phi = coords[:, 0], coords[:, 1]
        x, y = None, None
    elif coords.shape[1] == 4:
        rho, phi, x, y = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]

    asteroid_velocities = np.array([asteroid['velocity'] for asteroid in asteroids])

    # Vérification des vitesses des astéroïdes
    if np.any(np.isnan(asteroid_velocities)):
        raise ValueError("NaN dans les vitesses des astéroïdes")

    asteroid_velocities_relative = asteroid_velocities - ship['velocity']
    asteroid_speed_relative = np.linalg.norm(asteroid_velocities_relative, axis=1)

    asteroid_info = np.stack([rho, phi, asteroid_speed_relative], axis=1)

    # Si des NaN sont détectés, on les transforme en zéro et on affiche un message de débogage
    if np.any(np.isnan(asteroid_info)):
        print("NaN détecté dans les informations des astéroïdes.")
        asteroid_info = np.nan_to_num(asteroid_info)

    # Trier par la première colonne (distance)
    asteroid_info = asteroid_info[asteroid_info[:, 0].argsort()]

    # Padding
    padding_len = N_CLOSEST_ASTEROIDS - asteroid_info.shape[0]
    if padding_len > 0:
        pad_shape = (padding_len, asteroid_info.shape[1])
        asteroid_info = np.concatenate([asteroid_info, np.empty(pad_shape)])

    obs = {
        "asteroid_dist": asteroid_info[:N_CLOSEST_ASTEROIDS, 0],
        "asteroid_angle": asteroid_info[:N_CLOSEST_ASTEROIDS, 1],
        "asteroid_rel_speed": asteroid_info[:N_CLOSEST_ASTEROIDS, 2],
        "ship_heading": np.array([ship["heading"]]),
        "ship_speed": np.array([ship["speed"]]),
    }

    coords = center_coords(ship['position'], ship['heading'], asteroid_positions, map_size)

    # Vérification finale des observations avant de les retourner
    if np.any(np.isnan(obs["asteroid_dist"])) or np.any(np.isnan(obs["asteroid_angle"])) or np.any(np.isnan(obs["asteroid_rel_speed"])) or np.any(np.isnan(coords)):
        raise ValueError("NaN détecté dans les observations avant de les retourner.")

    return obs


class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Si la file est vide, on ajoute une action par défaut
        if not self.action_queue:
            self.action_queue.append((0.0, 0.0, False, False))  # Action par défaut
            print("File d'actions vide, ajout d'une action par défaut")

        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"
