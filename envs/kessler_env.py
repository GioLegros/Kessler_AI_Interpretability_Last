import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
from collections import deque
from src.center_coords import center_coords
from src.reward.stay_alive import stay_alive_reward

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
    # For now, we are assuming only one ship (ours)
    ship = game_state['ships'][0]
    asteroids = game_state['asteroids']

    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids])
    rho, phi, x, y = center_coords(ship['position'], ship['heading'], asteroid_positions)

    asteroid_velocities = np.array([asteroid['velocity'] for asteroid in asteroids])
    asteroid_velocities_relative = asteroid_velocities - ship['velocity']
    asteroid_speed_relative = np.linalg.norm(asteroid_velocities_relative, axis=1)

    asteroid_info = np.stack([
        rho, phi, asteroid_speed_relative
    ], axis=1)

    # Sort by first column (distance)
    asteroid_info = asteroid_info[
        asteroid_info[:, 0].argsort()
    ]

    # Pad
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

    return obs


class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"
