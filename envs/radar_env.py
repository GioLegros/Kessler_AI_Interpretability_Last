import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController
from typing import Dict, Tuple
from collections import deque
from src.center_coords import center_coords

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
SHIP_MAX_SPEED = 240
DEFAULT_RADAR_ZONES = [100, 250, 400]
DEFAULT_BUMPER_RANGE = 50
DEFAULT_FORECAST_FRAMES = 30


## TO TEST / VERIFY:
# - Effect of adding / removing ship speed
# - Effect of subtracting asteroid radius from rho when building radar
# - If time... extra radar / bumper zones tests (using an objective benchmark instead of reward function...)

class RadarEnv(gym.Env):
    def __init__(self, scenario, radar_zones=None, bumper_range=DEFAULT_BUMPER_RANGE,
                 forecast_frames=DEFAULT_FORECAST_FRAMES):
        if radar_zones is None:
            self.radar_zones = DEFAULT_RADAR_ZONES
        else:
            self.radar_zones = radar_zones
        self.bumper_range = bumper_range
        self.forecast_frames = forecast_frames

        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = scenario
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])

        self.observation_space = spaces.Dict(
            {
                # Radar: Density of asteroids in each zone
                "radar": spaces.Box(low=0, high=1, shape=(12,)),
                "forecast": spaces.Box(low=0, high=1, shape=(12,)),
                # Bumper: 0/1 indicator for if any asteroid hitbox encroaches within the bumper_zone
                "bumper": spaces.Box(low=0, high=1, shape=(4,)),
                "future_bumper": spaces.Box(low=0, high=1, shape=(4,)),

                # Ship speed
                #"speed": spaces.Box(low=0, high=SHIP_MAX_SPEED, shape=(1,))
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])
        score, perf_list, game_state = next(self.game_generator)
        obs = get_obs(game_state, forecast_frames=self.forecast_frames, radar_zones=self.radar_zones,
                      bumper_range=self.bumper_range)
        return obs, self._get_info()

    def step(self, action):
        thrust, turn_rate, fire, drop_mine = action[0] * THRUST_SCALE, action[1] * TURN_SCALE, False, False
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))
        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            score, perf_list, game_state = list(exp.args[0])
            terminated = True
        obs = get_obs(game_state, forecast_frames=self.forecast_frames, radar_zones=self.radar_zones,
                      bumper_range=self.bumper_range)
        reward = get_reward(game_state)
        return obs, reward, terminated, False, self._get_info()

    def _get_info(self):
        return {}


def get_obs(game_state, forecast_frames, radar_zones, bumper_range):
    ship = game_state['ships'][0]
    ship_position = np.array(ship['position'], dtype=np.float64)
    ship_heading = np.radians(ship['heading'])
    ship_velocity = np.array(ship['velocity'], dtype=np.float64)
    ship_speed = np.array([ship['speed']])

    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
    asteroid_velocity = np.array([asteroid['velocity'] for asteroid in asteroids], dtype=np.float64)
    asteroid_radii = np.array([asteroid['radius'] for asteroid in asteroids])
    map_size = np.array(game_state['map_size'])

    ship_future_position = ship_position + (forecast_frames * ship_velocity)
    asteroid_future_positions = asteroid_positions + (forecast_frames * asteroid_velocity)

    centered_asteroids = center_coords(ship_position, ship_heading, asteroid_positions, map_size)
    centered_future_asteroids = center_coords(ship_future_position, ship_heading, asteroid_future_positions, map_size)

    radar = get_radar(centered_asteroids, asteroid_radii, radar_zones)
    forecast = get_radar(centered_future_asteroids, asteroid_radii, radar_zones)
    bumper = get_bumper(centered_asteroids, asteroid_radii, bumper_range)
    future_bumper = get_bumper(centered_future_asteroids, asteroid_radii, bumper_range)

    obs = {
        "radar": radar,
        "forecast": forecast,
        "bumper": bumper,
        "future_bumper": future_bumper,
        #"speed": ship_speed,
    }

    return obs


def get_radar(centered_asteroids, asteroid_radii, radar_zones):
    asteroid_areas = np.pi * asteroid_radii * asteroid_radii
    rho, phi = centered_asteroids[:, 0], centered_asteroids[:, 1]

    #rho -= asteroid_radii
    is_near = rho < radar_zones[0]
    is_medium = np.logical_and(rho < radar_zones[1], rho >= radar_zones[0])
    is_far = np.logical_and(rho < radar_zones[2], rho >= radar_zones[1])

    is_front = np.logical_or(phi < 0.25 * np.pi, phi >= 1.75 * np.pi)
    is_left = np.logical_and(phi < 0.75 * np.pi, phi >= 0.25 * np.pi)
    is_behind = np.logical_and(phi < 1.25 * np.pi, phi >= 0.75 * np.pi)
    is_right = np.logical_and(phi < 1.75 * np.pi, phi >= 1.25 * np.pi)

    inner_area = np.pi * radar_zones[0] * radar_zones[0]
    middle_area = np.pi * radar_zones[1] * radar_zones[1]
    outer_area = np.pi * radar_zones[2] * radar_zones[2]
    # The area of one slice in the outer, middle, and inner donuts
    slice_areas = [(outer_area - (middle_area + inner_area)) / 4, (middle_area - inner_area) / 4, inner_area / 4]

    radar_info = np.zeros(shape=(12,))
    for idx, distance_mask in enumerate([is_far, is_medium, is_near]):
        slice_area = slice_areas[idx]
        for jdx, angle_mask in enumerate([is_front, is_left, is_behind, is_right]):
            mask = np.logical_and(distance_mask, angle_mask)
            total_asteroid_area = np.sum(asteroid_areas[mask])
            index = idx * 4 + jdx
            radar_info[index] = min(1, total_asteroid_area / slice_area)

    return radar_info

def get_bumper(centered_asteroids, asteroid_radii, bumper_range):
    rho, phi = centered_asteroids[:, 0], centered_asteroids[:, 1]

    rho -= asteroid_radii
    bumper_hit = rho < bumper_range
    is_front = np.logical_or(phi < 0.25 * np.pi, phi >= 1.75 * np.pi)
    is_left = np.logical_and(phi < 0.75 * np.pi, phi >= 0.25 * np.pi)
    is_behind = np.logical_and(phi < 1.25 * np.pi, phi >= 0.75 * np.pi)
    is_right = np.logical_and(phi < 1.75 * np.pi, phi >= 1.25 * np.pi)

    bumper = np.zeros(shape=(4,))
    for jdx, angle_mask in enumerate([is_front, is_left, is_behind, is_right]):
        hit = np.any(np.logical_and(bumper_hit, angle_mask)).astype(int)
        bumper[jdx] = hit

    return bumper


def get_reward(game_state):
    # It seems best if the majority of the reward comes from simply staying alive,
    # and let reinforcement learning figure out how best to actually do that.
    # However, we do want to "gently" guide the ship to sparse areas -- if any exist.

    ship = game_state['ships'][0]
    ship_position = np.array(ship['position'], dtype=np.float64)
    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
    dist = np.min(np.linalg.norm(asteroid_positions - ship_position, axis=1))
    return (dist * dist) / 1000

class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"
