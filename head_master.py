from src.asteroid import Asteroid
from src.ship import Ship
from center_coords import center_coords
import time
import numpy as np

class HeadMaster:
    def __init__(self):
        self.ship_states = []
        self.asteroid_states = []
        self.mine_states = []

    def get_ship_priority(game_state):
        ships = game_state['ships']
        ship_position = np.array([ship['position'] for ship in ships], dtype=np.float64)
        ship_heading = np.array([ship['heading']for ship in ships], dtype=np.float64)
        ship_velocity = np.array([ship['velocity']for ship in ships], dtype=np.float64)
        ship_speed = np.array([ship['speed']for ship in ships], dtype=np.float64)

        asteroids = game_state['asteroids']
        asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
        asteroid_velocity = np.array([asteroid['velocity'] for asteroid in asteroids], dtype=np.float64)
        asteroid_radii = np.array([asteroid['radius'] for asteroid in asteroids])
        map_size = np.array(game_state['map_size'])
        i= 0
        for ship in ships:
            ship_future_position = ship['position'] + (ship['forecast_frames'] * ship_velocity[i])
            asteroid_future_positions = np.array(asteroid_positions + ship['forecast_frames'] * asteroid_velocity, dtype=np.float64)
            centered_asteroids = center_coords(ship['position'], ship_heading, asteroid_positions, map_size)
            centered_future_asteroids = center_coords(ship_future_position, ship_heading, asteroid_future_positions, map_size)

            radar = get_radar(centered_asteroids, asteroid_radii, ship['radar_zones'])
            forecast = get_radar(centered_future_asteroids, asteroid_radii, ship['radar_zones'])
            bumper = get_bumper(centered_asteroids, asteroid_radii, ship['bumper_range'])
            future_bumper = get_bumper(centered_future_asteroids, asteroid_radii, ship['bumper_range'])

            obs = {
                "radar": radar,
                "forecast": forecast,
                "bumper": bumper,
                "future_bumper": future_bumper,
                "speed": ship_speed,
            }
            print(f"Ship {ship['id']} observation: {obs}")

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