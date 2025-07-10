import numpy as np

from src.center_coords import center_coords


def parse_game_state(ship_state, game_state, forecast_seconds=1):
    ship_position = np.array(ship_state['position'], dtype=np.float64)
    ship_heading = np.radians(ship_state['heading'])
    ship_velocity = np.array(ship_state['velocity'], dtype=np.float64)
    ship_speed = np.array([ship_state['speed']])

    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
    asteroid_velocity = np.array([asteroid['velocity'] for asteroid in asteroids], dtype=np.float64)
    asteroid_radii = np.array([asteroid['radius'] for asteroid in asteroids])

    map_size = np.array(game_state['map_size'])

    ship_future_position = np.mod(ship_position + (forecast_seconds * ship_velocity), map_size)
    asteroid_future_positions = np.mod(asteroid_positions + (forecast_seconds * asteroid_velocity), map_size)

    centered_asteroids = center_coords(ship_position, ship_heading, asteroid_positions, map_size)
    centered_future_asteroids = center_coords(ship_future_position, ship_heading, asteroid_future_positions, map_size)

    return {
        'ship': {
            'ship_position': ship_position,
            'future_position': ship_future_position,
            'ship_heading': ship_heading,
            'ship_velocity': ship_velocity,
            'ship_speed': ship_speed,
            'is_respawning': ship_state['is_respawning'],
        },
        'asteroids': {
            'xy_positions': asteroid_positions,
            'xy_future_positions': asteroid_future_positions,
            'xy_velocity': asteroid_velocity,
            'polar_positions': centered_asteroids,
            'polar_future_positions': centered_future_asteroids,
            'radii': asteroid_radii,
            'python_obj': game_state['asteroids'],
        },
        'game': {
            'map_size': map_size,
            'time': game_state['time'],
            'delta_time': game_state['delta_time']
        }
    }
