from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerGame, KesslerController, Scenario

from src.lib import parse_game_state
from src.rule2string import rule2string

FUZZY_SETS = [
    # Triangular fuzzy sets can be defined as functions of their center and radius
    lambda x: np.ones_like(x),  # Don't care
    lambda x: np.maximum(1 - np.abs((0.000 - x) / 0.125), 0),  # Low
    lambda x: np.maximum(1 - np.abs((0.125 - x) / 0.125), 0),  # Medium-Low
    lambda x: np.maximum(1 - np.abs((0.250 - x) / 0.125), 0),  # Medium
    lambda x: np.maximum(1 - np.abs((0.375 - x) / 0.125), 0),  # Medium-High
    lambda x: np.maximum(1 - np.abs((0.500 - x) / 0.125), 0),  # High
    lambda x: (x < 0.01).astype(int),  # Clear
    lambda x: (x > 0.01).astype(int),  # Dangerous
]

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

def get_compatibility(decoded_rule, observation):
    antecedents = decoded_rule[:, :-2]
    ob2 = np.broadcast_to(observation, shape=antecedents.shape)
    compat = np.ones_like(antecedents, dtype=np.float64)
    for i in range(4):
        mask = antecedents == i
        lmb = FUZZY_SETS[i]
        compat[mask] = lmb(ob2[mask])
    return np.prod(compat, axis=-1)

def get_winner_rule(ruleset, observation):
    compatibility = get_compatibility(ruleset, observation)
    weights = rule_weights(ruleset)
    weighted_compatibility = compatibility * weights
    winner_idx = np.argmax(weighted_compatibility)
    return ruleset[winner_idx]

def get_output(ruleset, observation):
    compatibility = get_compatibility(ruleset, observation)
    weights = rule_weights(ruleset)
    weighted_compatibility = compatibility * weights
    winner_idx = np.argmax(weighted_compatibility)
    return ruleset[winner_idx, -2:]

def parse_output(fuzzy_output):
    thrust_out = fuzzy_output[0]  # Range 0 - 5
    thrust = (thrust_out - 2) * 240

    turn_out = fuzzy_output[1]  # Range 0 - 5
    turn = (turn_out - 2) * 90  # Range (-2, 2) * 90 = (-180, 180)

    return thrust, turn

def rule_weights(rules):
    # In this case, all fuzzy sets have the same area
    nonzero = np.count_nonzero(rules, axis=-1)
    a = 0.5 ** nonzero
    return 1 - a

class GlobalController(KesslerController):
    def __init__(self):
        self.rules = np.array([
        [5, 7, 3, 6, 5, 4, 5, 2, 7, 5, 5, 0, 0, 0, ],
        [2, 7, 2, 6, 1, 1, 7, 7, 3, 5, 5, 1, 0, 0, ],
        [4, 0, 7, 0, 7, 5, 5, 0, 2, 7, 7, 1, 0, 0, ],
        [7, 3, 5, 6, 5, 0, 6, 7, 0, 4, 3, 2, 0, 0, ],
        [0, 2, 3, 1, 5, 0, 1, 7, 7, 4, 0, 5, 0, 0, ],
        [5, 5, 5, 5, 4, 7, 1, 1, 6, 4, 6, 5, 0, 0, ],
        [1, 4, 7, 1, 3, 6, 7, 5, 6, 4, 6, 5, 0, 0, ],
        [2, 4, 5, 2, 5, 0, 6, 5, 3, 0, 3, 0, 1, 0, ],
        [6, 4, 3, 6, 3, 5, 2, 5, 6, 7, 4, 2, 1, 0, ],
        [7, 2, 2, 3, 3, 3, 7, 7, 4, 1, 1, 6, 1, 0, ],
        [5, 3, 4, 1, 7, 5, 3, 3, 0, 5, 2, 6, 1, 0, ],
        [0, 3, 0, 7, 5, 6, 0, 0, 7, 4, 7, 6, 1, 0, ],
        [1, 2, 3, 5, 6, 4, 3, 7, 5, 5, 2, 2, 2, 0, ],
        [6, 1, 1, 1, 3, 7, 2, 5, 6, 4, 0, 3, 2, 0, ],
        [0, 1, 4, 0, 6, 6, 2, 7, 1, 1, 7, 4, 2, 0, ],
        [5, 4, 7, 0, 5, 0, 4, 3, 2, 2, 7, 6, 2, 0, ],
        [7, 5, 2, 3, 2, 4, 7, 7, 1, 0, 2, 0, 3, 0, ],
        [6, 3, 5, 3, 7, 6, 4, 5, 1, 1, 2, 0, 3, 0, ],
        [1, 7, 4, 1, 5, 1, 0, 5, 7, 1, 0, 1, 3, 0, ],
        [5, 3, 5, 3, 6, 0, 2, 4, 3, 7, 2, 1, 3, 0, ],
        [0, 1, 1, 2, 3, 4, 0, 6, 0, 7, 5, 2, 3, 0, ],
        [6, 1, 1, 6, 3, 1, 5, 3, 1, 5, 7, 2, 3, 0, ],
        [7, 5, 4, 6, 2, 6, 4, 5, 3, 1, 1, 5, 3, 0, ],
        [5, 7, 4, 3, 1, 6, 6, 2, 1, 3, 3, 5, 3, 0, ],
        [4, 4, 5, 1, 3, 7, 6, 3, 1, 6, 1, 6, 3, 0, ],
        [1, 6, 3, 5, 2, 3, 3, 7, 5, 1, 6, 7, 3, 0, ],
        [0, 5, 4, 0, 3, 0, 5, 4, 0, 3, 0, 0, 4, 0, ],
        [7, 0, 1, 5, 7, 6, 5, 3, 0, 0, 4, 3, 4, 0, ],
        [4, 3, 1, 7, 7, 2, 1, 7, 7, 7, 2, 4, 4, 0, ],
        [6, 1, 5, 3, 1, 0, 4, 6, 1, 5, 0, 5, 4, 0, ],
        [3, 5, 6, 7, 7, 6, 0, 3, 2, 7, 4, 7, 4, 0, ],
        [5, 4, 3, 0, 7, 6, 1, 3, 4, 5, 3, 3, 0, 1, ],
        [2, 0, 5, 7, 0, 6, 5, 1, 0, 7, 6, 3, 0, 1, ],
        [6, 6, 1, 5, 7, 4, 2, 1, 6, 6, 4, 4, 0, 1, ],
        [2, 3, 6, 6, 7, 2, 7, 3, 3, 4, 5, 4, 0, 1, ],
        [1, 6, 5, 3, 4, 2, 0, 2, 7, 5, 7, 4, 0, 1, ],
        [5, 6, 5, 7, 7, 7, 6, 7, 3, 4, 6, 6, 0, 1, ],
        [5, 1, 0, 5, 4, 2, 6, 6, 3, 3, 4, 0, 1, 1, ],
        [5, 5, 1, 4, 2, 3, 5, 3, 3, 6, 7, 2, 1, 1, ],
        [2, 2, 7, 2, 5, 6, 6, 1, 1, 0, 4, 4, 1, 1, ],
        [2, 7, 1, 2, 3, 1, 3, 1, 4, 3, 1, 0, 2, 1, ],
        [2, 1, 3, 5, 1, 0, 0, 1, 0, 2, 1, 1, 2, 1, ],
        [2, 5, 5, 4, 5, 3, 7, 6, 3, 2, 0, 2, 2, 1, ],
        [0, 4, 6, 7, 7, 4, 7, 5, 4, 0, 7, 2, 2, 1, ],
        [6, 1, 3, 3, 7, 5, 2, 7, 2, 3, 0, 3, 2, 1, ],
        [3, 7, 5, 6, 4, 2, 1, 1, 7, 3, 5, 3, 2, 1, ],
        [7, 5, 2, 0, 4, 0, 5, 4, 2, 7, 0, 4, 2, 1, ],
        [1, 5, 4, 2, 1, 1, 6, 2, 2, 4, 4, 4, 2, 1, ],
        [7, 0, 3, 2, 5, 2, 7, 7, 1, 2, 6, 4, 2, 1, ],
        [2, 7, 2, 0, 0, 3, 1, 7, 3, 6, 4, 6, 2, 1, ],
        [0, 6, 1, 4, 7, 3, 2, 3, 2, 6, 1, 3, 3, 1, ],
        [7, 4, 1, 4, 7, 6, 6, 7, 1, 6, 2, 3, 3, 1, ],
        [5, 6, 3, 4, 4, 3, 4, 0, 4, 0, 6, 3, 3, 1, ],
        [5, 4, 1, 7, 7, 1, 3, 4, 6, 4, 2, 4, 3, 1, ],
        [3, 6, 5, 0, 1, 0, 7, 4, 4, 4, 0, 7, 3, 1, ],
        [5, 6, 5, 1, 4, 0, 6, 6, 5, 5, 5, 4, 4, 1, ],
        [6, 0, 2, 4, 1, 5, 3, 2, 7, 1, 3, 6, 4, 1, ],
        [1, 0, 4, 0, 2, 1, 3, 1, 2, 1, 1, 1, 0, 2, ],
        [2, 1, 3, 3, 7, 3, 5, 1, 0, 0, 7, 1, 0, 2, ],
        [1, 0, 3, 2, 1, 4, 4, 3, 4, 4, 2, 2, 0, 2, ],
        [1, 2, 6, 3, 3, 5, 7, 4, 5, 4, 4, 3, 0, 2, ],
        [7, 5, 6, 3, 6, 1, 3, 5, 7, 4, 7, 1, 2, 2, ],
        [0, 0, 4, 2, 5, 1, 5, 6, 1, 7, 7, 1, 2, 2, ],
        [3, 6, 7, 1, 1, 0, 1, 3, 5, 4, 1, 2, 2, 2, ],
        [0, 0, 6, 1, 5, 6, 0, 0, 4, 0, 7, 2, 2, 2, ],
        [4, 0, 2, 6, 2, 5, 7, 3, 1, 0, 6, 6, 2, 2, ],
        [0, 2, 4, 6, 6, 4, 2, 3, 0, 3, 6, 7, 2, 2, ],
        [1, 7, 7, 6, 0, 4, 1, 6, 6, 3, 1, 1, 3, 2, ],
        [5, 4, 1, 0, 7, 7, 4, 0, 3, 3, 7, 6, 3, 2, ],
        [3, 0, 2, 7, 4, 4, 2, 5, 5, 2, 3, 1, 4, 2, ],
        [1, 3, 7, 5, 2, 3, 3, 1, 3, 0, 5, 3, 4, 2, ],
        [7, 6, 1, 3, 1, 2, 6, 0, 7, 3, 5, 3, 4, 2, ],
        [0, 2, 7, 1, 3, 3, 3, 7, 0, 1, 1, 5, 4, 2, ],
        [2, 5, 3, 6, 7, 7, 2, 6, 0, 1, 3, 5, 4, 2, ],
        [2, 3, 3, 7, 7, 1, 7, 7, 4, 0, 7, 0, 0, 3, ],
        [3, 0, 2, 0, 3, 0, 7, 2, 2, 0, 4, 1, 0, 3, ],
        [1, 3, 4, 0, 4, 1, 4, 4, 3, 7, 5, 2, 0, 3, ],
        [5, 2, 5, 3, 1, 3, 2, 1, 3, 4, 1, 3, 0, 3, ],
        [6, 4, 7, 0, 3, 6, 6, 0, 3, 4, 4, 4, 0, 3, ],
        [6, 1, 7, 4, 3, 0, 5, 0, 3, 5, 6, 4, 0, 3, ],
        [0, 7, 7, 1, 0, 7, 2, 6, 3, 0, 2, 0, 1, 3, ],
        [7, 4, 6, 7, 2, 3, 3, 0, 3, 6, 3, 6, 1, 3, ],
        [5, 5, 6, 1, 1, 3, 7, 6, 6, 3, 6, 1, 2, 3, ],
        [1, 1, 0, 5, 4, 2, 4, 7, 1, 7, 2, 3, 2, 3, ],
        [0, 0, 6, 2, 4, 6, 0, 0, 2, 7, 6, 3, 2, 3, ],
        [6, 5, 1, 5, 2, 1, 4, 6, 0, 6, 0, 4, 2, 3, ],
        [1, 3, 1, 0, 4, 6, 1, 1, 7, 5, 6, 4, 2, 3, ],
        [2, 4, 2, 5, 1, 6, 2, 7, 6, 0, 7, 4, 2, 3, ],
        [1, 1, 1, 6, 5, 4, 6, 5, 7, 3, 4, 5, 2, 3, ],
        [2, 7, 3, 7, 4, 3, 0, 7, 2, 4, 4, 5, 2, 3, ],
        [2, 2, 7, 1, 5, 4, 0, 2, 4, 1, 4, 6, 2, 3, ],
        [2, 7, 7, 0, 7, 0, 6, 3, 3, 7, 2, 7, 2, 3, ],
        [6, 1, 6, 7, 4, 2, 5, 7, 4, 4, 4, 0, 3, 3, ],
        [3, 3, 4, 7, 4, 6, 6, 0, 0, 3, 7, 3, 3, 3, ],
        [3, 2, 6, 7, 3, 1, 0, 0, 5, 0, 5, 4, 3, 3, ],
        [4, 0, 5, 6, 0, 2, 2, 3, 7, 5, 2, 5, 3, 3, ],
        [2, 4, 2, 4, 4, 6, 4, 0, 1, 0, 4, 4, 4, 3, ],
        [2, 0, 5, 5, 4, 3, 6, 0, 0, 0, 3, 5, 4, 3, ],
        [4, 2, 1, 0, 0, 6, 3, 7, 3, 0, 3, 5, 4, 3, ],
        [7, 1, 4, 7, 1, 6, 5, 5, 0, 7, 6, 6, 4, 3, ],
        [5, 6, 5, 5, 2, 6, 3, 7, 4, 1, 7, 7, 4, 3, ],
        [7, 4, 7, 1, 6, 0, 3, 4, 2, 6, 5, 1, 0, 4, ],
        [1, 1, 1, 6, 0, 0, 2, 2, 1, 0, 4, 3, 0, 4, ],
        [2, 1, 4, 3, 2, 1, 4, 0, 4, 4, 0, 6, 0, 4, ],
        [4, 2, 2, 0, 0, 4, 7, 7, 3, 2, 5, 7, 0, 4, ],
        [4, 1, 4, 4, 7, 1, 0, 4, 1, 7, 6, 0, 1, 4, ],
        [3, 1, 6, 4, 4, 4, 7, 4, 3, 0, 6, 3, 1, 4, ],
        [1, 5, 1, 6, 5, 3, 1, 6, 4, 1, 5, 6, 1, 4, ],
        [1, 4, 1, 4, 4, 2, 2, 3, 3, 6, 1, 7, 1, 4, ],
        [3, 2, 5, 3, 0, 2, 3, 7, 4, 0, 7, 0, 2, 4, ],
        [0, 0, 5, 4, 3, 3, 1, 7, 5, 3, 3, 2, 2, 4, ],
        [7, 4, 5, 0, 1, 6, 4, 5, 1, 3, 4, 3, 2, 4, ],
        [0, 4, 6, 0, 7, 1, 3, 1, 3, 5, 2, 7, 2, 4, ],
        [5, 0, 0, 7, 0, 1, 1, 6, 4, 6, 0, 0, 3, 4, ],
        [4, 5, 3, 1, 3, 1, 3, 5, 7, 5, 1, 3, 3, 4, ],
        [5, 4, 5, 0, 6, 5, 3, 0, 1, 7, 7, 5, 3, 4, ],
        [1, 6, 4, 4, 4, 6, 6, 0, 5, 5, 3, 0, 4, 4, ],
        [6, 2, 3, 4, 5, 6, 7, 3, 3, 4, 4, 0, 4, 4, ],
        [6, 5, 4, 3, 0, 4, 3, 3, 2, 7, 1, 6, 4, 4, ],
        [3, 5, 5, 5, 1, 2, 2, 5, 2, 5, 7, 7, 4, 4, ],
    ])

    def action_with_explain(self, ship_state, game_state):
        state = parse_game_state(ship_state, game_state)
        radar = get_radar(state['asteroids']['polar_positions'], state['asteroids']['radii'], [100, 250, 400])
        radar[radar > 0.5] = 0.5

        winner_rule = get_winner_rule(self.rules, radar)
        explanation = "Winning fuzzy rule: "
        explanation += rule2string(winner_rule)
        thrust, turn = parse_output(winner_rule[-2:])

        shooting_threshold = 15
        asteroid_angles = np.degrees(state['asteroids']['polar_positions'][:, 1])
        should_shoot = np.logical_or(np.any(asteroid_angles < shooting_threshold),
                                     np.any(asteroid_angles > 360 - shooting_threshold))
        if ship_state['is_respawning']:
            should_shoot = False

        return thrust, turn, should_shoot, False, explanation

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust, turn, shoot, mine, _ = list(self.action_with_explain(ship_state, game_state))
        return thrust, turn, shoot, mine


    @property
    def name(self) -> str:
        return "Global Controller"
    pass

def main():
    scenario = Scenario(
        ship_states=[
            {
                'position': (400, 400)
            }
        # ], asteroid_states=[
        #     {
        #         'position': (500, 500),
        #         'angle': 0,
        #         'speed': 0,
        #         'size': 4,
        #     }
         ], map_size=(1000, 800),
         num_asteroids=8
    )

    game = KesslerGame()
    game.run(scenario=scenario, controllers=[GlobalController()])


if __name__ == '__main__':
    main()
