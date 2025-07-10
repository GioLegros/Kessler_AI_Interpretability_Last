from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerGame, KesslerController, Scenario

from src.lib import parse_game_state


class RoombaController(KesslerController):
    def __init__(self):
        pass

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust, turn, shoot, mine, _ = list(self.action_with_explain(ship_state, game_state))
        return thrust, turn, shoot, mine

    def action_with_explain(self, ship_state, game_state):
        state = parse_game_state(ship_state, game_state)

        shooting_threshold = 15
        asteroid_angles = np.degrees(state['asteroids']['polar_positions'][:, 1])
        should_shoot = np.logical_or(np.any(asteroid_angles < shooting_threshold),
                                     np.any(asteroid_angles > 360 - shooting_threshold))
        if ship_state['is_respawning']:
            should_shoot = False

        nearest_asteroid_idx = np.argmin(state['asteroids']['polar_positions'][:, 0])
        angle = state['asteroids']['polar_positions'][nearest_asteroid_idx, 1]
        angle = np.degrees(angle)
        if angle > 180:
            angle -= 360

        explanation = f"Avoiding an obstacle at {angle:.2f} degrees relative to the ship."
        if -90 <= angle <= 90:
            # The obstacle is in front. Face towards it, and hit reverse!
            return -360, angle, should_shoot, False, explanation
        else:
            # The obstacle is behind us
            if angle < 0:
                turn = angle + 180
            else:
                turn = angle - 180
            return 360, turn, should_shoot, False, explanation

    @property
    def name(self) -> str:
        return "Roomba Controller"


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
    game.run(scenario=scenario, controllers=[RoombaController()])


if __name__ == '__main__':
    main()
