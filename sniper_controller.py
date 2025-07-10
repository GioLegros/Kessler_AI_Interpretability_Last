from typing import Dict, Tuple

import numpy as np
from kesslergame import KesslerGame, KesslerController, Scenario

from src.center_coords import c2p
from src.lib import parse_game_state


class SniperController(KesslerController):
    def __init__(self):
        self.fsm_state = 'STOPPING'
        self.locked_coordinates = None
        self.countdown = 999

    def activate(self):
        self.fsm_state = 'STOPPING'

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust, turn, shoot, mine, _ = list(self.action_with_explain(ship_state, game_state))
        return thrust, turn, shoot, mine

    def action_with_explain(self, ship_state, game_state):
        # One huge caveat for sniping is that bullets do *not* wrap around the map!! This changes a lot of things.
        explain = ""
        state = parse_game_state(ship_state, game_state)

        my_speed = state['ship']['ship_speed']
        my_xy = state['ship']['ship_position']

        if my_speed > 12:
            self.fsm_state = 'STOPPING'
            explain += "Stopping the ship for better accuracy. "
        elif my_speed <= 12 and self.fsm_state == 'STOPPING':
            explain += "Ship has stopped moving. Acquiring target. "
            self.fsm_state = 'ACQUIRE_TARGET'

        if self.fsm_state == 'STOPPING':
            return -1 * my_speed[0], 0, False, False, explain

        if self.fsm_state == 'ACQUIRE_TARGET':
            nearest_asteroid_idx = np.argmin(state['asteroids']['polar_positions'][:, 0])
            curr_xy = state['asteroids']['xy_positions'][nearest_asteroid_idx]
            xy_velocity = state['asteroids']['xy_velocity'][nearest_asteroid_idx]

            i_lim = int(3 / game_state['delta_time'])
            for i in range(i_lim):
                dt = i * game_state['delta_time']
                test_coordinates = np.mod(curr_xy + dt * xy_velocity, game_state['map_size'])
                dx = test_coordinates[0] - my_xy[0]
                dy = test_coordinates[1] - my_xy[1]
                angle = np.degrees(c2p(dx, dy)[1]) % 360
                dist = np.linalg.norm(test_coordinates - my_xy)
                turn_time = angle / 180
                turn_time = turn_time + (turn_time % game_state['delta_time'])
                travel_time = dist / 800
                travel_time = travel_time + (travel_time % game_state['delta_time'])
                if turn_time + travel_time < dt:
                    self.locked_coordinates = test_coordinates
                    self.countdown = game_state['time'] + dt - travel_time
                    self.fsm_state = 'AIMING'
                    explain += f"Locked onto coordinates ({test_coordinates[0]:.2f}, {test_coordinates[1]:.2f}). "
                    break

        if self.fsm_state == 'AIMING':
            dx = self.locked_coordinates[0] - my_xy[0]
            dy = self.locked_coordinates[1] - my_xy[1]
            angle = c2p(dx, dy)[1]
            angle = np.degrees(np.mod(angle, 2 * np.pi))
            heading = np.degrees(state['ship']['ship_heading'])
            # Angle, Heading range from 0 (pointed right) to 360
            to_turn = (angle - heading) % 360
            if to_turn > 180:
                to_turn -= 360
            if -1 < to_turn < 1:
                self.fsm_state = 'READY'
                explain += "Finished aiming. "
            else:
                explain += "Aiming... "
                if to_turn < 0:
                    # Turn right
                    # There is no acceleration, drag etc. on turning
                    # Turn as quickly as possible, unless we are very close already
                    if to_turn >= -3:
                        turn_rate = -3
                    else:
                        turn_rate = -180
                else:
                    # Turn left
                    if to_turn <= 3:
                        turn_rate = 3
                    else:
                        turn_rate = 180
                return 0, turn_rate, False, False, explain

        if self.fsm_state == 'READY':
            if state['game']['time'] >= self.countdown:
                self.fsm_state = 'ACQUIRE_TARGET'
                explain += "Fire!!"
                return 0, 0, True, False, explain
            else:
                explain += "Standby for asteroid to move into locked coordinates."
                return 0, 0, False, False, explain

        explain += "Wow! The controller totally broke!"
        return 0, 0, False, False, explain

    @property
    def name(self) -> str:
        return "Sniper Controller"


def main():
    scenario = Scenario(
        ship_states=[
            {
                'position': (200, 200)
            }
        # ], asteroid_states=[
        #     {
        #         'position': (200, 220),
        #         'angle': 0,
        #         'speed': 100,
        #         'size': 4,
        #     }
         ], map_size=(1000, 800),
        num_asteroids=3
    )

    game = KesslerGame()
    game.run(scenario=scenario, controllers=[SniperController()])

    print("HEllo, WORLD")


if __name__ == '__main__':
    main()
