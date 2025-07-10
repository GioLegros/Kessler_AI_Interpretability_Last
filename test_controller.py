import time
from typing import Dict, Tuple
from kesslergame import KesslerController, KesslerGame, Scenario, StopReason
from navigation_scenario import *

# See: https://github.com/ThalesGroup/kessler-game/blob/main/examples/test_controller.py
class TestController(KesslerController):
    def __init__(self):
        self.n = 0
        pass

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        if self.n < 120:
            thrust = 480
            self.n += 1
            fire = False
        else:
            thrust = 0
            fire = True
        turn_rate = 0
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Test Controller"

def main():
    controller = TestController()
    game = KesslerGame()
    game = game.run(scenario=Scenario(num_asteroids=1), controllers=[controller])
    #b = benchmark(controller)
   # print(b)


if __name__ == '__main__':
    main()
