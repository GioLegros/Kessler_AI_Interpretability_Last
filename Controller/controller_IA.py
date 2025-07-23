import math
from src.controller import KesslerController
from src.ship import Ship
import numpy as np
from typing import Dict, Tuple, Any

class IAController(KesslerController):
    IA_Hard = True
    def __init__(self):
        self._thrust = 0.0
        self._turn_rate = 0.0
        self._fire = False
        self._drop_mine = False

    @property
    def name(self) -> str:
        return "IA Controller"

    def actions(self, ship_state: Dict[str, Any], game_state: Dict[str, Any]) -> Tuple[float, float, bool, bool]:
        if self.IA_Hard:
            # Implement a simple IA logic
            if ship_state['speed'] < 100:
                self._thrust = 480.0
            else:
                self._thrust = 0.0

            if ship_state['heading'] < 180:
                self._turn_rate = 30.0
            else:
                self._turn_rate = -30.0

            self._fire = True if ship_state['can_fire'] else False
            self._drop_mine = False

        else:
            
            if ship_state['speed'] < 100:
                self._thrust = 480.0

        return self._thrust, self._turn_rate, self._fire, self._drop_mine