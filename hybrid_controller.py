# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple
from src.controller import KesslerController
from glass_controller import GlassBoxFleeController
from nn_controller import NeuralFleeController
from headmaster import HeadMaster

class HybridController(KesslerController):
    """
    Controller that switches between GlassBoxFleeController and NeuralFleeController
    """
    def __init__(self, headmaster: HeadMaster, nn_model_path: str, device: str = "cpu"):
        self._ship_id = 0
        self.hm = headmaster
        self.glass = GlassBoxFleeController()
        self.black = NeuralFleeController(model_path=nn_model_path, device=device)
        self._last_mode = "glass"

    @property
    #have a cool name display in the GUI
    def name(self) -> str:
        return f"Hybrid[{self._last_mode.capitalize()}]"

    def actions(self, ship_state, game_state):
        self.hm.update_assignments(game_state)
        if self.hm.is_black_box(ship_state["id"]):
            self._last_mode = "black"
            return self.black.actions(ship_state, game_state)
        else:
            self._last_mode = "glass"
            return self.glass.actions(ship_state, game_state)

