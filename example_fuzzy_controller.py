import numpy as np
from typing import Dict, Tuple
from src.controller import KesslerController
from src.kessler_game import KesslerGame
from src.scenario import Scenario

from envs.radar_env import get_obs, THRUST_SCALE, TURN_SCALE
from fuzzy import get_output
from fuzzy_rule_extractor import EXTRACTION_SCALE
from navigation_scenario import simple_scenario, scenario_D


class ExampleFuzzyController(KesslerController):
    def __init__(self):
        self.fuzzy_rule = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 3, 0, 13, 13, 4, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 10, ],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 3, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, ],
            [0, 0, 0, 0, 4, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 14, 0, 0, 0, 0, ],
            [14, 0, 0, 0, 12, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 2, 3, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 9, 0, 0, 0, ],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [2, 0, 0, 0, 9, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 13, 10, 0, 0, 0, 0, 0, 0, 0, 0, 14, ],
            [0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [14, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 6, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 2, 6, 0, 5, 5, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 6, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [14, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 10, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 8, ],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, ],
            [0, 10, 0, 0, 0, 0, 14, 12, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, ],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 14, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 8, ],
            [0, 4, 8, 0, 0, 14, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 2, 8, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, ],
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 6, 0, 14, 0, 0, ],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 12, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 11, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 9, 0, 0, 0, 7, 0, 0, 0, 1, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, ],
            [0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 9, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 14, 0, 0, 13, 0, 0, 0, 11, 0, ],
            [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, ],
            [0, 4, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 7, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 14, 1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 13, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 7, 0, 0, 0, ],
            [0, 0, 0, 11, 0, 0, 0, 0, 9, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 4, 0, 0, 13, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 13, 0, 8, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 14, 0, 1, 0, 0, 0, ],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 11, 0, 0, 0, 0, 6, 0, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 8, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, ],
            [2, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 11, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, ],
            [0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, ],
            [0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 6, 0, 0, 0, 0, 0, ],
        ])

        self.rule_outputs = np.array([
            [0.1018, -0.0628, ],
            [-1.0000, 0.0659, ],
            [0.0258, -0.8136, ],
            [0.0973, 0.3639, ],
            [0.7503, 1.0000, ],
            [-0.4163, -0.9965, ],
            [-1.0000, -0.4407, ],
            [-0.0993, -0.8329, ],
            [0.7562, -0.3817, ],
            [0.5141, -0.6858, ],
            [-0.4546, 0.7972, ],
            [-0.5642, 0.5802, ],
            [-0.7958, -0.0476, ],
            [0.4591, -0.3393, ],
            [-0.9990, 0.5966, ],
            [0.2672, -0.4217, ],
            [0.4893, -0.1246, ],
            [0.6432, -0.5633, ],
            [0.6984, 0.9268, ],
            [-0.0369, 1.0000, ],
            [0.1644, 0.8414, ],
            [-0.3289, -0.5762, ],
            [-0.9192, -0.7636, ],
            [-0.5556, 0.6162, ],
            [0.7672, 0.1598, ],
            [-0.0488, -0.9655, ],
            [1.0000, -0.8081, ],
            [-0.3966, -0.8958, ],
            [-0.0228, 0.7966, ],
            [-0.3396, 0.9877, ],
            [0.4460, 0.3038, ],
            [0.9841, 0.1398, ],
            [-0.0669, -0.6888, ],
            [0.5699, -0.3889, ],
            [-0.1279, 0.5596, ],
            [0.7729, -0.4032, ],
            [-0.8967, -0.4594, ],
            [0.5774, 0.2106, ],
            [0.2070, -0.8664, ],
            [-0.4155, -1.0000, ],
            [0.4173, -0.1182, ],
            [-0.3798, -0.9701, ],
            [0.8036, 0.9790, ],
            [0.5909, 0.3817, ],
            [-1.0000, 0.8153, ],
            [-0.1558, 0.5820, ],
            [-0.0496, 0.1902, ],
            [1.0000, -0.4214, ],
            [-0.2137, 0.0609, ],
            [0.2498, -0.3253, ],
            [-0.2964, -0.3982, ],
            [-0.2161, -0.8031, ],
            [0.9171, -0.7781, ],
            [0.6506, 0.4984, ],
            [0.3814, -0.0685, ],
            [-1.0000, 0.9822, ],
            [-0.3506, 0.7178, ],
            [0.3594, -0.2683, ],
            [-0.8310, 0.2921, ],
            [1.0000, -0.0903, ],
            [-0.3052, -0.0171, ],
            [-0.7927, -1.0000, ],
            [-1.0000, -0.9225, ],
            [0.0120, 0.2667, ],
            [-0.9611, -0.7087, ],
            [-0.1206, -0.4350, ],
            [-0.7430, -0.3119, ],
            [0.2077, -0.7117, ],
            [0.6651, 0.2006, ],
            [0.1562, 0.0796, ],
            [-0.7113, -0.2437, ],
            [-0.6519, 0.4803, ],
            [-0.7954, 0.1170, ],
            [0.4893, 0.7091, ],
            [0.7181, 0.1269, ],
            [-0.0856, 1.0000, ],
            [-0.4600, -0.2353, ],
            [-0.5437, 0.3731, ],
            [0.1666, 0.3710, ],
            [-0.5678, 0.1692, ],
            [-0.3033, 0.7011, ],
            [-0.1457, -1.0000, ],
            [0.3878, -0.7058, ],
            [0.7272, -0.1146, ],
            [-0.3701, 0.4001, ],
            [0.0268, 0.9901, ],
            [0.2940, -0.0026, ],
            [0.9011, -0.0631, ],
            [0.5563, -0.7679, ],
            [0.1137, 0.1828, ],
            [1.0000, 0.3708, ],
            [-0.0474, 0.5143, ],
            [0.2558, 0.1065, ],
            [0.0027, 0.1621, ],
            [0.5749, 0.1941, ],
            [-0.8241, -0.5321, ],
            [-0.8109, 0.3416, ],
            [-0.2037, 1.0000, ],
            [0.0173, 0.8895, ],
            [0.0109, -0.0571, ],
        ])

        self.weights = np.array([

            [16.0000, ],
            [8.0000, ],
            [6.0000, ],
            [8.0000, ],
            [7.0000, ],
            [4.0000, ],
            [6.0000, ],
            [3.0000, ],
            [1.0000, ],
            [1.0000, ],
            [12.0000, ],
            [9.0000, ],
            [0.0000, ],
            [7.0000, ],
            [7.0000, ],
            [6.0000, ],
            [8.0000, ],
            [7.0000, ],
            [12.0000, ],
            [15.0000, ],
            [12.0000, ],
            [3.0000, ],
            [12.0000, ],
            [0.0000, ],
            [2.0000, ],
            [10.0000, ],
            [5.0000, ],
            [12.0000, ],
            [2.0000, ],
            [8.0000, ],
            [0.0000, ],
            [8.0000, ],
            [14.0000, ],
            [5.0000, ],
            [9.0000, ],
            [15.0000, ],
            [7.0000, ],
            [14.0000, ],
            [6.0000, ],
            [4.0000, ],
            [0.0000, ],
            [12.0000, ],
            [12.0000, ],
            [4.0000, ],
            [2.0000, ],
            [6.0000, ],
            [0.0000, ],
            [5.0000, ],
            [7.0000, ],
            [14.0000, ],
            [2.0000, ],
            [4.0000, ],
            [9.0000, ],
            [0.0000, ],
            [4.0000, ],
            [11.0000, ],
            [2.0000, ],
            [6.0000, ],
            [6.0000, ],
            [7.0000, ],
            [5.0000, ],
            [2.0000, ],
            [2.0000, ],
            [4.0000, ],
            [12.0000, ],
            [11.0000, ],
            [15.0000, ],
            [2.0000, ],
            [7.0000, ],
            [6.0000, ],
            [1.0000, ],
            [9.0000, ],
            [9.0000, ],
            [2.0000, ],
            [11.0000, ],
            [4.0000, ],
            [0.0000, ],
            [0.0000, ],
            [5.0000, ],
            [13.0000, ],
            [8.0000, ],
            [2.0000, ],
            [2.0000, ],
            [3.0000, ],
            [11.0000, ],
            [3.0000, ],
            [17.0000, ],
            [6.0000, ],
            [12.0000, ],
            [7.0000, ],
            [1.0000, ],
            [4.0000, ],
            [11.0000, ],
            [5.0000, ],
            [9.0000, ],
            [13.0000, ],
            [8.0000, ],
            [2.0000, ],
            [7.0000, ],
            [9.0000, ],
        ])

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # obs = np.random.random(size=(N_ATTRIBUTES,))
        obs = np.array(ship_state['position']) / MAP_SIZE
        affinity = get_affinity(obs, self.fuzzy_rule)
        weighted_votes = affinity[:, np.newaxis] * self.rule_outputs
        total_weight = np.sum(affinity)
        final_output = np.sum(weighted_votes, axis=0) / total_weight
        thrust, turn, fire, mine = final_output[0], final_output[1], final_output[2] > 0.5, final_output[3] > 0.5
        return thrust, turn, fire, mine


    @property
    def name(self) -> str:
        return "OMU Example Controller"

def main():
    controller = ExampleFuzzyController()
    game = KesslerGame()
    score, _, state = game.run(scenario=scenario_D(n=64),
             controllers=[controller])
    print(score)


if __name__ == '__main__':
    main()
