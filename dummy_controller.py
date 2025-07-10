from typing import Dict, Tuple

from kesslergame import KesslerController
from stable_baselines3 import PPO

from src.envs.radar_env import get_obs, THRUST_SCALE, TURN_SCALE

class PPODummy(KesslerController):
    def __init__(self, model_name):
        self.model = PPO.load(model_name)

    @property
    def name(self) -> str:
        return "PPO Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = get_obs(game_state=game_state, forecast_frames=30, radar_zones=[100, 250, 400], bumper_range=50)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

    def eval_policy(self, obs):
        return self.model.predict(obs)

