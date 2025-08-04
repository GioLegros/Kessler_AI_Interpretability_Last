from typing import Dict, Tuple, List
import numpy as np
from collections import deque
from stable_baselines3.common.monitor import Monitor
from envs.kessler_env import KesslerEnv
from src.kessler_game import KesslerGame, TrainerEnvironment
from src.scenario import Scenario
from Controller.controller_IA import ControllerIA
from src.controller import KesslerController
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym 

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

test_scenario = Scenario(
    time_limit=60,
    name="test_scenario",
    map_size=(1000, 800),
    asteroid_states=[{"position": (0, 300), "angle": -90.0, "speed": 40},
                     {"position": (700, 300), "angle": 0.0, "speed": 0}],
    ship_states=[{"position": (600, 300)}],
    seed=0
)

def train_nn():
    kessler_env = Monitor(KesslerEnv(test_scenario))
    model = PPO("MultiInputPolicy", kessler_env)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'Initial Mean reward: {mean_reward:.2f}')

    model.learn(5000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+5000   Mean reward: {mean_reward:.2f}')
    model.save("out/5k")

    model.learn(50000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    model.save("out/50k")

    model.learn(500000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+500000 Mean reward: {mean_reward:.2f}')
    model.save("out/500k")

    print("Saving")
    model.save("out/test")

def run_game():
    kessler_game = KesslerGame()
    scenario = Scenario(num_asteroids=2, time_limit=180, map_size=(1000, 800))
    controller = SuperdummyController()
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller])
    print(f"Final Score: {score}")

class SuperdummyController(KesslerController):
    def __init__(self):
        self.model= PPO.load("out/50k")

    @property
    def name(self):
        return "NN_Controller"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
#        print(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False
    
    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]
        obs = {
            "ship position": np.array(ship['position']),
            "ship speed": np.array([ship['speed']]),
            "ship heading": np.array([ship['heading']]),
        }
        print(obs['ship position'])
        return obs
    
if __name__ == '__main__':
    #train_nn()
    run_game()