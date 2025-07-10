import os
from typing import Dict, Tuple

from kesslergame import KesslerGame, KesslerController
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.envs.xy_env import XYEnv, get_obs
from src.navigation_scenario import *

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train(scenario, target):
    print("hi..")
    vec_env = make_vec_env(XYEnv, n_envs=6, env_kwargs={
        'scenario': scenario,
        'target_xy': target,
    })
    eval_env = Monitor(XYEnv(scenario=scenario, target_xy=target))
    model = PPO("MultiInputPolicy", vec_env)
    os.makedirs(f'../out/xy_test', exist_ok=True)
    for i in range(1200):
        model.learn(total_timesteps=20_000)
        model.save(f'../out/xy_test/bookmark_{i}')
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20, return_episode_rewards=False)
        print(f'{i:d} .. Mean reward: {mean_reward:.2f}')
    print("")

def run(scenario, model_name):
    kessler_game = KesslerGame()
    controller = SuperDummyController(model_name)
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=False)

class SuperDummyController(KesslerController):
    def __init__(self, model_name):
        self.model = PPO.load(model_name)

    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = get_obs(game_state=game_state)
        #print(obs)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

def main():
    target = np.array([300, 300])
    scenario = Scenario(num_asteroids=0, map_size=(600, 600), ship_states=[
        {
            'position': (100, 100),
        }
    ])
    train(scenario=scenario, target=target)
    #run(scenario, '../out/xy_test/bookmark_6')

def run_benchmark():
    controller = SuperDummyController(model_name='out/10_GUNS_OFF_1S_FORECAST/9')
    results = benchmark(controller)
    print(results)
    print(np.mean(results))


if __name__ == '__main__':
    main()
    #run_benchmark()
