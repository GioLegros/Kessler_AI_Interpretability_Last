import os
import numpy as np
from typing import Dict, Tuple

from kesslergame import KesslerGame, KesslerController
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.envs.radar_env import RadarEnv, get_obs
from src.navigation_scenario import *

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train(scenario, radar_zones, bumper, forecast_frames, name):
    print(f"Starting {name}...")
    vec_env = make_vec_env(RadarEnv, n_envs=6, env_kwargs={
        'scenario': scenario,
        'radar_zones': radar_zones,
        'bumper_range': bumper,
        'forecast_frames': forecast_frames,
    })
    model = PPO("MultiInputPolicy", vec_env)
    eval_env = Monitor(RadarEnv(scenario=scenario))
    os.makedirs(f'../out/{name}', exist_ok=True)
    for i in range(120):
        model.learn(total_timesteps=200_000)
        model.save(f'../out/{name}/bookmark_{i}')
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=30, return_episode_rewards=False)
        print(f'{i:d} .. Mean reward: {mean_reward:.2f}')
    print("")

def run(scenario, model_name):
    kessler_game = KesslerGame()
    controller = SuperDummyController(model_name)
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller])


class SuperDummyController(KesslerController):
    def __init__(self, model_name):
        self.model = PPO.load(model_name)

    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = get_obs(game_state=game_state, forecast_frames=30, radar_zones=[100, 250, 400], bumper_range=50)
        area_clear = np.all(obs['radar'][4:] < 0.001).astype(int)

        action = self.model.predict(obs)
        thrust, turn = list(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

def main():
    zones = [100, 250, 400]
    bumper = 50
    name = 'SimpleScenario'

    asteroid_states = [{
        'position': (50, 400),
        'angle': 0,
        'speed': 0,
        'size': 4
    }]

    s = Scenario(
        map_size=(1000, 800),
        ship_states=[{
            'position': (150, 400),
            'angle': 0,
            'lives': 1,
        }],
        asteroid_states=asteroid_states,
        time_limit=30,
    )

    #train(scenario=s, radar_zones=zones, bumper=bumper, forecast_frames=30, name=name)
    run(s, 'out/expApril3_0_0/bookmark_0')

def run_benchmark():
    controller = SuperDummyController(model_name='out/10_GUNS_OFF_1S_FORECAST/9')
    results = benchmark(controller)
    print(results)
    print(np.mean(results))


if __name__ == '__main__':
    main()
    #run_benchmark()


#     marker_size = 50 * (asteroid_size ** 2)
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xlim(-1 * MAP_SIZE, MAP_SIZE)
#     ax.set_ylim(-1 * MAP_SIZE, MAP_SIZE)
#     ax.scatter(x=asteroid_xy[:, 0], y=asteroid_xy[:, 1], s=marker_size)
#
#     ax.plot([-353, 353], [-353, 353], color='k')
#     ax.plot([-353, 353], [353, -353], color='k')
#
#     my_ship = plt.Circle((0, 0), 25, color='red')
#     ax.add_patch(my_ship)
#
#     circle1 = plt.Circle((0, 0), 100, color='k', fill=False)
#     ax.add_patch(circle1)
#
#     circle2 = plt.Circle((0, 0), 300, color='k', fill=False)
#     ax.add_patch(circle2)
#
#     circle3 = plt.Circle((0, 0), 500, color='k', fill=False)
#     ax.add_patch(circle3)
#
#
#     plt.show()
#
#     pass
