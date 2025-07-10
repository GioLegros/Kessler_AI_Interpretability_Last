import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from envs import KesslerEnv
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController, StopReason
from typing import Dict, Tuple
import numpy as np

THRUST_SCALE, TURN_SCALE = 480.0, 180.0

def train():
    n_actions = 2
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    kessler_env = Monitor(KesslerEnv())
#    kessler_env = make_vec_env(KesslerEnv, n_envs=4)
#    kessler_env = DummyVecEnv([lambda: kessler_env])
#    check_env(kessler_env, warn=True)
    model = DDPG("MultiInputPolicy", kessler_env, verbose=False, action_noise=action_noise)


    model.learn(total_timesteps=50000, log_interval=10)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    model.save("ddpgout/50k")


    print("Saving")
    model.save("ddpgout/test")

def run():
    kessler_game = KesslerGame()
    scenario = Scenario(num_asteroids=0, time_limit=180, map_size=(400, 400))
    controller = SuperDummyController()
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller], stop_on_no_asteroids=False)
    # print(score)


class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = DDPG.load("out/current")
        #self.model = PPO.load("out/500k")
    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

if __name__ == '__main__':
    my_scenario = Scenario(time_limit=180, map_size=(800, 800),
                           asteroid_states=[{'position': (0, 0)}] * 5,
                           ship_states=[
                               {
                                   'position': (400, 400),
                                   'lives': 1
                               }
                           ])
    #train(my_scenario)
    run(my_scenario)

if __name__ == '__main__':
    train()
    run()

