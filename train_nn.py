# -*- coding: utf-8 -*-
# Training script using Stable-Baselines3 (PPO) on the custom Gymnasium env.
from __future__ import annotations
import os
from typing import Optional
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from env_gym import KesslerFleeEnv

def make_env(time_limit: float = 30.0):
    def _thunk():
        return KesslerFleeEnv(time_limit=time_limit)
    return _thunk

def main(save_dir: str = "models/kessler_ppo", total_timesteps: int = 500_000, time_limit: float = 30.0):
    os.makedirs(save_dir, exist_ok=True)
    env = DummyVecEnv([make_env(time_limit)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb"),
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2
    )
    new_logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    ckpt_cb = CheckpointCallback(save_freq=50_000 // env.num_envs, save_path=save_dir, name_prefix="ppo_kessler")
    model.learn(total_timesteps=total_timesteps, callback=ckpt_cb)
    final_path = os.path.join(save_dir, "ppo_kessler_final.zip")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="models/kessler_ppo")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--time_limit", type=float, default=30.0)
    args = parser.parse_args()
    main(args.save_dir, args.steps, args.time_limit)


#train python -m train_nn --steps 400000 --save_dir models/kessler_ppo