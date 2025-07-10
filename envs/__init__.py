from gymnasium.envs.registration import register

from .kessler_env import KesslerEnv

__all__ = [KesslerEnv]

register(
    id="envs/KesslerEnv-v0",
    entry_point="envs:KesslerEnv"
)