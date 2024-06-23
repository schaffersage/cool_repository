from gymnasium.envs.registration import register

register(
    id="Hohmann_Transfer-v0",
    entry_point="gym_examples.envs:HohmannTransferEnv",
)
