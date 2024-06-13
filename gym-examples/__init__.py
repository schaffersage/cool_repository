from gymnasium.envs.registration import register

register(
    id="gym_examples/Hohmann_Transfer-v0",
    entry_point="gym_examples.envs:Hohmann_TransferEnv",
)