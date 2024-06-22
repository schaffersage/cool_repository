import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Hohmann_Transfer-v0",render_mode="human")
#make/render_mode defines how an env should be visualized
observation, info = env.reset()
#restarts env

model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
