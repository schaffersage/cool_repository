import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v2", render_mode="human")
#make/render_mode defines how an env should be visualized
observation, info = env.reset()
#restarts env

model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent

model.save("dqn_lunar")
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    #step means action ai takes to alter env

    if terminated or truncated:
        #observation, info = env.reset()
        break


env.close()
#it works... finally....
