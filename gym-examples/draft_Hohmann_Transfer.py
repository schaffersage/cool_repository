import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gym_examples

env_train = gym.make("Hohmann_Transfer-v0")
#make/render_mode defines how an env should be visualized
observation, info = env_train.reset()
#restarts env

model = SAC('MlpPolicy', env_train, learning_rate=1e-3, verbose=1)
# Train the agent
print("starting training")
model.learn(total_timesteps=int(100))
# Show the points getting closer
env_train.close()


#TESTING MODEL
# this will evaluate and show the dots hopefully getting closer
env_see = gym.make("Hohmann_Transfer-v0", render_mode = "human")
env_see.reset()
reward,rewardstd = evaluate_policy(model, env_see, n_eval_episodes=10, render=True) #Evaluates 10 episodes and renders the trained model
model.save("Trained_model")
env_see.close()
