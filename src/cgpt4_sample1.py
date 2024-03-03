import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Initialize the environment
env = make_vec_env('LunarLander-v2', n_envs=4)

# Initialize the model
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)

# Train the model
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("ppo_lunarlander")

# Test the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()

env.close()
