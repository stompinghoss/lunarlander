import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import pygame
from pygame.locals import QUIT
import numpy as np
from stable_baselines3.common.monitor import Monitor
import os
import time

# Define the problem name and other constants
PROBLEM_NAME = "LunarLander-v2"
EVAL_EPISODES = 10
EVAL_ENV_INTERACTION_STEPS = 100000


class LunarLanderCustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def custom_reward(self, state, reward, action):
        # Constants for the state indices
        LEFT_LEG_GROUND_CONTACT = 6
        RIGHT_LEG_GROUND_CONTACT = 7
        VERTICAL_VELOCITY = 3

        # Constants for actions
        MAIN_ENGINE = 2
        LEFT_ENGINE = 1
        RIGHT_ENGINE = 3

        # Engine use penalty when on ground
        GROUND_ENGINE_USE_PENALTY = -0.4
        MAIN_ENGINE_USE_PENALTY = -0.6
        SIDE_ENGINE_USE_PENALTY = -0.06
        EFFICIENT_DESCENT_REWARD = 0.2

        # Check if either leg is in contact with the ground
        ground_contact = state[LEFT_LEG_GROUND_CONTACT] or state[RIGHT_LEG_GROUND_CONTACT]
        side_engine_use = action == LEFT_ENGINE or action == RIGHT_ENGINE

        # Main engine costs more than side engines
        if action == MAIN_ENGINE:
            reward += MAIN_ENGINE_USE_PENALTY

        if side_engine_use:
            reward += SIDE_ENGINE_USE_PENALTY

        # Reward for efficient descent (using gravity)
        # Positive reward if the lander is descending (vertical velocity < 0) efficiently
        if state[VERTICAL_VELOCITY] < 0:
            reward += EFFICIENT_DESCENT_REWARD

        # Penalize if engines are fired while on ground
        if ground_contact and action == MAIN_ENGINE or side_engine_use:
            reward += GROUND_ENGINE_USE_PENALTY

        return reward


# Function to load the model
def load_model():
    # Ask the user for the model number
    model_number = input("Enter the model number (e.g., 'v5.0' for version 5.0): ")

    # Construct the model path
    model_path = f"./good/v{model_number}/ppo_lunarlander_v{model_number}"

    # Load the model
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully.")
        # Print hyperparameters
        print("\nModel Hyperparameters:")
        print(f"Learning Rate: {model.learning_rate}")
        print(f"Batch Size: {model.batch_size}")
        print(f"Gamma: {model.gamma}")
        print(f"GAE Lambda: {model.gae_lambda}")
        print(f"Number of Steps per Update: {model.n_steps}")
        print(f"Number of Epochs per Update Cycle: {model.n_epochs}")
        print(f"Entropy Coefficient: {model.ent_coef}")
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()

    return model


def evaluate_model(model):

    # Initialize Pygame and create a window
    pygame.init()
    screen_width, screen_height = 600, 400  # Adjust as needed
    screen = pygame.display.set_mode((screen_width, screen_height))

    # Create and wrap the environment
    env = Monitor(gym.make(PROBLEM_NAME, render_mode='rgb_array'))  # generate frames for manual rendering
    wrapped_env = LunarLanderCustomReward(env)
    monitored_env = Monitor(wrapped_env)

    # Evaluation
    mean_reward, std_reward = evaluate_policy(model, monitored_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Rendering loop
    for _ in range(EVAL_ENV_INTERACTION_STEPS):
        obs = wrapped_env.reset()[0]
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            results = wrapped_env.step(action)
            # TODO: Below seems convoluted.
            obs, _, dones, _ = results if len(results) == 4 else results[0:4]

            # Render the frame
            frame = wrapped_env.render()
            frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(frame, (0, 0))
            pygame.display.flip()

            terminated = dones

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame window closed by user")
                    wrapped_env.close()
                    return

        # Close the rendering window at the end of each episode
        wrapped_env.close()

    # Close the environment at the end of evaluation
    wrapped_env.close()


# Main execution
if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print("Model loaded. Starting evaluation...")
    evaluate_model(model)
    print("Evaluation completed.")
