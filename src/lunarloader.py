import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import pygame
from stable_baselines3.common.monitor import Monitor
from custom_reward import LunarLanderCustomReward

# Define the problem name and other constants
PROBLEM_NAME = "LunarLander-v2"
EVAL_EPISODES = 10
EVAL_ENV_INTERACTION_STEPS = 100000


# Function to load the model
def load_model():
    # Construct the model path
    model_path = "ppo_lunarlander"

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
