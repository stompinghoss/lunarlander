import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import numpy as np
import os
import time

'''
TODO: Factor out common code between trainer and evaluator.
TODO: Factor into class structure
Notes on hyper parameters and approach.
Shifting from DQN to PPO definitely helped.
Shifting from low level pytorch to sb3 definitely made things easier.
Higher exploration definitely helps.
Higher exploration needs more time steps to run.
But, reward shaping definitely has the highest impact.
Some comments/explanations courtesy of ChatGPT4 answering my questions.
'''

PROBLEM_NAME = "LunarLander-v2"

'''
 Maximum timesteps specifies how many times the agent will take an action in the environment.
 Each timestep typically represents one action-selection and the observation
 of its outcome (the next state and reward).
 This is a measure of how long the training will run. More timesteps usually
 mean longer training, during which the agent has more opportunities to learn
 from the environment.
'''

MAX_TIMESTEPS = 2000000

'''
The entropy coefficient scales the entropy in the policy loss function which
if increased, rewards higher randomness of actions taken.
'''

ENTROPY_COEFFICIENT = 0.1

'''
* Drives exploration vs exploitation *
For each update, this many steps are run, generating n steps of experiences from the environment.
In a vectorized environment, this many steps will be taken per environment.
'''

INTERACTIONS_PER_POLICY_UPDATE = 1024

'''
How much experience is collected before learning.
Those experiences are then sampled into batches. Those batches may in turn be
samples into smaller batches. This paramater is the mini batch size.
'''

MINI_BATCH_SIZE = 64

'''
Interations with the environment are built into experiences which are batched.
Normally, as tuples like (state, action, reward, next state, done).
An epoch is 1  pass through this data. Higher might squeeze more benefit.
Too high can lead to overfitting.
Obervations: 6 no better than 4, but might be because time steps was too low--
those scenarios were at 100,000.
Was still no better with 1000,000.
'''

EPOCHS_PER_UPDATE_CYCLE = 4

'''
*Drives long term gain*
Gamma (aka discount factor) is the discount factor which is between 0 and 1.
Higher means favour long term rewards.
'''

DISCOUNT_FACTOR = 0.999

'''
* Drives exploration vs exploitation *
GAE is a technique used to estimate the advantage function in policy gradient
methods like Proximal Policy Optimization (PPO). The advantage function
measures how much better it is to take a specific action compared to the
average action in a given state

High Lambda (λ close to 1): A higher value of lambda leads to an advantage estimate that
incorporates more steps into the future, increasing the variance but
otentially capturing a more accurate picture of future rewards.
This can lead to more exploration but might also introduce instability in
learning due to higher variance.

Generalised Advantage Estimation Lambda balances variance and bias in the GAE
calculation. Higher variance means less stable learning but drives more exploration.

'''

GAE_LAMBDA = 0.95

'''
This parameter defines how many times the environment will be reset and
the policy run from start to finish (one episode). Each episode typically
runs until a terminal state (like a win/loss state in a game or the end of a
task) is reached or until a maximum number of steps per episode is exceeded.
'''

EVAL_EPISODES = 10
EVAL_ENV_INTERACTION_STEPS = 100000
PARALLEL_ENVIRONMENTS = 16
LOG_DIR = "logs/"


'''
For reward shaping, we want to reward:
Minimal engine use. Main engines penalised more than side engines.
Coming in under gravity.
Not firing the engines once in contact with the ground.
'''


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
        EFFICIENT_DESCENT_REWARD = 0.3

        # Check if either leg is in contact with the ground
        ground_contact = state[LEFT_LEG_GROUND_CONTACT] or state[RIGHT_LEG_GROUND_CONTACT]
        side_engine_use = action == LEFT_ENGINE or action == RIGHT_ENGINE

        # Main engine costs more than side engines
        if action == MAIN_ENGINE:
            reward += MAIN_ENGINE_USE_PENALTY

        if side_engine_use:
            reward += SIDE_ENGINE_USE_PENALTY

        # Positive reward if the lander is descending (vertical velocity < 0) efficiently
        if state[VERTICAL_VELOCITY] < 0:
            reward += EFFICIENT_DESCENT_REWARD

        # Penalize if engines are fired while one or both legs on the ground
        if ground_contact and action == MAIN_ENGINE or side_engine_use:
            reward += GROUND_ENGINE_USE_PENALTY

        return reward

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        reward = self.custom_reward(state, reward, action)
        return state, reward, terminated, truncated, info


# Setup logging for TensorBoard
print("Setting up logging for tensorboard")
os.makedirs(LOG_DIR, exist_ok=True)
logger = configure(LOG_DIR, ["stdout", "tensorboard"])

print("Initialising gymnasium environment")
# env = gym.make(PROBLEM_NAME)
wrapped_env = make_vec_env(PROBLEM_NAME, n_envs=PARALLEL_ENVIRONMENTS, wrapper_class=LunarLanderCustomReward)

model = PPO(policy='MlpPolicy',
            env=wrapped_env,
            n_steps=INTERACTIONS_PER_POLICY_UPDATE,
            batch_size=MINI_BATCH_SIZE,
            n_epochs=EPOCHS_PER_UPDATE_CYCLE,
            gamma=DISCOUNT_FACTOR,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENTROPY_COEFFICIENT,
            verbose=1,
            tensorboard_log=LOG_DIR)

# Set the logger
model.set_logger(logger)

print("Learning")
model.learn(total_timesteps=MAX_TIMESTEPS)

# Save the trained model
print("Saving model")
model.save("ppo_lunarlander")

print("Evaluating policy")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=EVAL_EPISODES, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Save evaluation metrics
print("Saving evaluation metrics")
with open("evaluation_results.txt", "w") as file:
    file.write(f"Mean Reward: {mean_reward}\n")
    file.write(f"Standard Deviation of Reward: {std_reward}\n")

print("Testing the trained agent")
terminated = False
truncated = False

for i in range(EVAL_ENV_INTERACTION_STEPS):
    # Reset the environments at the start of each episode
    if i == 0 or terminated or truncated:
        obs = wrapped_env.reset()

    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = wrapped_env.step(action)

    # Extract the action as a single integer value
    single_action = action[0] if isinstance(action, np.ndarray) else action

    terminated = dones.any()  # Check if any environment is don

wrapped_env.close()

print("Training and evaluation finished.")
