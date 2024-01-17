import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import numpy as np
import optuna
from tqdm import tqdm
import logging
import time
import os
from custom_reward import LunarLanderCustomReward

'''
For observation space and reward details, see
https://gymnasium.farama.org/environments/box2d/lunar_lander/

'''

'''
TODO: Factor out common code between trainer and evaluator.
TODO: Factor into class structure
Notes on hyper parameters and approach:

Shifting from DQN to PPO definitely helped.
Shifting from low level pytorch to sb3 definitely made things easier.
Higher exploration definitely helps.
Higher exploration needs more time steps to run.
But, reward shaping definitely has the highest impact, at least so far.
Some comments/explanations courtesy of ChatGPT4 answering my questions.
'''

PROBLEM_NAME = "LunarLander-v2"

LEARNING_RATE = 0.0003 # Not used yet

'''
 Maximum timesteps specifies how many times the agent will take an action in
 the environment.
 Each timestep typically represents one action-selection and the observation
 of its outcome (the next state and reward).
 This is a measure of how long the training will run. More timesteps usually
 mean longer training, during which the agent has more opportunities to learn
 from the environment.
'''

MAX_TIMESTEPS = 2000000

# Utilise an approach to calculate good hyper parameters
AUTO_TUNE = False

# The number of steps to run when trying to auto tune hyper parameters.
STUDY_NUM_TIMESTEPS = 10000
STUDY_NUM_TRIALS = 30
STUDY_SHOW_PROGRESS_BAR = True
STUDY_TRIAL_TIMEOUT = 240   # seconds
STUDY_TIMEOUT = 2 * STUDY_NUM_TRIALS * STUDY_TRIAL_TIMEOUT
STUDY_TRIAL_EPISODES = 10

'''
The entropy coefficient scales the entropy in the policy loss function which
if increased, rewards higher randomness of actions taken.
'''

ENTROPY_COEFFICIENT = 0.1

'''
* Drives exploration vs exploitation *
For each update, this many steps are run, generating n steps of experiences
from the environment.
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

High Lambda (λ close to 1): A higher value of lambda leads to an advantage
estimate that incorporates more steps into the future, increasing the
variance but potentially capturing a more accurate picture of future rewards.
This can lead to more exploration but might also introduce instability in
learning due to higher variance.

Generalised Advantage Estimation Lambda balances variance and bias in the GAE
calculation. Higher variance means less stable learning but drives more
exploration.

'''
GAE_LAMBDA = 0.9

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
v5.2: added horizontal alignment reward - didn't help. Made slightly worse.
'''

def objective(trial):
    start_time = time.time()  # Record the start time of the trial

    # Define the hyperparameter search space using the trial object
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    gamma = trial.suggest_categorical('gamma', [0.997, 0.999, 0.9999])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.93, 0.95, 0.97])

    # Environment setup
    env = make_vec_env(PROBLEM_NAME, n_envs=4)

    # Model training
    model = PPO('MlpPolicy',
                env,
                learning_rate=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                verbose=1)
    model.learn(total_timesteps=STUDY_NUM_TIMESTEPS)

    # Evaluate the model
    episode_rewards = []
    for _ in range(STUDY_TRIAL_EPISODES):
        obs = env.reset()
        # Initialize done flags for each env
        dones = [False] * env.num_envs
        # Initialize total rewards for each env
        total_rewards = [0] * env.num_envs

        while not all(dones):
            if time.time() - start_time > STUDY_TRIAL_TIMEOUT:
                raise optuna.exceptions.TrialPruned("Trial exceeded time.")

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            for i in range(env.num_envs):
                if not dones[i]:
                    total_rewards[i] += rewards[i]

        average_reward = np.mean(total_rewards)
        episode_rewards.append(average_reward)

    # Compute the average reward of the last 10 episodes
    average_reward = np.mean(episode_rewards)
    return average_reward


def auto_tune():
    # Setup the Optuna study
    print(f"Auto tuning study with {STUDY_NUM_TRIALS} trials and\
                                   {STUDY_NUM_TIMESTEPS} time steps.")
    optuna.logging.get_logger("optuna").setLevel(logging.DEBUG)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,
                   n_trials=STUDY_NUM_TRIALS,
                   show_progress_bar=STUDY_SHOW_PROGRESS_BAR)

    # Fetch and print the best parameters
    print('Best trial:', study.best_trial.params)
    print('Best performance:', study.best_trial.value)

    print("Finished study.")

    return study.best_trial.params


lr = LEARNING_RATE
gamma = DISCOUNT_FACTOR
gae_lambda = GAE_LAMBDA

if AUTO_TUNE:
    results = auto_tune()

    lr = results['lr']
    gamma = results['gamma']
    gae_lambda = results['gae_lambda']

    print("Results of auto tuning...")
    print(f"Learning Rate: {lr}")
    print(f"Discount Factor: {gamma}")
    print(f"GAE Lambda: {gae_lambda}")

# Setup logging for TensorBoard
print("Setting up logging for tensorboard")
os.makedirs(LOG_DIR, exist_ok=True)
logger = configure(LOG_DIR, ["stdout", "tensorboard"])

print("Initialising gymnasium environment")
# env = gym.make(PROBLEM_NAME)
wrapped_env = make_vec_env(PROBLEM_NAME,
                           n_envs=PARALLEL_ENVIRONMENTS,
                           wrapper_class=LunarLanderCustomReward)

model = PPO(policy='MlpPolicy',
            env=wrapped_env,
            learning_rate=lr,
            n_steps=INTERACTIONS_PER_POLICY_UPDATE,
            batch_size=MINI_BATCH_SIZE,
            n_epochs=EPOCHS_PER_UPDATE_CYCLE,
            gamma=gamma,
            gae_lambda=gae_lambda,
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
mean_reward, std_reward = evaluate_policy(model,
                                          model.get_env(),
                                          n_eval_episodes=EVAL_EPISODES,
                                          deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Save evaluation metrics
print("Saving evaluation metrics")
with open("evaluation_results.txt", "w") as file:
    file.write(f"Mean Reward: {mean_reward}\n")
    file.write(f"Standard Deviation of Reward: {std_reward}\n")

wrapped_env.close()

print("Training finished.")
