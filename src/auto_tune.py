import numpy as np
from stable_baselines3 import PPO
import time
import optuna
from tqdm import tqdm
import logging
from stable_baselines3.common.env_util import make_vec_env

# The number of steps to run when trying to auto tune hyper parameters.
STUDY_NUM_TIMESTEPS = 10000
STUDY_NUM_TRIALS = 30
STUDY_SHOW_PROGRESS_BAR = True
STUDY_TRIAL_TIMEOUT = 240   # seconds
STUDY_TIMEOUT = 2 * STUDY_NUM_TRIALS * STUDY_TRIAL_TIMEOUT
STUDY_TRIAL_EPISODES = 10


class Tuning:
    def __init__(self, problem_name):
        self.problem_name = problem_name

    def objective(self, trial):

        start_time = time.time()  # Record the start time of the trial

        # Define the hyperparameter search space using the trial object
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        gamma = trial.suggest_categorical('gamma', [0.997, 0.999, 0.9999])
        gae_lambda = trial.suggest_categorical('gae_lambda', [0.93, 0.95, 0.97])

        # Environment setup
        env = make_vec_env(self.problem_name, n_envs=4)

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

        average_reward = np.mean(episode_rewards)
        return average_reward

    def auto_tune(self):
        # Setup the Optuna study
        print(f"Auto tuning study with {STUDY_NUM_TRIALS} trials and {STUDY_NUM_TIMESTEPS} time steps.")
        optuna.logging.get_logger("optuna").setLevel(logging.DEBUG)
        study = optuna.create_study(direction='maximize')
        study.optimize( self.objective,
                        n_trials=STUDY_NUM_TRIALS,
                        show_progress_bar=STUDY_SHOW_PROGRESS_BAR)

        # Fetch and print the best parameters
        print('Best trial:', study.best_trial.params)
        print('Best performance:', study.best_trial.value)

        print("Finished study.")

        return study.best_trial.params