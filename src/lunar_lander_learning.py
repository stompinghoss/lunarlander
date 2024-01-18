from auto_tune import Tuning
import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from custom_reward import LunarLanderCustomReward


class LunarLanderLearning:
    EVAL_EPISODES = 10
    EVAL_ENV_INTERACTION_STEPS = 100000
    PARALLEL_ENVIRONMENTS = 16
    LOG_DIR = "logs/"
    MODEL_SAVE_AS = "ppo_lunarlander"

    '''
    * Drives exploration vs exploitation *
    For each update, this many steps are run, generating n steps of experiences
    from the environment.
    In a vectorized environment, this many steps will be taken per environment.
    '''

    def __init__(self,
                 learning_rate,
                 discount_factor,
                 gae_lambda,
                 entropy_coefficient,
                 epochs,
                 mini_batch_size,
                 interactions_per_policy_update,
                 max_timesteps,
                 auto_tune,
                 problem_name):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.interactions_per_policy_update = interactions_per_policy_update
        self.max_timesteps = max_timesteps
        self.problem_name = problem_name

        if auto_tune:
            self.auto_tune(problem_name)

        self.setup_logging()
        self.build_model()

    def build_model(self):
        print("Building model")
        print("Initialising gymnasium environment")
        wrapped_env = make_vec_env( self.problem_name,
                                    n_envs=self.PARALLEL_ENVIRONMENTS,
                                    wrapper_class=LunarLanderCustomReward)

        # Leave learning rate to the default-- for now.
        self.model = PPO(   policy='MlpPolicy',
                            env=wrapped_env,
                            n_steps=self.interactions_per_policy_update,
                            batch_size=self.mini_batch_size,
                            n_epochs=self.epochs,
                            gamma=self.gamma,
                            gae_lambda=self.gae_lambda,
                            ent_coef=self.entropy_coefficient,
                            verbose=1,
                            tensorboard_log=self.LOG_DIR)

        # Set the logger
        self.model.set_logger(self.logger)

        print("Learning")
        self.model.learn(total_timesteps=self.max_timesteps)

        # Save the trained model
        print("Saving model")
        self.model.save(self.MODEL_SAVE_AS)

        self.evaluate_model()

        wrapped_env.close()

    def evaluate_model(self):
        print("Evaluating policy")
        mean_reward, std_reward = evaluate_policy(  self.model,
                                                    self.model.get_env(),
                                                    n_eval_episodes=self.EVAL_EPISODES,
                                                    deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

        # Save evaluation metrics
        print("Saving evaluation metrics")
        with open("evaluation_results.txt", "w") as file:
            file.write(f"Mean Reward: {mean_reward}\n")
            file.write(f"Standard Deviation of Reward: {std_reward}\n")

    def auto_tune(self):
        tuning = Tuning(self.problem_name)
        results = tuning.auto_tune()

        self.lr = results['lr']
        self.gamma = results['gamma']
        self.gae_lambda = results['gae_lambda']

    def setup_logging(self):
        print("Setting up logging for tensorboard")
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.logger = configure(self.LOG_DIR, ["stdout", "tensorboard"])