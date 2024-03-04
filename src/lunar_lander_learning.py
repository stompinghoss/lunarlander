from auto_tune import Tuning
from custom_reward import LunarLanderCustomReward
import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize


class LunarLanderLearning:
    EVAL_EPISODES = 10
    PARALLEL_ENVIRONMENTS = 4
    LOG_DIR = "logs/"
    RL_ALGORITHM = "PPO"
    MODEL_SAVE_AS = RL_ALGORITHM + "_lunarlander"
    POLICY = "MlpPolicy"
    NORMALISE_OBSERVATIONS = True
    VERBOSITY = 1
    DEVICE = "cpu"
    H1_SIZE = 128
    H2_SIZE = 128

    def __init__(self,
                 learning_rate,
                 discount_factor,
                 gae_lambda,
                 entropy_coefficient,
                 epochs,
                 mini_batch_size,
                 interactions_per_policy_update,
                 clip_range,
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
        self.clip_range = clip_range
        self.max_timesteps = max_timesteps
        self.problem_name = problem_name

        if auto_tune:
            self.auto_tune(problem_name)

        self.setup_logging()
        self.build_model()

    def build_model(self):
        print("Initialising gymnasium environment")
        wrapped_env = make_vec_env(
            self.problem_name,
            n_envs=self.PARALLEL_ENVIRONMENTS,
            wrapper_class=LunarLanderCustomReward
        )

        print(f"Log directory: {self.LOG_DIR}")
        print(f"Verbose: {self.VERBOSITY}")

        # Normalize observations
        if self.NORMALISE_OBSERVATIONS:
            print("Normalising observations")
            wrapped_env = VecNormalize(wrapped_env, norm_obs=True)

        print(f"Building model with algorithm {self.RL_ALGORITHM} and policy {self.POLICY}")
        print(f"Log directory: {self.LOG_DIR}")
        print(f"Verbose: {self.VERBOSITY}")

        policy_kwargs = dict(
            net_arch=[self.H1_SIZE, self.H2_SIZE],  # Specify the sizes of the hidden layers here
        )

        if self.RL_ALGORITHM == "PPO":
            print(f"Learning rate: {self.lr}\nGamma: {self.gamma}\nGAE Lambda: {self.gae_lambda}\nEntropy Coefficient: {self.entropy_coefficient}\nEpochs: {self.epochs}\nMini Batch Size: {self.mini_batch_size}\nInteractions per Policy Update: {self.interactions_per_policy_update}\nClip Range: {self.clip_range}\nVerbose: 1")
            self.model = PPO(
                policy=self.POLICY,
                env=wrapped_env,
                n_steps=self.interactions_per_policy_update,
                batch_size=self.mini_batch_size,
                n_epochs=self.epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                ent_coef=self.entropy_coefficient,
                clip_range=self.clip_range,
                verbose=self.VERBOSITY,
                tensorboard_log=self.LOG_DIR,
                policy_kwargs=policy_kwargs  # Pass the policy_kwargs parameter
            )
        elif self.RL_ALGORITHM == "A2C":
            print("Everything default")
            self.model = A2C(   policy=self.POLICY,
                                env=wrapped_env,
                                device=self.DEVICE,
                                verbose=self.VERBOSITY,
                                tensorboard_log=self.LOG_DIR)

        # Set the logger
        self.model.set_logger(self.logger)

        print("Learning")
        print("Model actor architecture:")
        print(self.model.policy)
        self.model.learn(total_timesteps=self.max_timesteps)
        print("Model critic architecture:")
        print(self.model.policy.critic)

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
