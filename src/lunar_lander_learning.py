from auto_tune import Tuning
from custom_reward import LunarLanderCustomReward
import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import numpy as np
import datetime


class LunarLanderLearning:
    # Reference value is 10
    EVAL_EPISODES = 10

    EVAL_ENV_INTERACTION_STEPS = 100000

    # Reference value is 16
    PARALLEL_ENVIRONMENTS = 16
    # TODO sort out logging
    LOG_DIR = "logs/"
    RL_ALGORITHM = "PPO"
    POLICY = "MlpPolicy"
    VERBOSITY = 1

    def __init__(self,
                 learning_rate,
                 discount_factor,
                 gae_lambda,
                 entropy_coefficient,
                 epochs,
                 mini_batch_size,
                 interactions_per_policy_update,
                 clip_range,
                 problem_name):
        # TODO: How is model_num retaining state across instances?
        self.lr = learning_rate
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.interactions_per_policy_update = interactions_per_policy_update
        self.clip_range = clip_range
        self.problem_name = problem_name

        print(f"Parallel environments: {self.PARALLEL_ENVIRONMENTS}")
        print(f"Policy: {self.POLICY}")
        print(f"Evaluating episodes: {self.EVAL_EPISODES}")

        self.setup_env()
        self.build_model()

    def __del__(self):
        self.wrapped_env.close()

    def learn(self, total_timesteps):
        print("Learning")
        print("Setting up to save the model")
        model_prefix = self.RL_ALGORITHM + "_"+self.problem_name

        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_save_as = f"{model_prefix}_{date_time}"

        print("Model will be saved as: ", model_save_as)

        self.setup_logging()

        print("Learning")
        self.model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        print("Saving model")
        self.model.save(model_save_as)

    def setup_env(self):
        print("Initialising gymnasium environment")
        self.wrapped_env = make_vec_env(
            self.problem_name,
            n_envs=self.PARALLEL_ENVIRONMENTS,
            wrapper_class=LunarLanderCustomReward
        )

        print(f"Log directory: {self.LOG_DIR}")
        print(f"Verbose: {self.VERBOSITY}")

    def build_model(self):
        '''
        The actor in the actor-critic model
            Is the policy network.
            Maps states to actions.
            For discrete action spaces, the number of output neurons is (normally?!) equal to the number of possible actions.
            Has output which is a probability distribution over the actions. You'd aim to pick the action with the highest probability.
            The action with the highest probability should be the action that maximises the expected future reward.
        The critic
            Is the value network. Maps states to values.
            Calculates an estimate of future rewards, and the actor uses this estimate to update the policy.
            For V(s) the output layer of the critic is a single neuron, and the output is the value function.
            For Q(s,a) the output layer of the critic is a neuron for each possible action, and the output is the Q function.
        TODO: Try V(S) vs Q(S,A) for the critic
        TODO: Try epsilon-greedy policy
        '''

        print(f"Learning rate (not in use right now): {self.lr}\nGamma: {self.gamma}\nGAE Lambda: {self.gae_lambda}\nEntropy Coefficient: {self.entropy_coefficient}\nEpochs: {self.epochs}\nMini Batch Size: {self.mini_batch_size}\nInteractions per Policy Update: {self.interactions_per_policy_update}\nClip Range (not in use right now): {self.clip_range}\nVerbose: 1")
        self.model = PPO(
            policy=self.POLICY,
            env=self.wrapped_env,
            n_steps=self.interactions_per_policy_update,
            batch_size=self.mini_batch_size,
            n_epochs=self.epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.entropy_coefficient,
            verbose=self.VERBOSITY,
            tensorboard_log=self.LOG_DIR
        )

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

    def evaluate_model_manual(self):
        print("Testing the trained agent")
        terminated = False
        truncated = False

        for i in range(self.EVAL_ENV_INTERACTION_STEPS):
            # Reset the environments at the start of each episode
            if i == 0 or terminated or truncated:
                obs = self.wrapped_env.reset()

            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self.wrapped_env.step(action)


            # Extract the action as a single integer value
            # TODO: do we need single_action?
            single_action = action[0] if isinstance(action, np.ndarray) else action

            terminated = dones.any()  # Check if any environment is don

    def auto_tune(self):
        tuning = Tuning(self.problem_name)
        results = tuning.auto_tune()

        self.lr = results['lr']
        self.gamma = results['gamma']
        self.gae_lambda = results['gae_lambda']

        print("Results of auto tuning:")
        print(f"Learning rate: {self.lr}")

        print(f"Gamma: {self.gamma}")
        print(f"gae_lambda: {self.gae_lambda}")
        print("These results are not fed into the training yet-- purely for experimentation for now")
        # TODO: Get the tuning working and get the results into the training

    def setup_logging(self):
        print("Setting up logging for tensorboard")
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.logger = configure(self.LOG_DIR, ["stdout", "tensorboard"])

