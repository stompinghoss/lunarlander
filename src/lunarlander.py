from lunar_lander_learning import LunarLanderLearning


'''
Notes on hyper parameters and approach:

Shifting from DQN to PPO definitely helped.
Shifting from low level pytorch to sb3 definitely made things easier.
Higher exploration definitely helps.
Higher exploration needs more time steps to run.
But, reward shaping definitely has the highest impact, at least so far.
Some comments/explanations courtesy of ChatGPT4 answering my questions.
'''

PROBLEM_NAME = "LunarLander-v2"

'''
 Maximum timesteps specifies how many times the agent will take an action in
 the environment.
 Each timestep typically represents one action-selection and the observation
 of its outcome (the next state and reward).
 This is a measure of how long the training will run. More timesteps usually
 mean longer training, during which the agent has more opportunities to learn
 from the environment.
'''

# Reference value is 2e6
MAX_TIMESTEPS = 2e6

'''
The entropy coefficient scales the entropy in the policy loss function which
if increased, rewards higher randomness of actions taken.
'''
# Reference value is 0.1
ENTROPY_COEFFICIENT = 0.1

'''
* Drives exploration vs exploitation *
For each update, this many steps are run, generating n steps of experiences from the environment.
In a vectorized environment, this many steps will be taken per environment.
'''
# Reference value is 1024
INTERACTIONS_PER_POLICY_UPDATE = 1024

'''
How much experience is collected before learning.
Those experiences are then sampled into batches. Those batches may in turn be
samples into smaller batches. This paramater is the mini batch size.
'''
# Reference value is 64
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
# Reference value is 0.999
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
# Reference value is 0.95
GAE_LAMBDA = 0.95


# Reference value is 0.0003
LEARNING_RATE = 0.0003


# Utilise an approach to calculate good hyper parameters
AUTO_TUNE = False

'''
The clipping parameter (epsilon) is used in the PPO objective to clip the
probability ratio. A larger epsilon permits larger policy updates, which
'''
# Reference value is 0.1
CLIP_RANGE = 0.1

print("Starting training.")
learner = LunarLanderLearning(LEARNING_RATE,
                              DISCOUNT_FACTOR,
                              GAE_LAMBDA,
                              ENTROPY_COEFFICIENT,
                              EPOCHS_PER_UPDATE_CYCLE,
                              MINI_BATCH_SIZE,
                              INTERACTIONS_PER_POLICY_UPDATE,
                              CLIP_RANGE,
                              PROBLEM_NAME)

if AUTO_TUNE:
    learner.auto_tune()

print("About to learn.")
learner.learn(total_timesteps=MAX_TIMESTEPS)

 # TODO: Do the eval with a separate model
learner.evaluate_model()
learner.evaluate_model_manual()

print("Training finished.")
