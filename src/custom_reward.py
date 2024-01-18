import gymnasium as gym

'''
For reward shaping, we want to reward:
Minimal engine use. Main engines penalised more than side engines.
Coming in under gravity.
Not firing the engines once in contact with the ground.
v5.2: added horizontal alignment reward - didn't help. Made slightly worse.
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
        ANGLE = 4

        # Engine use penalty when on ground
        GROUND_ENGINE_USE_PENALTY = -0.4
        MAIN_ENGINE_USE_PENALTY = -0.6
        SIDE_ENGINE_USE_PENALTY = -0.06
        EFFICIENT_DESCENT_REWARD = 0.2
        EFFICIENT_DESCENT_MAX_REWARD = 3
        ANGLE_THRESHOLD = 0.1  # Smaller values mean stricter control
        HORIZONTAL_ALIGNMENT_REWARD = 0.1
        ANGLE_PENALTY = -0.1  # Adjust as needed

        # Reward keeping the craft horizontal
        lander_angle = state[ANGLE]

        # Check if either leg is in contact with the ground
        ground_contact = (state[LEFT_LEG_GROUND_CONTACT] or
                          state[RIGHT_LEG_GROUND_CONTACT])
        side_engine_use = action == LEFT_ENGINE or action == RIGHT_ENGINE

        # Main engine costs more than side engines
        if action == MAIN_ENGINE:
            reward += MAIN_ENGINE_USE_PENALTY

        if side_engine_use:
            reward += SIDE_ENGINE_USE_PENALTY

        # Positive reward if the lander is descending (vertical velocity < 0)
        # under gravity and without engines
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
