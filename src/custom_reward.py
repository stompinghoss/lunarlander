import gymnasium as gym


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

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        reward = self.custom_reward(state, reward, action)
        return state, reward, terminated, truncated, info