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
        ANGLE = 4

        # Engine use penalty when on ground
        GROUND_ENGINE_USE_PENALTY = -0.4
        MAIN_ENGINE_USE_PENALTY = -0.6
        SIDE_ENGINE_USE_PENALTY = -0.06
        EFFICIENT_DESCENT_REWARD = 0.4
        EFFICIENT_DESCENT_MAX_REWARD = 0.8
        ANGLE_THRESHOLD = 0.1  # Smaller values mean stricter control
        HORIZONTAL_ALIGNMENT_REWARD = 0.1
        ANGLE_PENALTY = -0.1  # Adjust as needed

        # Reward keeping the craft horizontal
        lander_angle = state[ANGLE]

        if abs(lander_angle) < ANGLE_THRESHOLD:
            reward += HORIZONTAL_ALIGNMENT_REWARD
        else:
            reward += ANGLE_PENALTY * abs(lander_angle)

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
        # / abs(state[VERTICAL_VELOCITY]) is to reward more for slower descent
        if state[VERTICAL_VELOCITY] < 0:
            reward += min(EFFICIENT_DESCENT_REWARD / abs(state[VERTICAL_VELOCITY]),
                          EFFICIENT_DESCENT_MAX_REWARD)

        # Penalize if engines are fired while one or both legs on the ground
        if ground_contact and action == MAIN_ENGINE or side_engine_use:
            reward += GROUND_ENGINE_USE_PENALTY

        return reward

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        reward = self.custom_reward(state, reward, action)
        return state, reward, terminated, truncated, info
