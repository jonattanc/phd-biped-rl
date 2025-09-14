import math


class Agent:
    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_action(self):
        # For now, returns a fixed velocity

        velocity = 0
        # velocity = -15 * math.sin(self.state / 50)

        target_velocities = [velocity] * 4

        return target_velocities
