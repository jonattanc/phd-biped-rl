import math


class Agent:
    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_action(self):
        # For now, returns a fixed sinusoidal velocity
        return -15 * math.sin(self.state / 50)
