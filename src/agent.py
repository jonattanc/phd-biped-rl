import random

class Agent:
    def __init__(self):
        self.revolute_indices = []
        self.len_revolute_indices = 0

    def set_revolute_indices(self, revolute_indices):
        self.revolute_indices = revolute_indices
        self.len_revolute_indices = len(revolute_indices)

    def set_state(self, state):
        pass

    def get_action(self):
        # Gera uma velocidade aleatória entre -10 e 10 para cada junta revolute
        target_velocities = [random.uniform(-10, 10) for _ in range(self.len_revolute_indices)]
        return target_velocities