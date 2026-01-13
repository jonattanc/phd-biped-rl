# best_model_tracker.py


class BestModelTracker:
    def __init__(self, sim):
        self.TRACK_SUCCESS_TIME = True  # True para salvar modelo com sucesso em menor tempo. False para salvar com maior recompensa.
        self.patience_steps = 0.5e6
        self.original_patience = self.patience_steps

        self.sim = sim

        if self.TRACK_SUCCESS_TIME:
            self.improvement_threshold = 0.01 / 100.0
            self.best_tracked_value = self.sim.episode_init_success_time

        else:
            self.improvement_threshold = 1.0 / 100.0
            self.best_tracked_value = -float("inf")

        self.best_distance = 0.0
        self.steps_since_improvement = 0
        self.last_improvement_steps = 0
        self.auto_save_count = 0
        self.reward_reference = 0

    def update(self):
        """Atualiza tracker com nova recompensa e retorna se houve melhoria"""
        # Atualizar melhor distância se for maior
        if self.sim.episode_distance > self.best_distance:
            self.best_distance = self.sim.episode_distance

        if self.sim.episode_filtered_reward < self.reward_reference:
            self.reward_reference = self.sim.episode_filtered_reward

        # Calcular melhoria percentual
        if self.TRACK_SUCCESS_TIME:
            improvement = (self.best_tracked_value - self.sim.episode_filtered_success_time) / abs(self.best_tracked_value)

        else:
            # Primeira recompensa sempre é considerada melhoria
            if self.best_tracked_value == -float("inf"):
                self.best_tracked_value = self.sim.episode_filtered_reward
                self.last_improvement_steps = self.sim.total_steps
                self.steps_since_improvement = 0
                return False

            normalized_episode_filtered_reward = self.sim.episode_filtered_reward - self.reward_reference
            normalized_best_reward = self.best_tracked_value - self.reward_reference

            if normalized_best_reward == 0:
                improvement = 0.0

            else:
                improvement = (normalized_episode_filtered_reward - normalized_best_reward) / abs(normalized_best_reward)

        if improvement >= self.improvement_threshold:
            if self.TRACK_SUCCESS_TIME:
                self.best_tracked_value = self.sim.episode_filtered_success_time

            else:
                self.best_tracked_value = self.sim.episode_filtered_reward

            self.steps_since_improvement = 0
            self.last_improvement_steps = self.sim.total_steps

            # Apenas conta auto-save se tiver passado do mínimo de steps e distância
            if self.sim.total_steps >= self.sim.agent.minimum_steps_to_save and self.sim.episode_distance >= self.sim.agent.minimum_distance_to_save:
                self.auto_save_count += 1
                return True

            else:
                return False

        else:
            # Atualiza steps sem melhoria mesmo quando não há melhoria
            self.steps_since_improvement = self.sim.total_steps - self.last_improvement_steps
            return False

    def should_pause(self):
        """Verifica se deve pausar por plateau"""
        return self.steps_since_improvement >= self.patience_steps

    def get_status(self):
        """Retorna status atual para logging"""
        return {
            "improvement_threshold": self.improvement_threshold,
            "patience_steps": self.patience_steps,
            "best_tracked_value": self.best_tracked_value,
            "best_distance": self.best_distance,
            "steps_since_improvement": self.steps_since_improvement,
            "last_improvement_steps": self.last_improvement_steps,
            "auto_save_count": self.auto_save_count,
            "reward_reference": self.reward_reference,
            "TRACK_SUCCESS_TIME": self.TRACK_SUCCESS_TIME,
        }
