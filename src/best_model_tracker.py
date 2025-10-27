# best_model_tracker.py
import time


class BestModelTracker:
    def __init__(self):
        self.improvement_threshold = 0.05
        self.patience_steps = 3e6

        self.best_reward = -float("inf")
        self.best_distance = 0.0
        self.steps_since_improvement = 0
        self.total_steps = 0
        self.last_improvement_steps = 0
        self.auto_save_count = 0

        # Adicionar flag para controle de estado
        self._active = True

    def update(self, episode_reward, episode_distance, current_steps, minimum_steps_to_save):
        """Atualiza tracker com nova recompensa e retorna se houve melhoria"""
        if not self._active:
            return False, "tracker_inactive"

        self.total_steps = current_steps

        # Atualizar melhor distância se for maior
        if episode_distance > self.best_distance:
            self.best_distance = episode_distance

        if episode_reward < self.reward_reference:
            self.reward_reference = episode_reward

        # Primeira recompensa sempre é considerada melhoria
        if self.best_reward == -float("inf"):
            self.best_reward = episode_reward
            self.last_improvement_steps = current_steps
            self.steps_since_improvement = 0
            return False, "first_reward"

        # Calcular melhoria percentual
        normalized_episode_reward = episode_reward - self.reward_reference
        normalized_best_reward = self.best_reward - self.reward_reference

        if normalized_best_reward == 0:
            improvement = 0.0

        else:
            improvement = (normalized_episode_reward - normalized_best_reward) / abs(normalized_best_reward)

        if improvement >= self.improvement_threshold and self.total_steps >= minimum_steps_to_save:
            self.best_reward = episode_reward
            self.steps_since_improvement = 0
            self.last_improvement_steps = current_steps
            self.auto_save_count += 1
            return True, f"improvement_{improvement:.2%}"
        else:
            self.steps_since_improvement = current_steps - self.last_improvement_steps
            return False, "no_improvement"

    def should_pause(self):
        """Verifica se deve pausar por plateau"""
        if not self._active:
            return False
        return self.steps_since_improvement >= self.patience_steps

    def get_auto_save_filename(self):
        """Gera nome de arquivo para salvamento automático"""
        timestamp = int(time.time())
        return f"best_model_{timestamp}.zip"

    def get_status(self):
        """Retorna status atual para logging"""
        if not self._active:
            return {"status": "inactive"}

        return {
            "best_reward": self.best_reward,
            "best_distance": self.best_distance,
            "total_steps": self.total_steps,
            "steps_since_improvement": self.steps_since_improvement,
            "improvement_threshold": self.improvement_threshold,
            "patience_steps": self.patience_steps,
            "auto_save_count": self.auto_save_count,
            "status": "active",
        }

    def deactivate(self):
        """Desativa o tracker para evitar erros"""
        self._active = False

    def reset(self):
        """Reseta o tracker para novo treinamento"""
        self._active = True
        self.best_reward = -float("inf")
        self.best_distance = 0.0
        self.steps_since_improvement = 0
        self.total_steps = 0
        self.last_improvement_steps = 0
        self.auto_save_count = 0
        self.reward_reference = 0
