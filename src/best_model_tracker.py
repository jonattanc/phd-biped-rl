# best_model_tracker.py
import os
import json

class BestModelTracker:
    def __init__(self, improvement_threshold=0.05, patience_steps=500000, checkpoint_steps=300000):
        self.best_reward = -float('inf')
        self.improvement_threshold = improvement_threshold
        self.patience_steps = patience_steps
        self.checkpoint_steps = checkpoint_steps
        self.steps_since_improvement = 0
        self.total_steps = 0
        self.last_improvement_steps = 0
        
    def update(self, episode_reward, current_steps):
        """Atualiza tracker com nova recompensa e retorna se houve melhoria"""
        self.total_steps = current_steps
        
        # Primeira recompensa sempre é considerada melhoria
        if self.best_reward == -float('inf'):
            self.best_reward = episode_reward
            self.last_improvement_steps = current_steps
            self.steps_since_improvement = 0
            return True
            
        # Calcular melhoria percentual
        improvement = (episode_reward - self.best_reward) / abs(self.best_reward)
        
        if improvement >= self.improvement_threshold:
            self.best_reward = episode_reward
            self.steps_since_improvement = 0
            self.last_improvement_steps = current_steps
            return True  # Indica que deve salvar modelo
        else:
            self.steps_since_improvement = current_steps - self.last_improvement_steps
            return False
            
    def should_pause(self):
        """Verifica se deve pausar por plateau"""
        return self.steps_since_improvement >= self.patience_steps
        
    def should_checkpoint(self):
        """Verifica se deve fazer checkpoint por tempo"""
        return (self.total_steps - self.last_improvement_steps) >= self.checkpoint_steps
        
    def get_status(self):
        """Retorna status atual para logging"""
        return {
            "best_reward": self.best_reward,
            "total_steps": self.total_steps,
            "steps_since_improvement": self.steps_since_improvement,
            "improvement_threshold": self.improvement_threshold,
            "patience_steps": self.patience_steps
        }
        
    def save_state(self, filepath):
        """Salva estado do tracker"""
        state = {
            "best_reward": self.best_reward,
            "total_steps": self.total_steps,
            "steps_since_improvement": self.steps_since_improvement,
            "last_improvement_steps": self.last_improvement_steps
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)
            
    def load_state(self, filepath):
        """Carrega estado do tracker"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.best_reward = state["best_reward"]
            self.total_steps = state["total_steps"]
            self.steps_since_improvement = state["steps_since_improvement"]
            self.last_improvement_steps = state["last_improvement_steps"]
            return True
        return False