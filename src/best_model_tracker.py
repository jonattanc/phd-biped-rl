# best_model_tracker.py
import os
import json
import time

class BestModelTracker:
    def __init__(self, improvement_threshold=0.05, patience_steps=500000, checkpoint_steps=300000):
        self.best_reward = -float('inf')
        self.best_distance = 0.0
        self.improvement_threshold = improvement_threshold
        self.patience_steps = patience_steps
        self.checkpoint_steps = checkpoint_steps
        self.steps_since_improvement = 0
        self.total_steps = 0
        self.last_improvement_steps = 0
        self.auto_save_count = 0
        
    def update(self, episode_reward, episode_distance, current_steps):
        """Atualiza tracker com nova recompensa e retorna se houve melhoria"""
        self.total_steps = current_steps
        
        # Atualizar melhor distância se for maior
        if episode_distance > self.best_distance:
            self.best_distance = episode_distance
            
        # Primeira recompensa sempre é considerada melhoria
        if self.best_reward == -float('inf'):
            self.best_reward = episode_reward
            self.last_improvement_steps = current_steps
            self.steps_since_improvement = 0
            self.auto_save_count += 1
            return True, "first_reward"
            
        # Calcular melhoria percentual
        improvement = (episode_reward - self.best_reward) / abs(self.best_reward)
        
        if improvement >= self.improvement_threshold:
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
        return self.steps_since_improvement >= self.patience_steps
        
    def should_checkpoint(self):
        """Verifica se deve fazer checkpoint por tempo"""
        return (self.total_steps - self.last_improvement_steps) >= self.checkpoint_steps
        
    def get_auto_save_filename(self):
        """Gera nome de arquivo para salvamento automático"""
        timestamp = int(time.time())
        return f"best_model_{timestamp}.zip"
        
    def get_status(self):
        """Retorna status atual para logging"""
        return {
            "best_reward": self.best_reward,
            "best_distance": self.best_distance,
            "total_steps": self.total_steps,
            "steps_since_improvement": self.steps_since_improvement,
            "improvement_threshold": self.improvement_threshold,
            "patience_steps": self.patience_steps,
            "auto_save_count": self.auto_save_count
        }