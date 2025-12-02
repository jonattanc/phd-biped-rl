# dpg_manager.py
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from stable_baselines3 import TD3
import torch
import torch.nn.functional as F


class PhaseManager:
    def __init__(self):
        self.current_phase = 1
        self.phase_history = []
        self.metrics_buffer = []
        self.buffer_size = 100
        
        # Critérios de transição
        self.phase1_to_2_threshold = 4.0  # distância média > 4m
        self.phase2_to_3_threshold = 9.0  # primeiro sucesso de 9m
        self.success_achieved = False
        
        # AJUSTES de peso por fase (em relação ao default.json)
        self.phase_weight_adjustments = {
            1: {},  # Fase 1: usa 100% dos pesos do default.json
            2: {    # Fase 2: ajusta componentes de eficiência
                'efficiency_bonus': 25.0,  # 2500% - Eficiência avançada
                'progress': 2.0,           # 200% do peso original  
                'gait_state_change': 1.5,  # 150% do peso original
                'foot_clearance': 15.0,    # 1500% do peso original
                'y_axis_deviation_square_penalty': 5.0,  # 500% - Precisão lateral
            },
            3: {    # Fase 3: ajusta componentes de performance
                'efficiency_bonus': 15.0,  # 1500% - Eficiência avançada
                'progress': 2.5,           # 250% do peso original
                'gait_state_change': 2.0,  # 200% do peso original 
                'foot_clearance': 10.0,    # 1000% do peso original 
                'fall_penalty': 2.0,       # 200% - Penalidade máxima por queda
                'y_axis_deviation_square_penalty': 15.0, # 1500% - Precisão lateral
            }
        }
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas do episódio"""
        self.metrics_buffer.append(episode_metrics)
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer.pop(0)
            
        if episode_metrics.get('success', False):
            self.success_achieved = True
    
    def should_transition_phase(self):
        """Verifica se deve transicionar de fase"""
        if len(self.metrics_buffer) < 20:
            return False
            
        current_metrics = self.get_current_metrics()
        
        if self.current_phase == 1:
            # Fase 1 -> 2: distância média > 4m
            if current_metrics['avg_distance'] > self.phase1_to_2_threshold:
                return True
                
        elif self.current_phase == 2:
            # Fase 2 -> 3: primeiro sucesso de 9m alcançado
            if self.success_achieved:
                return True
                
        return False
    
    def get_current_metrics(self):
        """Calcula métricas atuais do buffer"""
        if not self.metrics_buffer:
            return {
                'avg_reward': 0, 
                'avg_distance': 0, 
                'success_rate': 0,
                'reward_per_step': 0,
                'distance_per_step': 0
            }

        avg_reward = np.mean([m.get('reward', 0) for m in self.metrics_buffer])
        avg_distance = np.mean([m.get('distance', 0) for m in self.metrics_buffer])
        success_rate = np.mean([m.get('success', False) for m in self.metrics_buffer])
        total_reward = sum([m.get('reward', 0) for m in self.metrics_buffer])
        total_distance = sum([m.get('distance', 0) for m in self.metrics_buffer])
        total_steps = sum([m.get('steps', 1) for m in self.metrics_buffer])

        if total_steps > 0:
            reward_per_step = total_reward / total_steps
            distance_per_step = total_distance / total_steps
        else:
            reward_per_step = 0
            distance_per_step = 0

        return {
            'avg_reward': avg_reward,
            'avg_distance': avg_distance, 
            'success_rate': success_rate,
            'reward_per_step': reward_per_step,
            'distance_per_step': distance_per_step
        }
    
    def get_phase_weight_adjustments(self):
        """Retorna ajustes de peso para a fase atual"""
        return self.phase_weight_adjustments.get(self.current_phase, {})
    
    def transition_to_next_phase(self):
        """Transiciona para próxima fase"""
        if self.current_phase < 3:
            self.current_phase += 1
            self.phase_history.append({
                'phase': self.current_phase,
                'timestamp': time.time(),
                'metrics': self.get_current_metrics()
            })
            return True
        return False
    
    def get_phase_info(self):
        """Retorna informações detalhadas da fase atual"""
        current_metrics = self.get_current_metrics()
        
        return {
            'phase': self.current_phase,
            'current_rps': current_metrics['reward_per_step'],
            'current_dps': current_metrics['distance_per_step'], 
            'current_success': current_metrics['success_rate'],
            'avg_distance': current_metrics['avg_distance'],
            'avg_reward': current_metrics['avg_reward'],
            'success_achieved': self.success_achieved,
            'weight_adjustments': self.get_phase_weight_adjustments()
        }


class FastTD3(TD3):
    def __init__(self, policy, env, custom_logger=None, **kwargs):
        # Inicializar TD3 normalmente
        super().__init__(policy, env, **kwargs)
        
        self.custom_logger = custom_logger
        self.phase_manager = PhaseManager()
        
        # Controle de episódios para phase manager
        self.episode_count = 0
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0

    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas de fase - interface para simulação"""
        self.episode_count += 1
        self.phase_manager.update_phase_metrics(episode_metrics)
        
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                new_phase = self.phase_manager.current_phase
                if self.custom_logger:
                    self.custom_logger.info(f"🎉 FastTD3 - TRANSIÇÃO PARA FASE {new_phase}!")
                    current_metrics = self.phase_manager.get_current_metrics()
                    self.custom_logger.info(f"🏆 Métricas: Distância média: {current_metrics['avg_distance']:.2f}m, "
                                          f"Recompensa/step: {current_metrics['reward_per_step']:.3f}, "
                                          f"Sucesso: {current_metrics['success_rate']:.1%}")
                return True
        return False
    
    def get_phase_info(self):
        return self.phase_manager.get_phase_info()
    
    def get_phase_weight_adjustments(self):
        return self.phase_manager.get_phase_weight_adjustments()

    def get_dpg_status(self):
        """Retorna status completo do sistema DPG integrado"""
        return {
            "enabled": True,
            "episode_count": self.episode_count,
            "buffer_size": len(self.replay_buffer),
            "phase": self.phase_manager.current_phase,
            "phase_info": self.get_phase_info()
        }