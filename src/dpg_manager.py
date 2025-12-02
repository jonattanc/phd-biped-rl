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
        self.phase1_to_2_threshold = 2.5  # distância média > 2.5m
        self.phase2_to_3_threshold = 9.0  # primeiro sucesso de 9m
        self.success_achieved = False
        
        # HIPERPARÂMETROS ADAPTATIVOS APENAS PARA FASE 2 e 3
        # Fase 1: Usa hiperparâmetros padrão do TD3 (definidos no agent.py)
        self.adaptive_hyperparams = {
            2: {  # Fase 2: Consolidação com aprendizado mais estável
                'learning_rate': 1e-4,      
                'target_noise_clip': 0.3,   # Reduzido para 0.3 (de 0.5)
                'policy_delay': 2,          # Mais frequente (de 3)
                'tau': 0.002,              # Atualização mais rápida (de 0.005)
                'gamma': 0.98,             # Ligeiramente maior (de 0.99)
            },
            3: {  # Fase 3: Refinamento com foco em estabilidade
                'learning_rate': 5e-5,      # Reduzido para refinamento
                'target_noise_clip': 0.1,   # Mínimo ruído para política estável
                'policy_delay': 1,          # Atualização mais frequente
                'tau': 0.001,              # Atualização mais suave
                'gamma': 0.99,             # Igual ao padrão
            }
        }
        
        # AJUSTES de peso por fase (em relação ao default.json)
        self.phase_weight_adjustments = {
            1: {},  # Fase 1: usa 100% dos pesos do default.json
            2: {    # Fase 2: Foco em Progresso e Estabilidade
                'progress': 3.0,           # 300% do peso original 
                'efficiency_bonus': 15.0,  # 1500% - Foco em eficiência energética
                'gait_state_change': 1.0,  # 100% - Mantém normal
                'foot_clearance': 8.0,     # 800% - Garantir elevação adequada dos pés
                'y_axis_deviation_square_penalty': 8.0,  # 800% - Manter trajetória reta
                'foot_back_penalty': 2.0,   # 200% - Evitar movimento para trás
                'stability_roll': 2.0,      # 200% - Manter equilíbrio lateral
                'stability_pitch': 2.0,     # 200% - Manter inclinação frontal
                'distance_bonus': 1.5,      # 150% 
                'success_bonus': 2.0,       # 200% - Premiar sucesso antecipado
            },
            3: {    # Fase 3: Foco em Sucesso e Velocidade
                'progress': 4.0,           # 400% do peso original
                'efficiency_bonus': 10.0,  # 1000% - Eficiência avançada
                'distance_bonus': 3.0,     # 300% - Distância é crítica
                'fall_penalty': 3.0,       # 300% - Queda inaceitável
                'yaw_penalty': 2.0,        # 200% - Desvio fatal
                'y_axis_deviation_square_penalty': 20.0, # 2000% - Trajetória precisa
                'gait_pattern_cross': 1.5, # 150% - Padrão cruzado aprimorado
                'foot_clearance': 5.0,     # 500% - Clearance consistente
                'alternating_foot_contact': 2.0, # 200% - Alternância perfeita
                'success_bonus': 5.0,      # 500% - Sucesso vale muito
                'gait_rhythm': 2.0,        # 200% - Ritmo consistente
                'effort_square_penalty': 2.0,  # 200% - Movimentos suaves
                'jerk_penalty': 1.5,       # 150% - Suavidade na transição
            }
        }

        # Armazenar hiperparâmetros originais do TD3
        self.original_hyperparams = {}
    
    def store_original_hyperparams(self, model):
        """Armazena hiperparâmetros originais do TD3"""
        self.original_hyperparams = {
            'learning_rate': model.learning_rate,
            'tau': model.tau,
            'gamma': model.gamma,
            'target_noise_clip': model.target_noise_clip,
            'policy_delay': model.policy_delay,
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
            # Fase 1 -> 2: distância média
            if current_metrics['avg_distance'] > self.phase1_to_2_threshold:
                return True
                
        elif self.current_phase == 2:
            # Fase 2 -> 3: primeiro sucesso de 9m alcançado
            if (current_metrics['avg_distance'] > self.phase2_to_3_threshold and 
                current_metrics['success_rate'] > 0.2):
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
    
    def get_phase_hyperparams(self):
        """Retorna hiperparâmetros para a fase atual"""
        return self.adaptive_hyperparams.get(self.current_phase, {})
    
    def get_original_hyperparams(self):
        """Retorna hiperparâmetros originais do TD3"""
        return self.original_hyperparams
    
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
        
        phase_info = {
            'phase': self.current_phase,
            'current_rps': current_metrics['reward_per_step'],
            'current_dps': current_metrics['distance_per_step'], 
            'current_success': current_metrics['success_rate'],
            'avg_distance': current_metrics['avg_distance'],
            'avg_reward': current_metrics['avg_reward'],
            'success_achieved': self.success_achieved,
            'weight_adjustments': self.get_phase_weight_adjustments(),
        }
        
        # Adicionar hiperparâmetros se não for fase 1
        if self.current_phase > 1:
            phase_info['hyperparams'] = self.get_phase_hyperparams()
        
        return phase_info
    


class FastTD3(TD3):
    def __init__(self, policy, env, custom_logger=None, **kwargs):
        # Inicializar TD3 normalmente
        super().__init__(policy, env, **kwargs)
        
        self.custom_logger = custom_logger
        self.phase_manager = PhaseManager()
        
        # Armazenar hiperparâmetros originais
        self.phase_manager.store_original_hyperparams(self)
        
        # Controle de episódios para phase manager
        self.episode_count = 0
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0
    
    def __len__(self):
        """Retorna o tamanho atual do replay buffer para compatibilidade"""
        return self.replay_buffer.size() if hasattr(self.replay_buffer, 'size') else len(self.replay_buffer.buffer) if hasattr(self.replay_buffer, 'buffer') else 0

    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas de fase - interface para simulação"""
        self.episode_count += 1
        self.phase_manager.update_phase_metrics(episode_metrics)
        
        transition_occurred = False
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                transition_occurred = True
                new_phase = self.phase_manager.current_phase
                if self.custom_logger:
                    self.custom_logger.info(f"🎉 FastTD3 - TRANSIÇÃO PARA FASE {new_phase}!")
                    current_metrics = self.phase_manager.get_current_metrics()
                    self.custom_logger.info(f"🏆 Métricas: Distância média: {current_metrics['avg_distance']:.2f}m, "
                                          f"Recompensa/step: {current_metrics['reward_per_step']:.3f}, "
                                          f"Sucesso: {current_metrics['success_rate']:.1%}")
                
                # APLICAR HIPERPARÂMETROS DA NOVA FASE (apenas fase 2 e 3)
                if new_phase > 1:
                    self.apply_phase_hyperparams()
                
        return transition_occurred
    
    def apply_phase_hyperparams(self):
        """Aplica hiperparâmetros da fase atual ao modelo (apenas fase 2 e 3)"""
        hyperparams = self.phase_manager.get_phase_hyperparams()
        
        if not hyperparams:  # Fase 1 ou sem hiperparâmetros definidos
            return
        
        if self.custom_logger:
            self.custom_logger.info(f"🔄 FastTD3 - Aplicando hiperparâmetros da Fase {self.phase_manager.current_phase}")
        
        # Aplicar learning rate
        if 'learning_rate' in hyperparams:
            new_lr = hyperparams['learning_rate']
            if new_lr != self.learning_rate:
                self.learning_rate = new_lr
                # Atualizar otimizadores
                for param_group in self.actor.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                for param_group in self.critic.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                if self.custom_logger:
                    self.custom_logger.info(f"  Learning rate: {self.learning_rate}")
        
        # Aplicar outros hiperparâmetros
        if 'tau' in hyperparams:
            self.tau = hyperparams['tau']
            if self.custom_logger:
                self.custom_logger.info(f"  Tau: {self.tau}")
        
        if 'gamma' in hyperparams:
            self.gamma = hyperparams['gamma']
            if self.custom_logger:
                self.custom_logger.info(f"  Gamma: {self.gamma}")
        
        if 'target_noise_clip' in hyperparams:
            self.target_noise_clip = hyperparams['target_noise_clip']
            if self.custom_logger:
                self.custom_logger.info(f"  Target noise clip: {self.target_noise_clip}")
        
        if 'policy_delay' in hyperparams:
            self.policy_delay = hyperparams['policy_delay']
            if self.custom_logger:
                self.custom_logger.info(f"  Policy delay: {self.policy_delay}")
    
    def restore_original_hyperparams(self):
        """Restaura hiperparâmetros originais do TD3"""
        original = self.phase_manager.get_original_hyperparams()
        
        if original:
            self.learning_rate = original['learning_rate']
            self.tau = original['tau']
            self.gamma = original['gamma']
            self.target_noise_clip = original['target_noise_clip']
            self.policy_delay = original['policy_delay']
            
            # Atualizar otimizadores
            for param_group in self.actor.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            for param_group in self.critic.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
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
    
    def clear_half_buffer(self):
        """Limpa a metade inicial do buffer de replay"""
        if hasattr(self.replay_buffer, 'buffer'):
            buffer_size = len(self.replay_buffer.buffer)
            half_size = buffer_size // 2
            
            # Manter apenas a segunda metade do buffer
            self.replay_buffer.buffer = deque(list(self.replay_buffer.buffer)[half_size:])
            
            if self.custom_logger:
                self.custom_logger.info(f"🔄 FastTD3 - Buffer reduzido: {buffer_size} → {len(self.replay_buffer.buffer)} transições")