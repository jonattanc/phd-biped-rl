# dpg_manager.py
import os
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from stable_baselines3 import TD3
import torch.nn.functional as F


class PhaseManager:
    def __init__(self, custom_logger=None):
        self.current_phase = 1
        self.phase_history = []
        self.metrics_buffer = []
        self.buffer_size = 100
        self.custom_logger = custom_logger
        
        self.phase1_success_counter = 0  
        self.phase2_success_counter = 0  
        self.phase1_success_criterio = 2.0  # Distancia fase 1 em metros
        self.phase2_success_criterio = 7.0  # Distancia fase 2 em metros
        self.phase1_success_threshold = 10  # Vezes fase 1 em metros
        self.phase2_success_threshold = 20  # Vezes fase 2 em metros
        
        self.phase_themes = {
            1: "Fase 1 - ESTABILIDADE BÁSICA",
            2: "Fase 2 - PROGRESSO CONSISTENTE", 
            3: "Fase 3 - SUCESSO FINAL"
        }
        
        # HIPERPARÂMETROS ADAPTATIVOS APENAS PARA FASE 2 e 3
        self.adaptive_hyperparams = {
            2: {  # Fase 2: Consolidação com aprendizado mais estável
                'learning_rate': 1e-4,      
                'target_noise_clip': 0.3,  # (de 0.5)
                'policy_delay': 2,         # (de 3)
                'tau': 0.002,              # (de 0.005)
                'gamma': 0.98,             # (de 0.99)
            },
            3: {  # Fase 3: Refinamento com foco em estabilidade     
                'target_noise_clip': 0.25,          
                'tau': 0.003,              
                'gamma': 0.99,             
                'noise_std': 0.1,
            }
        }
        
        # AJUSTES de peso por fase (em relação ao default.json)
        self.phase_weight_adjustments = {
            1: {},  # Fase 1: usa os pesos do default.json
            2: {    # Fase 2: Foco em Progresso e Estabilidade
                'gait_state_change': 1.5,
                'progress': 3.0,           
                'xcom_stability': 3.0, 
                'simple_stability': 2.5, 
                'pitch_forward_bonus': 2.0,   
                'knee_flexion': 2.0,
                'gait_pattern_cross': 10.0, 
                'efficiency_bonus': 10.0,   
                'foot_clearance': 10.0,  
                'hip_extension': 5.0, 
                'distance_bonus': 5.0,       
                'alternating_foot_contact': 2.0,
                'foot_back_penalty': 2.0,    
                'stability_roll': 3.0,      
                'y_axis_deviation_square_penalty': 10.0, 
            },
            3: {    # Fase 3: Foco em Sucesso e Velocidade
                'gait_state_change': 2.0,
                'progress': 4.0,           
                'xcom_stability': 6.0,  
                'simple_stability': 5.0, 
                'pitch_forward_bonus': 5.0,
                'knee_flexion': 3.0,  
                'gait_pattern_cross': 10.0, 
                'efficiency_bonus': 10.0,  
                'foot_clearance': 5.0,    
                'hip_extension': 5.0,  
                'distance_bonus': 10.0,          
                'gait_rhythm': 5.0,        
                'alternating_foot_contact': 3.0, 
                'fall_penalty': 3.0,       
                'yaw_penalty': 2.0,    
                'foot_back_penalty': 3.0,    
                'stability_roll': 4.0,
                'foot_inclination_penalty': 3.0,
                'effort_square_penalty': 5.0,  
                'y_axis_deviation_square_penalty': 20.0, 
                'jerk_penalty': 5.0,  
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
        
        # Contagem de episódios bem-sucedidos
        episode_distance = episode_metrics.get('distance', 0)
        episode_success = episode_metrics.get('success', False)
        
        if self.current_phase == 1:
            if episode_distance > self.phase1_success_criterio:
                self.phase1_success_counter += 1
                if self.custom_logger:
                    self.custom_logger.info(f"🏆 FASE 1 - EPISÓDIO VÁLIDO {self.phase1_success_counter}/"
                                            f"{self.phase1_success_threshold} (distância: {episode_distance:.2f}m)")
        
        elif self.current_phase == 2:
            if episode_distance > self.phase2_success_criterio:
                self.phase2_success_counter += 1
                if self.custom_logger:
                    self.custom_logger.info(f"🏆 FASE 2 - EPISÓDIO VÁLIDO {self.phase2_success_counter}/"
                                            f"{self.phase2_success_threshold} (distância: {episode_distance:.2f}m)")
    
    def should_transition_phase(self):
        """Verifica se deve transicionar de fase"""
        if self.current_phase == 1:
            if self.phase1_success_counter >= self.phase1_success_threshold:
                if self.custom_logger:
                    self.custom_logger.info(f"🎯 FASE 1 CONCLUÍDA: {self.phase1_success_counter} episódios > 2.5m")
                return True
                
        elif self.current_phase == 2:
            if self.phase2_success_counter >= self.phase2_success_threshold:
                if self.custom_logger:
                    self.custom_logger.info(f"🎯 FASE 2 CONCLUÍDA: {self.phase2_success_counter} episódios > 8m")
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
            
            # Reiniciar contadores ao mudar de fase
            if self.current_phase == 2:
                self.phase1_success_counter = 0
            elif self.current_phase == 3:
                self.phase2_success_counter = 0
                
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
        phase_theme = self.phase_themes.get(self.current_phase, "DESCONHECIDA")
        
        # Construir string de ajustes no formato solicitado
        weight_adjustments = self.get_phase_weight_adjustments()
        adjustments_str = " | ".join([f"{k}: {v}x" for k, v in weight_adjustments.items() if v != 1.0])
        
        phase_info = {
            'phase': self.current_phase,
            'phase_theme': phase_theme,
            'adjustments_str': adjustments_str if adjustments_str else "nenhum",
            'current_rps': current_metrics['reward_per_step'],
            'current_dps': current_metrics['distance_per_step'], 
            'current_success': current_metrics['success_rate'],
            'avg_distance': current_metrics['avg_distance'],
            'avg_reward': current_metrics['avg_reward'],
            'phase1_counter': self.phase1_success_counter,
            'phase2_counter': self.phase2_success_counter,
            'weight_adjustments': weight_adjustments,
        }
        
        # Adicionar hiperparâmetros se não for fase 1
        if self.current_phase > 1:
            phase_info['hyperparams'] = self.get_phase_hyperparams()
        
        return phase_info
    
    def set_custom_logger(self, logger):
        """Permite configuração do logger após inicialização"""
        self.custom_logger = logger


class FastTD3(TD3):
    def __init__(self, policy, env, custom_logger=None, **kwargs):
        # Inicializar TD3 normalmente
        super().__init__(policy, env, **kwargs)
        
        self.custom_logger = custom_logger
        self.phase_manager = PhaseManager(custom_logger=custom_logger)
        self.phase_manager.store_original_hyperparams(self)
        
        # Controle de episódios para phase manager
        self.episode_count = 0
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0

        # VERIFICAR SE BUFFER TEM ATRIBUTO 'buffer' OU 'storage'
        self.has_buffer_attribute = (
            hasattr(self.replay_buffer, 'buffer') or 
            hasattr(self.replay_buffer, 'storage') or
            hasattr(self.replay_buffer, '_storage')
        )

        # Limpeza de buffer
        self.old_remove_ratio = 0.3  # Remove 30% mais antigas
        self.padrao_buffer_size = 100000  # Mínimo de transições
    
    def __len__(self):
        """Retorna o tamanho atual do replay buffer para compatibilidade"""
        try:
            if hasattr(self.replay_buffer, 'size'):
                return self.replay_buffer.size()
            elif hasattr(self.replay_buffer, 'buffer'):
                return len(self.replay_buffer.buffer)
            else:
                return 0
        except:
            return 0
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas de fase - interface para simulação"""
        self.episode_count += 1
        self.phase_manager.update_phase_metrics(episode_metrics)
        
        transition_occurred = False
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                transition_occurred = True
                new_phase = self.phase_manager.current_phase
                phase_theme = self.phase_manager.phase_themes.get(new_phase, "DESCONHECIDA")
                
                if self.custom_logger:
                    self.custom_logger.info(f"🎉 FastTD3 - TRANSIÇÃO PARA {phase_theme} (FASE {new_phase})!")
                
                # APLICAR HIPERPARÂMETROS DA NOVA FASE (apenas fase 2 e 3)
                if new_phase > 1:
                    self.apply_phase_hyperparams()
                
        return transition_occurred
    
    def apply_phase_hyperparams(self):
        """Aplica hiperparâmetros da fase atual ao modelo (apenas fase 2 e 3)"""
        hyperparams = self.phase_manager.get_phase_hyperparams()
        
        if not hyperparams:  # Fase 1 ou sem hiperparâmetros definidos
            return
        
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
                
        # Aplicar outros hiperparâmetros
        if 'tau' in hyperparams:
            self.tau = hyperparams['tau']
            
        if 'gamma' in hyperparams:
            self.gamma = hyperparams['gamma']
            
        if 'target_noise_clip' in hyperparams:
            self.target_noise_clip = hyperparams['target_noise_clip']
            
        if 'policy_delay' in hyperparams:
            self.policy_delay = hyperparams['policy_delay']
            
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

    def get_buffer_info(self):
        """Retorna informações detalhadas sobre o buffer"""
        try:
            replay_buffer = self.replay_buffer
            buffer_info = {
                'type': 'unknown',
                'size': 0,
                'capacity': 0
            }

            if hasattr(replay_buffer, 'buffer_size') and hasattr(replay_buffer, 'storage'):
                buffer_info['type'] = 'SB3_ReplayBuffer'
                buffer_info['size'] = replay_buffer.size()
                buffer_info['capacity'] = replay_buffer.buffer_size
                buffer_info['pos'] = replay_buffer.pos
                buffer_info['full'] = replay_buffer.full

            elif hasattr(replay_buffer, 'buffer'):
                buffer_info['type'] = 'deque_buffer'
                buffer_info['size'] = len(replay_buffer.buffer)
                buffer_info['capacity'] = getattr(replay_buffer, 'maxlen', 'unlimited')

            elif hasattr(replay_buffer, '_storage'):
                buffer_info['type'] = 'storage_buffer'
                buffer_info['size'] = replay_buffer.size()

            return buffer_info

        except Exception as e:
            return {'type': 'error', 'error': str(e)}

    def clear_half_buffer(self):
        """Limpa parte do buffer: transições mais antigas + piores recompensas"""
        replay_buffer = self.replay_buffer
        current_size = replay_buffer.size()
        
        # Não limpar se já estiver no mínimo
        if current_size <= self.padrao_buffer_size:
            if self.custom_logger:
                self.custom_logger.info(f"Buffer no mínimo ({current_size}), pulando limpeza")
            return
        
        start_time = time.time()
        
        try:
            # Coletar todas as transições válidas
            valid_indices, rewards = self._get_valid_transitions(replay_buffer)
            
            if len(valid_indices) <= self.padrao_buffer_size:
                return
            
            # Ordenar por índice (mais antigas primeiro)
            sorted_by_age = list(zip(valid_indices, rewards))
            sorted_by_age.sort(key=lambda x: x[0])
            
            # Remover porcentagem das mais antigas
            remove_old = int(len(sorted_by_age) * self.old_remove_ratio)
            if remove_old > 0:
                sorted_by_age = sorted_by_age[remove_old:]
            
            # Garantir mínimo de 100.000
            final_count = len(sorted_by_age)
            if final_count < self.padrao_buffer_size:
                # Manter as melhores até atingir o mínimo
                sorted_by_age.sort(key=lambda x: x[1], reverse=True)  # Melhores primeiro
                sorted_by_age = sorted_by_age[:self.padrao_buffer_size]
            
            # Extrair índices finais
            keep_indices = [idx for idx, _ in sorted_by_age]
            
            # Reconstruir buffer
            self._rebuild_buffer(replay_buffer, keep_indices)
            
            elapsed = time.time() - start_time
            if self.custom_logger:
                self.custom_logger.info(
                    f"Buffer limpo: {current_size} → {len(keep_indices)} "
                    f"(removidas {current_size - len(keep_indices)}) ({elapsed:.2f}s)"
                )
                
        except Exception as e:
            if self.custom_logger:
                self.custom_logger.error(f"Erro ao limpar buffer: {e}")
    
    def _get_valid_transitions(self, replay_buffer):
        """Coleta índices válidos e suas recompensas"""
        valid_indices = []
        rewards = []
        
        for idx in range(replay_buffer.buffer_size):
            # Verificar se transição é válida
            if np.any(replay_buffer.observations[idx] != 0):
                valid_indices.append(idx)
                rewards.append(float(replay_buffer.rewards[idx]))
        
        return valid_indices, rewards
    
    def _rebuild_buffer(self, replay_buffer, keep_indices):
        """Reconstrói buffer com transições mantidas"""
        buffer_capacity = replay_buffer.buffer_size
        keep_count = len(keep_indices)
        
        # Preparar novos arrays
        obs_shape = replay_buffer.observations.shape[1:]
        action_shape = replay_buffer.actions.shape[1:]
        
        new_obs = np.zeros((buffer_capacity, *obs_shape), dtype=replay_buffer.observations.dtype)
        new_next_obs = np.zeros((buffer_capacity, *obs_shape), dtype=replay_buffer.next_observations.dtype)
        new_actions = np.zeros((buffer_capacity, *action_shape), dtype=replay_buffer.actions.dtype)
        new_rewards = np.zeros((buffer_capacity, 1), dtype=replay_buffer.rewards.dtype)
        new_dones = np.zeros((buffer_capacity, 1), dtype=replay_buffer.dones.dtype)
        
        # Copiar transições mantidas
        for i, idx in enumerate(keep_indices):
            new_obs[i] = replay_buffer.observations[idx]
            new_next_obs[i] = replay_buffer.next_observations[idx]
            new_actions[i] = replay_buffer.actions[idx]
            new_rewards[i] = replay_buffer.rewards[idx]
            new_dones[i] = replay_buffer.dones[idx]
        
        # Atualizar buffer
        replay_buffer.observations = new_obs
        replay_buffer.next_observations = new_next_obs
        replay_buffer.actions = new_actions
        replay_buffer.rewards = new_rewards
        replay_buffer.dones = new_dones
        
        # Atualizar estado
        replay_buffer.pos = min(keep_count, buffer_capacity)
        replay_buffer.full = (keep_count >= buffer_capacity)
