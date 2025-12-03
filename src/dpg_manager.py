# dpg_manager.py
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from stable_baselines3 import TD3
import torch
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
        self.phase1_success_threshold = 10  
        self.phase2_success_threshold = 10 
        
        self.phase_themes = {
            1: "Fase 1 - ESTABILIDADE BÁSICA",
            2: "Fase 2 - PROGRESSO CONSISTENTE", 
            3: "Fase 3 - SUCESSO FINAL"
        }
        
        # HIPERPARÂMETROS ADAPTATIVOS APENAS PARA FASE 2 e 3
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
                'foot_clearance': 10.0,    # 1000% - Garantir elevação adequada dos pés
                'y_axis_deviation_square_penalty': 10.0,  # 1000% - Manter trajetória reta
                'foot_back_penalty': 2.0,   # 200% - Evitar movimento para trás
                'stability_roll': 2.0,      # 200% - Manter equilíbrio lateral
                'stability_pitch': 2.0,     # 200% - Manter inclinação frontal
                'distance_bonus': 5.0,      # 500% 
                'success_bonus': 5.0,       # 500% - Premiar sucesso antecipado
            },
            3: {    # Fase 3: Foco em Sucesso e Velocidade
                'progress': 4.0,           # 400% do peso original
                'efficiency_bonus': 10.0,  # 1000% - Eficiência avançada
                'distance_bonus': 10.0,    # 1000% - Distância é crítica
                'fall_penalty': 3.0,       # 300% - Queda inaceitável
                'yaw_penalty': 2.0,        # 200% - Desvio fatal
                'y_axis_deviation_square_penalty': 20.0, # 2000% - Trajetória precisa
                'gait_pattern_cross': 1.5, # 150% - Padrão cruzado aprimorado
                'foot_clearance': 5.0,     # 500% - Clearance consistente
                'alternating_foot_contact': 2.0, # 200% - Alternância perfeita
                'success_bonus': 5.0,      # 500% - Sucesso vale muito
                'gait_rhythm': 5.0,        # 500% - Ritmo consistente
                'effort_square_penalty': 5.0,  # 500% - Movimentos suaves
                'jerk_penalty': 5.0,       # 500% - Suavidade na transição
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
            # Fase 1: episódios > 2.5m
            if episode_distance > 2.5:
                self.phase1_success_counter += 1
                if self.custom_logger:
                    self.custom_logger.info(f"🏆 FASE 1 - EPISÓDIO VÁLIDO {self.phase1_success_counter}/10 (distância: {episode_distance:.2f}m)")
        
        elif self.current_phase == 2:
            # Fase 2: episódios > 8m
            if episode_distance > 8.0:
                self.phase2_success_counter += 1
                if self.custom_logger:
                    self.custom_logger.info(f"🏆 FASE 2 - EPISÓDIO VÁLIDO {self.phase2_success_counter}/10 (distância: {episode_distance:.2f}m)")
    
    def should_transition_phase(self):
        """Verifica se deve transicionar de fase"""
        if self.current_phase == 1:
            if self.phase1_success_counter >= self.phase1_success_threshold:
                if self.custom_logger:
                    self.custom_logger.info(f"🎯 FASE 1 CONCLUÍDA: {self.phase1_success_counter} episódios > 3m")
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
        
        # Armazenar hiperparâmetros originais
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
        """Limpa a metade inicial do buffer de replay para o buffer do SB3"""
        try:
            replay_buffer = self.replay_buffer

            # Método específico do SB3 ReplayBuffer
            if hasattr(replay_buffer, 'buffer_size') and hasattr(replay_buffer, 'pos'):
                # É o buffer do SB3
                current_size = replay_buffer.size()
                buffer_capacity = replay_buffer.buffer_size

                if current_size > 1000:
                    half_size = current_size // 2

                    # Criar um NOVO buffer com metade da capacidade
                    # e copiar apenas as transições mais recentes

                    # 1. Coletar todas as transições válidas
                    all_transitions = []

                    # Se o buffer não está cheio, as transições válidas vão de 0 a pos-1
                    if not replay_buffer.full:
                        valid_indices = range(replay_buffer.pos)
                    else:
                        # Buffer cheio: circular, precisa percorrer do pos até o final e depois do início até pos-1
                        valid_indices = list(range(replay_buffer.pos, buffer_capacity)) + list(range(0, replay_buffer.pos))

                    # Coletar todas as transições
                    for idx in valid_indices:
                        try:
                            transition = {
                                'obs': replay_buffer.observations[idx].copy(),
                                'next_obs': replay_buffer.next_observations[idx].copy(),
                                'action': replay_buffer.actions[idx].copy(),
                                'reward': replay_buffer.rewards[idx].copy(),
                                'done': replay_buffer.dones[idx].copy()
                            }
                            all_transitions.append(transition)
                        except Exception as e:
                            if self.custom_logger:
                                self.custom_logger.warning(f"🔄 FastTD3 - Erro ao coletar transição {idx}: {e}")

                    # 2. Manter apenas as últimas half_size transições
                    if len(all_transitions) > half_size:
                        recent_transitions = all_transitions[-half_size:]

                        # 3. Criar um NOVO buffer com metade da capacidade
                        from stable_baselines3.common.buffers import ReplayBuffer
                        import torch as th

                        # Obter dimensões das observações e ações
                        obs_shape = replay_buffer.observations.shape[1:]
                        action_shape = replay_buffer.actions.shape[1:]

                        # Criar novo buffer com metade da capacidade original
                        new_buffer = ReplayBuffer(
                            buffer_size=half_size,
                            observation_space=self.observation_space,
                            action_space=self.action_space,
                            device=self.device,
                            n_envs=self.n_envs
                        )

                        # 4. Adicionar transições ao novo buffer
                        for transition in recent_transitions:
                            new_buffer.add(
                                transition['obs'],
                                transition['next_obs'],
                                transition['action'],
                                transition['reward'],
                                transition['done'],
                                [{}]  # infos vazio
                            )

                        # 5. Substituir o buffer antigo pelo novo
                        self.replay_buffer = new_buffer

                    else:
                        if self.custom_logger:
                            self.custom_logger.info(f"🔄 FastTD3 - Buffer muito pequeno para limpar: {len(all_transitions)} transições")

            # SEGUNDA TENTATIVA: Buffer com estrutura de deque
            elif hasattr(replay_buffer, 'buffer') and isinstance(replay_buffer.buffer, (list, deque)):
                buffer_list = list(replay_buffer.buffer)
                buffer_size = len(buffer_list)

                if buffer_size > 1000:
                    half_size = buffer_size // 2
                    recent_transitions = buffer_list[half_size:]

                    # Atualizar o buffer mantendo apenas transições recentes
                    replay_buffer.buffer = deque(recent_transitions, maxlen=replay_buffer.buffer.maxlen if hasattr(replay_buffer.buffer, 'maxlen') else None)

                    if self.custom_logger:
                        self.custom_logger.info(f"🔄 FastTD3 - Buffer (deque) reduzido: {buffer_size} → {len(recent_transitions)} transições")

            # TERCEIRA TENTATIVA: Outra estrutura conhecida
            elif hasattr(replay_buffer, '_storage'):
                try:
                    storage_size = replay_buffer.size()
                    if storage_size > 1000:
                        half_size = storage_size // 2

                        # Tentar abordagem genérica: manter apenas índices recentes
                        if hasattr(replay_buffer, 'pos'):
                            # Simplesmente mover a posição para trás (perdendo transições antigas)
                            replay_buffer.pos = max(0, replay_buffer.pos - half_size)

                            if self.custom_logger:
                                self.custom_logger.info(f"🔄 FastTD3 - Buffer ajustado via posição: mantidas últimas {half_size} transições")
                except:
                    pass
                
            # SE NENHUMA DAS ANTERIORES FUNCIONOU, TENTAR UMA ABORDAGEM RADICAL
            else:
                if self.custom_logger:
                    self.custom_logger.warning(f"🔄 FastTD3 - Estrutura de buffer não reconhecida. Tentando reinicialização parcial...")

                # Tentar recriar o buffer do zero
                try:
                    current_size = replay_buffer.size()
                    if current_size > 1000:
                        half_size = current_size // 2

                        # Importar o buffer do SB3
                        from stable_baselines3.common.buffers import ReplayBuffer

                        # Criar novo buffer vazio
                        new_buffer = ReplayBuffer(
                            buffer_size=half_size,
                            observation_space=self.observation_space,
                            action_space=self.action_space,
                            device=self.device,
                            n_envs=self.n_envs
                        )

                        # Substituir o buffer
                        self.replay_buffer = new_buffer

                        if self.custom_logger:
                            self.custom_logger.info(f"🔄 FastTD3 - Buffer recriado vazio. Capacidade: {half_size}")

                except Exception as e:
                    if self.custom_logger:
                        self.custom_logger.error(f"🔄 FastTD3 - ERRO ao recriar buffer: {str(e)}")

        except Exception as e:
            if self.custom_logger:
                import traceback
                self.custom_logger.error(f"🔄 FastTD3 - ERRO CRÍTICO ao limpar buffer: {str(e)}")
                self.custom_logger.error(f"🔄 FastTD3 - Traceback: {traceback.format_exc()}")