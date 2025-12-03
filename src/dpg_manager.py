# dpg_manager.py
import os
import threading
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
        self.phase1_success_criterio = 2.5  # Distancia fase 1 em metros
        self.phase2_success_criterio = 8.0  # Distancia fase 2 em metros
        self.phase1_success_threshold = 10  # Vezes fase 1 em metros
        self.phase2_success_threshold = 10  # Vezes fase 2 em metros
        
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
                'learning_rate': 3e-5,      
                'target_policy_noise': 0.05,
                'target_noise_clip': 0.05,  
                'policy_delay': 1,          
                'tau': 0.0005,              
                'gamma': 0.995,             
                'noise_std': 0.05,
            }
        }
        
        # AJUSTES de peso por fase (em relação ao default.json)
        self.phase_weight_adjustments = {
            1: {},  # Fase 1: usa 100% dos pesos do default.json
            2: {    # Fase 2: Foco em Progresso e Estabilidade
                'progress': 3.0,           
                'efficiency_bonus': 15.0,  
                'gait_state_change': 1.0,  
                'foot_clearance': 10.0,   
                'y_axis_deviation_square_penalty': 10.0,  
                'foot_back_penalty': 2.0,   
                'stability_roll': 2.0,      
                'stability_pitch': 2.0,     
                'distance_bonus': 5.0,       
                'success_bonus': 5.0,      
            },
            3: {    # Fase 3: Foco em Sucesso e Velocidade
                'progress': 4.0,           
                'efficiency_bonus': 10.0,  
                'distance_bonus': 10.0,    
                'fall_penalty': 3.0,       
                'yaw_penalty': 2.0,        
                'y_axis_deviation_square_penalty': 20.0, 
                'gait_pattern_cross': 1.5, 
                'foot_clearance': 5.0,     
                'alternating_foot_contact': 2.0, 
                'success_bonus': 5.0,      
                'gait_rhythm': 5.0,        
                'effort_square_penalty': 5.0,  
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
                    self.custom_logger.info(f"🏆 FASE 1 - EPISÓDIO VÁLIDO {self.phase1_success_counter}/10 (distância: {episode_distance:.2f}m)")
        
        elif self.current_phase == 2:
            if episode_distance > self.phase2_success_criterio:
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
        self.corte_antigas = 0.5
        self.corte_piores = 0.5
    
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

    def clear_half_buffer(self, use_threads=True):
        """Remove as transições mais antigas + as piores recompensas."""
        replay_buffer = self.replay_buffer
        current_size = replay_buffer.size()
        buffer_capacity = replay_buffer.buffer_size

        if current_size < 10000:
            if self.custom_logger:
                self.custom_logger.info(f"Buffer muito pequeno ({current_size}), pulando limpeza")
            return

        start_time = time.time()

        try:
            # Determinar estratégia com base no tamanho e configuração
            if current_size < 100000 or not use_threads:
                keep_indices = self._collect_indices_sequential(replay_buffer, current_size, buffer_capacity)
            else:
                keep_indices = self._collect_indices_parallel(replay_buffer, current_size, buffer_capacity)

            # Reconstruir buffer com os índices mantidos
            self._rebuild_buffer_from_indices(replay_buffer, keep_indices, buffer_capacity)

            total_time = time.time() - start_time
            new_size = replay_buffer.size()

            if self.custom_logger:
                self.custom_logger.info(f"Limpeza em {total_time:.2f}s | "
                    f"   Transições: {current_size} → {new_size}")

        except Exception as e:
            if self.custom_logger:
                self.custom_logger.error(f"❌ Erro na limpeza do buffer: {e}")
                import traceback
                self.custom_logger.error(f"Traceback: {traceback.format_exc()[:500]}")

            # Fallback: manter apenas transições recentes
            self._fallback_keep_recent(replay_buffer, buffer_capacity)

    def _collect_indices_sequential(self, replay_buffer, current_size, buffer_capacity):
        """Coleta índices a manter usando abordagem sequencial"""
        all_indices = list(range(buffer_capacity))
        valid_indices = []
        rewards = []

        # Coletar todas as recompensas válidas
        for idx in all_indices:
            try:
                if np.any(replay_buffer.observations[idx] != 0):
                    reward = float(replay_buffer.rewards[idx])
                    valid_indices.append(idx)
                    rewards.append(reward)
            except:
                continue
            
        return self._filter_indices_by_strategy(valid_indices, rewards)

    def _collect_indices_parallel(self, replay_buffer, current_size, buffer_capacity):
        """Coleta índices a manter usando abordagem paralela"""
        import queue

        # Dividir em blocos para processamento paralelo
        block_size = 10000
        num_threads = min(4, os.cpu_count() or 2)
        blocks = []

        for i in range(0, buffer_capacity, block_size):
            end = min(i + block_size, buffer_capacity)
            blocks.append((i, end))

        results_queue = queue.Queue()

        def process_block(start, end):
            """Processa um bloco de índices"""
            block_data = []
            for idx in range(start, end):
                try:
                    if np.any(replay_buffer.observations[idx] != 0):
                        reward = float(replay_buffer.rewards[idx])
                        block_data.append((idx, reward))
                except:
                    continue
            results_queue.put(block_data)

        # Processar blocos em paralelo
        threads = []
        for start, end in blocks[:num_threads * 2]:
            thread = threading.Thread(target=process_block, args=(start, end))
            threads.append(thread)
            thread.start()

        # Coletar resultados
        all_rewards = []
        for thread in threads:
            thread.join()

        while not results_queue.empty():
            all_rewards.extend(results_queue.get())

        # Separar índices e recompensas
        valid_indices = [idx for idx, _ in all_rewards]
        rewards = [reward for _, reward in all_rewards]

        return self._filter_indices_by_strategy(valid_indices, rewards)

    def _filter_indices_by_strategy(self, valid_indices, rewards):
        """Filtra índices baseado na estratégia: mais antigas + piores recompensas"""
        if len(valid_indices) < 1000:
            return valid_indices  # Retorna tudo se muito pequeno

        # Combinar e ordenar por índice (proxy para timestamp)
        indexed_rewards = list(zip(valid_indices, rewards))
        indexed_rewards.sort(key=lambda x: x[0])

        # Separar mais antigas
        half_point = len(indexed_rewards) * self.corte_antigas

        # Ordenar as recentes por recompensa
        recent_rewards = indexed_rewards[half_point:]
        recent_rewards.sort(key=lambda x: x[1])

        # Remover piores recompensas
        remove_count = int(len(recent_rewards) * self.corte_piores)
        keep_rewards = recent_rewards[remove_count:]

        # Coletar índices a manter
        keep_indices = [idx for idx, _ in keep_rewards]

        return keep_indices

    def _rebuild_buffer_from_indices(self, replay_buffer, keep_indices, buffer_capacity):
        """Reconstrói buffer a partir dos índices mantidos"""
        final_count = len(keep_indices)

        if final_count < 1000:
            self._fallback_keep_recent(replay_buffer, buffer_capacity)
            return

        # Preparar novos arrays mantendo a capacidade original
        obs_shape = replay_buffer.observations.shape[1:]
        action_shape = replay_buffer.actions.shape[1:]

        new_observations = np.zeros((buffer_capacity, *obs_shape), dtype=replay_buffer.observations.dtype)
        new_next_observations = np.zeros((buffer_capacity, *obs_shape), dtype=replay_buffer.next_observations.dtype)
        new_actions = np.zeros((buffer_capacity, *action_shape), dtype=replay_buffer.actions.dtype)
        new_rewards = np.zeros((buffer_capacity, 1), dtype=replay_buffer.rewards.dtype)
        new_dones = np.zeros((buffer_capacity, 1), dtype=replay_buffer.dones.dtype)

        # Preencher apenas com as transições mantidas
        for i, idx in enumerate(keep_indices):
            new_observations[i] = replay_buffer.observations[idx]
            new_next_observations[i] = replay_buffer.next_observations[idx]
            new_actions[i] = replay_buffer.actions[idx]
            new_rewards[i] = replay_buffer.rewards[idx]
            new_dones[i] = replay_buffer.dones[idx]

        # Atualizar buffer
        replay_buffer.observations = new_observations
        replay_buffer.next_observations = new_next_observations
        replay_buffer.actions = new_actions
        replay_buffer.rewards = new_rewards
        replay_buffer.dones = new_dones

        # Configurar posição e flag de cheio corretamente
        replay_buffer.pos = min(final_count, buffer_capacity)
        replay_buffer.full = (final_count >= buffer_capacity)

    def _fallback_keep_recent(self, replay_buffer, buffer_capacity):
        """Fallback: mantém apenas as transições mais recentes"""
        current_size = replay_buffer.size()
        keep_count = min(current_size // 2, buffer_capacity)

        # Coletar índices das transições mais recentes
        recent_indices = []

        if replay_buffer.full:
            # Buffer circular
            start_idx = (replay_buffer.pos - keep_count) % buffer_capacity
            for i in range(keep_count):
                idx = (start_idx + i) % buffer_capacity
                recent_indices.append(idx)
        else:
            # Buffer linear
            recent_indices = list(range(max(0, replay_buffer.pos - keep_count), replay_buffer.pos))

        # Coletar transições válidas
        valid_indices = []
        for idx in recent_indices:
            try:
                if np.any(replay_buffer.observations[idx] != 0):
                    valid_indices.append(idx)
            except:
                continue
            
        # Reconstruir buffer
        if valid_indices:
            self._rebuild_buffer_from_indices(replay_buffer, valid_indices, buffer_capacity)

