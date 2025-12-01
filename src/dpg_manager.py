# dpg_manager.py
import numpy as np
from collections import deque
import random
from typing import List, Dict, Any
import time
from dataclasses import dataclass
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch
import torch.nn.functional as F

class Experience:
    def __init__(self, state, action, reward, next_state, done, metrics):
        self.state = np.array(state, dtype=np.float32)
        self.action = np.array(action, dtype=np.float32)
        self.reward = float(reward)
        self.next_state = np.array(next_state, dtype=np.float32)
        self.done = done
        self.quality = self._calculate_quality(metrics)
        
    def _calculate_quality(self, metrics):
        """Qualidade baseada em recompensa e estabilidade"""
        reward = self.reward
        
        if reward > 0:
            base_quality = 0.5 + min(0.5, reward / 100.0) 
        else:
            base_quality = max(0.1, 0.3 + (reward / 500.0))  

        distance = metrics.get('distance', 0)
        if distance > 0:
            distance_bonus = min(0.3, distance / 30.0) 
            base_quality += distance_bonus

        # Penalidade por queda rápida
        steps = metrics.get('steps', 1)
        if steps < 10 and reward < -100:  
            base_quality *= 0.5

        return min(1.0, max(0.05, base_quality))


class SimpleBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.quality_threshold = 0.1  
        self.min_quality_for_phase = {}
        
    def add(self, experience):
        """Adiciona experiência se atender qualidade mínima - MAIS PERMISSIVO"""
        # Apenas rejeita se a qualidade for muito baixa
        if experience.quality >= 0.05:  
            self.buffer.append(experience)
            return True
        return False
    
    def sample(self, batch_size: int):
        """Amostragem APENAS das melhores experiências"""
        if len(self.buffer) < batch_size:
            return None

        # Amostrar apenas do top 30% do buffer
        elite_size = max(batch_size, int(len(self.buffer) * 0.5))
        elite_pool = sorted(self.buffer, key=lambda x: x.quality, reverse=True)[:elite_size]
        
        qualities = [exp.quality for exp in elite_pool]
        probs = np.array(qualities) / sum(qualities)

        indices = np.random.choice(len(elite_pool), size=batch_size, p=probs, replace=False)
        selected = [elite_pool[i] for i in indices]

        return self._convert_to_arrays(selected)
    
    def _convert_to_arrays(self, batch):
        """Converts batch of experiences to numpy arrays for training"""
        obs = np.array([exp.state for exp in batch])
        next_obs = np.array([exp.next_state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        dones = np.array([exp.done for exp in batch])

        return (obs, actions, rewards, next_obs, dones)
    
    def set_phase(self, phase):
        self.current_phase = phase
    
    def __len__(self):
        return len(self.buffer)

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
        kwargs.pop('action_dim', None)
        kwargs['learning_starts'] = 0
        
        super().__init__(policy, env, **kwargs)
        
        # Sistema DPG integrado
        observation_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]
        
        # Buffer elite com capacidade otimizada
        self.replay_buffer = SimpleBuffer(
            capacity=kwargs.get('buffer_size', 10000),
        )
        
        self.custom_logger = custom_logger
        self.phase_manager = PhaseManager()
        
        # Controle de treinamento DPG
        self.episode_count = 0
        self.training_interval = 5
        self.min_buffer_size = 200
        self.learning_progress = 0.0
        self.performance_history = deque(maxlen=50)
        
        # Histórico para adaptação
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0
        
        if custom_logger:
            custom_logger.info("FastTD3 com DPG integrado inicializado")

    def store_experience(self, state, action, reward, next_state, done, episode_results):
        """Armazena experiência no buffer DPG"""
        try:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metrics=episode_results
            )

            return self.replay_buffer.add(experience)

        except Exception as e:
            if self.custom_logger:
                self.custom_logger.error(f"Erro ao armazenar experiência: {e}")
            return False

    def update_phase_progression(self, episode_results):
        """Atualiza progresso baseado em desempenho recente"""
        self.episode_count += 1
        
        # Garantir que a distância não seja negativa
        distance = max(episode_results.get("distance", 0), 0)
        
        # Cálculo de estabilidade
        roll = abs(episode_results.get("roll", 0))
        pitch = abs(episode_results.get("pitch", 0))
        stability = max(0.0, 1.0 - min((roll + pitch) / 1.0, 1.0))
        
        # Progresso simples
        distance_progress = min(distance / 9.0, 1.0) 
        stability_progress = stability
        
        self.learning_progress = max(0.0, (distance_progress * 0.5 + stability_progress * 0.5))
        
        # Atualizar phase manager
        self.phase_manager.update_phase_metrics(episode_results)
        
        # Verificar transição de fase
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                new_phase = self.phase_manager.current_phase
                if self.custom_logger:
                    self.custom_logger.info(f"🎉 Transição para Fase {new_phase}!")

    def _store_transition(self, replay_buffer, action, new_obs, reward, done, infos):
        """Armazena transição - compatibilidade com SB3"""
        obs = self._last_obs
        if hasattr(obs, 'flatten'):
            obs = obs.flatten()
        if hasattr(new_obs, 'flatten'):
            new_obs = new_obs.flatten()
        if hasattr(action, 'flatten'):
            action = action.flatten()
            
        # Para FastTD3, usamos nosso próprio buffer
        episode_results = infos[0] if infos else {}
        success = self.store_experience(obs, action, reward, new_obs, done, episode_results)
    
        return success

    def _setup_model(self):
        """Configuração inicial do modelo"""
        super()._setup_model()
        self.replay_buffer = SimpleBuffer(capacity=self.buffer_size)
    
    def _polyak_update(self, params, target_params, tau):
        """Implementação manual do polyak update"""
        with torch.no_grad():
            for param, target_param in zip(params, target_params):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(param.data * tau)

    def train(self, gradient_steps, batch_size=256):
        """Treinamento usando nosso buffer personalizado"""
        current_phase = self.phase_manager.current_phase
        self.replay_buffer.set_phase(current_phase)
    
        successful_steps = 0
    
        for gradient_step in range(gradient_steps):
            # Amostrar do nosso buffer personalizado
            batch_data = self.replay_buffer.sample(batch_size)
    
            if batch_data is None:
                continue
            
            obs, actions, rewards, next_obs, dones = batch_data
            actual_batch_size = len(obs)
    
            with torch.no_grad():
                # Converter para tensores
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                actions_tensor = torch.FloatTensor(actions).to(self.device)
                next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)  # [batch_size, 1]
                dones_tensor = torch.FloatTensor(dones).to(self.device).unsqueeze(1)      # [batch_size, 1]
    
                # Noise para regularização
                noise = (torch.randn_like(actions_tensor) * self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
    
                # Ações alvo com noise
                next_actions = (self.actor_target(next_obs_tensor) + noise).clamp(-1, 1)
    
                # Q-values alvo (twin critics)
                target_q1, target_q2 = self.critic_target(next_obs_tensor, next_actions)

                def ensure_q_shape(tensor, batch_size):
                    # Flatten all but first dimension, then squeeze to [batch_size]
                    tensor = torch.flatten(tensor, start_dim=1)  # [B, *] → [B, N]
                    if tensor.shape[1] > 1:
                        # Se critic retornar múltiplas saídas (ex: ensemble ou bug), pegue a primeira coluna
                        tensor = tensor[:, 0:1]  # [B, 1]
                    else:
                        tensor = tensor.view(batch_size, 1)  # garantir [B, 1]
                    return tensor

                target_q1 = ensure_q_shape(target_q1, batch_size)
                target_q2 = ensure_q_shape(target_q2, batch_size)
                target_q_value = torch.min(target_q1, target_q2)  # [B, 1]
    
            # Q-values atuais
            current_q1, current_q2 = self.critic(obs_tensor, actions_tensor)
            
            # CORREÇÃO ROBUSTA: mesma lógica para current_q
            current_q1 = current_q1.reshape(current_q1.shape[0], -1)[:, :1]
            current_q2 = current_q2.reshape(current_q2.shape[0], -1)[:, :1]
    
            # Loss do critic - AGORA COM SHAPES COMPATÍVEIS
            critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)
    
            # Otimizar critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
    
            # Atualizar actor (com policy delay)
            if gradient_step % self.policy_delay == 0:
                actor_q_values = self.critic.q1_forward(obs_tensor, self.actor(obs_tensor))
                
                # CORREÇÃO: garantir shape [batch_size, 1]
                actor_q_values = actor_q_values.reshape(actor_q_values.shape[0], -1)[:, :1]
                actor_loss = -actor_q_values.mean()
    
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
    
                # Atualizar redes alvo usando nossa implementação polyak
                self._polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                self._polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
    
            successful_steps += 1
    
        return successful_steps

    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas de fase - interface para simulação"""
        self.update_phase_progression(episode_metrics)
        
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                new_phase = self.phase_manager.current_phase
                if self.custom_logger:
                    self.custom_logger.info(f"🎉 Transição para Fase {new_phase}!")
                return True
        return False
    
    def get_phase_info(self):
        return self.phase_manager.get_phase_info()
    
    def get_phase_weight_adjustments(self):
        return self.phase_manager.get_phase_weight_adjustments()

    def get_dpg_status(self):
        """Retorna status completo do sistema DPG integrado"""
        buffer_list = list(self.replay_buffer.buffer)
        qualities = [exp.quality for exp in buffer_list] if buffer_list else []
        
        high_quality = sum(1 for q in qualities if q > 0.7) if qualities else 0
        medium_quality = sum(1 for q in qualities if 0.3 <= q <= 0.7) if qualities else 0
        low_quality = sum(1 for q in qualities if q < 0.3) if qualities else 0
        
        return {
            "enabled": True,
            "episode_count": self.episode_count,
            "buffer_size": len(self.replay_buffer),
            "learning_progress": self.learning_progress,
            "phase": self.phase_manager.current_phase,
            "buffer_quality": {
                "total": len(buffer_list),
                "avg_quality": np.mean(qualities) if qualities else 0,
                "high": high_quality,
                "medium": medium_quality,
                "low": low_quality
            },
            "phase_info": self.get_phase_info()
        }

    def _cleanup_buffer(self):
        """Remove experiências de baixa qualidade periodicamente"""
        if len(self.replay_buffer.buffer) > 2000:
            # Mantém apenas as melhores 2000 experiências
            sorted_experiences = sorted(self.replay_buffer.buffer, key=lambda x: x.quality, reverse=True)
            self.replay_buffer.buffer = deque(sorted_experiences[:2000], maxlen=self.replay_buffer.capacity)