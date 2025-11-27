# dpg_manager.py
import numpy as np
from collections import deque
import random
from typing import List, Dict, Any
import time
from dataclasses import dataclass
import copy

class Experience:
    def __init__(self, state, action, reward, next_state, done, metrics):
        self.state = np.array(state, dtype=np.float32)
        self.action = np.array(action, dtype=np.float32)
        self.reward = float(reward)
        self.next_state = np.array(next_state, dtype=np.float32)
        self.done = done
        self.quality = self._calculate_quality(metrics)
        
    def _calculate_quality(self, metrics):
        """Qualidade simples baseada em distância e estabilidade"""
        distance = max(metrics.get("distance", 0), 0)
        stability = 1.0 - min((abs(metrics.get("roll", 0)) + abs(metrics.get("pitch", 0))) / 1.0, 1.0)
        
        if distance <= 0:
            return 0.0
            
        return min(distance / 2.0, 0.6) + stability * 0.4
        

class DPGManager:
    def __init__(self, logger, robot, reward_system):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.enabled = True
        self.learning_enabled = True
        
        # Sistemas principais
        self.buffer = SimpleBuffer(capacity=3000)
        
        # Estado simples
        self.episode_count = 0
        self.performance_history = deque(maxlen=50)
        self.learning_progress = 0.0
        self.current_terrain = "normal"  

        # Controle de treinamento
        self.training_interval = 5  
        self.min_buffer_size = 200  
        self.learning_rate_factor = 1.0

        # Histórico para adaptação
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0

        # Variáveis para resumo
        self._last_training_episode = 0
        self.last_report_episode = 0
        self.episode_metrics_history = deque(maxlen=50)
        self.terrain_performance = {}

        # Configuração de pesos por terreno (mantendo a lógica de valências)
        self.terrain_weights = self._initialize_terrain_weights()

    def _initialize_terrain_weights(self):
        """Inicializa os pesos específicos para cada terreno"""
        return {
            "normal": self._get_default_weights(),
            "ramp_up": self._get_ramp_up_weights(),
            "ramp_down": self._get_ramp_down_weights(),
            "uneven": self._get_uneven_weights(),
            "PG": self._get_pg_weights(),
        }

    def _get_default_weights(self):
        """Pesos padrão para terreno normal"""
        return {
            "progress": 2.0,
            "stability_pitch": 1.2,
            "stability_roll": 1.2,
            "alternating_foot_contact": 1.8,
            "foot_clearance": 1.5,
            "gait_rhythm": 1.0,
            "distance_bonus": 1.0
        }

    def _get_ramp_up_weights(self):
        """Pesos para rampa ascendente - foco em estabilidade e coordenação"""
        weights = self._get_default_weights()
        weights.update({
            "stability_pitch": 1.6,
            "stability_roll": 1.4,
            "alternating_foot_contact": 2.0,
            "foot_clearance": 1.8
        })
        return weights

    def _get_ramp_down_weights(self):
        """Pesos para rampa descendente - foco em controle e estabilidade"""
        weights = self._get_default_weights()
        weights.update({
            "stability_pitch": 1.8,
            "stability_roll": 1.6,
            "foot_clearance": 1.6
        })
        return weights

    def _get_uneven_weights(self):
        """Pesos para terreno irregular - foco em adaptação"""
        weights = self._get_default_weights()
        weights.update({
            "foot_clearance": 2.0,
            "alternating_foot_contact": 1.6,
            "gait_rhythm": 1.2
        })
        return weights

    def _get_pg_weights(self):
        """Pesos para PG - balance entre todos os componentes"""
        weights = self._get_default_weights()
        weights.update({
            "progress": 2.2,
            "stability_pitch": 1.3,
            "stability_roll": 1.3
        })
        return weights

    def _apply_terrain_weights(self, sim):
        """Aplica os pesos específicos do terreno atual ao RewardSystem"""
        if not self.enabled:
            return

        current_weights = self.terrain_weights.get(self.current_terrain, self._get_default_weights())
        
        for component_name, weight in current_weights.items():
            if component_name in self.reward_system.components:
                # Ajusta o peso baseado no progresso de aprendizado
                adjusted_weight = weight * self.learning_rate_factor
                self.reward_system.components[component_name].weight = adjusted_weight

    def calculate_reward(self, sim, action) -> float:
        if not self.enabled:
            return self.reward_system.calculate_standard_reward(sim, action)
            
        # Aplica pesos específicos do terreno
        self._apply_terrain_weights(sim)
        
        # Usa o cálculo padrão do RewardSystem com os pesos ajustados
        reward = self.reward_system.calculate_standard_reward(sim, action)
        self.performance_history.append(reward)
        
        return reward

    def set_current_terrain(self, terrain):  
        """Define o terreno atual para cálculos de recompensa"""
        self.current_terrain = terrain

    def store_experience(self, state, action, reward, next_state, done, episode_results):
        if not self.enabled:
            return False

        try:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metrics=episode_results
            )
            
            success = self.buffer.add(experience)
            return success
            
        except Exception as e:
            self.logger.error(f"Erro ao armazenar experiência: {e}")
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
        
        # Ajusta learning_rate_factor baseado no progresso
        self.learning_rate_factor = 0.5 + (self.learning_progress * 0.5)
        
        # Atualizar histórico de métricas
        self.episode_metrics_history.append({
            "reward": episode_results.get("reward", 0),
            "distance": distance,
            "success": distance > 0.5 and stability > 0.6
        })
        
        # Atualizar desempenho por terreno
        terrain = self.current_terrain
        if terrain not in self.terrain_performance:
            self.terrain_performance[terrain] = {
                'episodes': 0,
                'total_distance': 0,
                'total_reward': 0,
                'successes': 0
            }
        
        self.terrain_performance[terrain]['episodes'] += 1
        self.terrain_performance[terrain]['total_distance'] += distance
        self.terrain_performance[terrain]['total_reward'] += episode_results.get("reward", 0)
        if distance > 0.5 and stability > 0.6:
            self.terrain_performance[terrain]['successes'] += 1
         
        if self.episode_count % 500 == 0:
            self._generate_comprehensive_report()
            self.last_report_episode = self.episode_count
            
        # Limpeza periódica do buffer
        if self.episode_count % 100 == 0 and len(self.buffer) > 2000:
            self._cleanup_buffer()

    def _cleanup_buffer(self):
        """Remove experiências de baixa qualidade periodicamente"""
        if len(self.buffer.buffer) > 2000:
            # Mantém apenas as melhores 2000 experiências
            sorted_experiences = sorted(self.buffer.buffer, key=lambda x: x.quality, reverse=True)
            self.buffer.buffer = deque(sorted_experiences[:2000], maxlen=self.buffer.capacity)

    def should_train(self, current_episode: int) -> bool:
        """Decide se deve treinar neste episódio"""
        if not self.learning_enabled:
            return False
            
        buffer_ready = len(self.buffer) >= self.min_buffer_size
        interval_ok = (current_episode - self._last_training_episode) >= self.training_interval
        
        return buffer_ready and interval_ok

    def on_training_completed(self, episode: int):
        """Callback quando o treinamento é completado"""
        self._last_training_episode = episode

    def get_training_batch(self, batch_size=32) -> List[Experience]:
        return self.buffer.sample(batch_size)

    def get_integrated_status(self):
        return {
            "enabled": self.enabled,
            "episode_count": self.episode_count,
            "buffer_size": len(self.buffer),
            "learning_progress": self.learning_progress,
            "learning_rate_factor": self.learning_rate_factor,
            "current_terrain": self.current_terrain,
            "avg_recent_reward": np.mean(list(self.performance_history)) if self.performance_history else 0
        }
    
    def _get_buffer_status(self):
        """Status simplificado do buffer"""
        if len(self.buffer) == 0:
            return {
                "total_experiences": 0,
                "avg_quality": 0,
                "buffer_utilization": 0,
                "active_experiences": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0}
            }
        
        buffer_list = list(self.buffer.buffer) if hasattr(self.buffer, 'buffer') else list(self.buffer)
        qualities = [exp.quality for exp in buffer_list]
        
        if not qualities:  
            return {
                "total_experiences": 0,
                "avg_quality": 0,
                "buffer_utilization": 0,
                "active_experiences": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0}
            }
        
        high_quality = sum(1 for q in qualities if q > 0.7)
        medium_quality = sum(1 for q in qualities if 0.3 <= q <= 0.7)
        low_quality = sum(1 for q in qualities if q < 0.3)
        
        return {
            "total_experiences": len(buffer_list),
            "avg_quality": np.mean(qualities),
            "buffer_utilization": len(buffer_list) / self.buffer.capacity,
            "active_experiences": len(buffer_list),
            "quality_distribution": {
                "high": high_quality,
                "medium": medium_quality,
                "low": low_quality
            }
        }

    def _generate_comprehensive_report(self):
        """RELATÓRIO a cada 500 episódios"""
        try:
            # Dados básicos do sistema
            integrated_status = self.get_integrated_status()
            buffer_status = self._get_buffer_status()
            
            # Estatísticas recentes (últimos 50 episódios)
            recent_episodes = list(self.episode_metrics_history)
            if recent_episodes:
                recent_rewards = [ep['reward'] for ep in recent_episodes]
                recent_distances = [ep['distance'] for ep in recent_episodes]
                recent_successes = sum(1 for ep in recent_episodes if ep['success'])
                
                avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_recent_distance = np.mean(recent_distances) if recent_distances else 0
                success_rate = recent_successes / len(recent_episodes) if recent_episodes else 0
            else:
                avg_recent_reward = 0
                avg_recent_distance = 0
                success_rate = 0

            # RELATÓRIO 
            self.logger.info("=" * 60)
            self.logger.info(f"📊 RELATÓRIO DPG - Episódio {self.episode_count}")
            
            # Status principal
            self.logger.info(f"Progresso: {self.learning_progress:.1%} | "
                           f"Recompensa: {avg_recent_reward:.1f} | "
                           f"Sucesso: {success_rate:.1%}")
            self.logger.info(f"Fator Aprendizado: {self.learning_rate_factor:.2f} | "
                           f"Terreno: {self.current_terrain}")
            
            # Buffer
            self.logger.info(f"Buffer: {len(self.buffer)} exp | "
                           f"Qualidade: {buffer_status['avg_quality']:.3f}")
            
            # Terrenos
            self.logger.info("Desempenho por Terreno:")
            if hasattr(self, 'terrain_performance') and self.terrain_performance:
                for terrain, stats in self.terrain_performance.items():
                    if stats['episodes'] > 0:
                        avg_dist = stats['total_distance'] / stats['episodes']
                        success_rate_terrain = stats['successes'] / stats['episodes']
                        self.logger.info(f"  {terrain}: {avg_dist:.2f}m, {success_rate_terrain:.1%} sucesso")
            else:
                self.logger.info("  Nenhum dado de terreno disponível")
            
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Erro no relatório DPG: {e}")


class SimpleBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.quality_threshold = 0.3
        
    def add(self, experience):
        """Adiciona experiência se atender qualidade mínima"""
        if experience.quality >= self.quality_threshold:
            self.buffer.append(experience)
            return True
        return False
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Amostragem balanceada com foco em experiências educativas"""
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
            
        # Estratificação por qualidade
        high_quality = [exp for exp in self.buffer if exp.quality > 0.7]
        medium_quality = [exp for exp in self.buffer if 0.3 <= exp.quality <= 0.7]
        low_quality = [exp for exp in self.buffer if exp.quality < 0.3]
        
        # Proporções otimizadas
        high_count = min(int(batch_size * 0.4), len(high_quality)) 
        medium_count = min(int(batch_size * 0.4), len(medium_quality))
        low_count = batch_size - high_count - medium_count
        
        samples = []
        if high_count > 0:
            samples.extend(random.sample(high_quality, high_count))
        if medium_count > 0:
            samples.extend(random.sample(medium_quality, medium_count))
        if low_count > 0 and len(low_quality) > 0:
            samples.extend(random.sample(low_quality, min(low_count, len(low_quality))))
            
        # Preencher com amostras aleatórias se necessário
        if len(samples) < batch_size:
            additional = random.sample(self.buffer, batch_size - len(samples))
            samples.extend(additional)
            
        return samples
    
    def __len__(self):
        return len(self.buffer)