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
        
        # Estado e progressão
        self.episode_count = 0
        self.performance_history = deque(maxlen=50)
        self.learning_progress = 0.0
        self.current_terrain = "normal"
        self.current_phase = 1  

        # Controle de treinamento
        self.training_interval = 5
        self.min_buffer_size = 200
        self.learning_rate_factor = 1.0

        # Histórico
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0
        self.episode_metrics_history = deque(maxlen=50)
        self.terrain_performance = {}

         # Variáveis para resumo
        self._last_training_episode = 0
        self.last_report_episode = 0

        # Fatores de ajuste por terreno (multiplicadores dos pesos base)
        self.terrain_factors = self._initialize_terrain_factors()
        
        # NOVO: Armazenar pesos base originais para evitar overflow
        self.base_weights = self._capture_base_weights()

    def _capture_base_weights(self):
        """Captura os pesos base originais do reward system"""
        base_weights = {}
        for name, component in self.reward_system.components.items():
            base_weights[name] = component.weight
        return base_weights

    def _initialize_terrain_factors(self):
        """Fatores multiplicativos para cada tipo de terreno"""
        return {
            "PR": self._get_pr_factors(),      # Pista normal
            "PBA": self._get_pba_factors(),    # Gelo - baixo atrito
            "PG": self._get_pg_factors(),      # Areia - piso granulado
            "PRB": self._get_prb_factors(),    # Bloqueios articulares
            "PRD": self._get_prd_factors(),    # Rampa descendente
            "PRA": self._get_pra_factors(),    # Rampa ascendente
        }

    def _get_pr_factors(self):
        """Pista normal - fatores neutros"""
        return {
            "progress": 1.0,
            "stability_pitch": 1.0,
            "stability_roll": 1.0,
            "alternating_foot_contact": 1.0,
            "foot_clearance": 1.0,
            "gait_rhythm": 1.0,
            "gait_state_change": 1.0,
            "fall_penalty": 1.0,
            "yaw_penalty": 1.0,
            "foot_back_penalty": 1.0,
            "effort_square_penalty": 1.0,
        }

    def _get_pba_factors(self):
        """Gelo - FOCO MÁXIMO EM ESTABILIDADE"""
        return {
            "progress": 0.6,           # Reduz progresso (aumentado de 0.4)
            "stability_pitch": 1.5,    # Reduzido de 2.0 para evitar overflow
            "stability_roll": 1.8,     # Reduzido de 3.0 para evitar overflow
            "alternating_foot_contact": 1.5,  # Reduzido de 3.0
            "foot_clearance": 0.8,     # Aumentado de 0.5
            "gait_rhythm": 0.8,        # Aumentado de 0.5
            "gait_state_change": 1.2,  # Reduzido de 1.4
            "fall_penalty": 1.2,       # Reduzido de 1.4
            "yaw_penalty": 1.1,        # Reduzido de 1.2
            "foot_back_penalty": 1.5,  # Reduzido de 2.0
            "effort_square_penalty": 1.5,  # Reduzido de 2.0
        }

    def _get_pg_factors(self):
        """Areia - FOCO EM IMPULSO E CLEARANCE"""
        return {
            "progress": 1.0,           # Aumentado de 0.8
            "stability_pitch": 1.0,    # Aumentado de 0.8
            "stability_roll": 1.0,     # Estabilidade normal
            "alternating_foot_contact": 1.0,  # Normal
            "foot_clearance": 2.0,     # REDUZIDO de 3.0 para evitar overflow
            "gait_rhythm": 1.2,        # Reduzido de 1.5
            "gait_state_change": 1.0,  # Aumentado de 0.8
            "fall_penalty": 1.0,       # Aumentado de 0.9
            "yaw_penalty": 1.0,        # Aumentado de 0.8
            "foot_back_penalty": 1.0,  # Aumentado de 0.8
            "effort_square_penalty": 0.8,  # Aumentado de 0.5
        }

    def _get_prb_factors(self):
        """Bloqueios articulares - FOCO EM ROBUSTEZ"""
        return {
            "progress": 0.8,           # Aumentado de 0.6
            "stability_pitch": 1.1,    # Reduzido de 1.25
            "stability_roll": 1.3,     # Reduzido de 2.0
            "alternating_foot_contact": 1.2,  # Reduzido de 2.0
            "foot_clearance": 1.0,     # Normal
            "gait_rhythm": 1.1,        # Reduzido de 1.25
            "gait_state_change": 1.2,  # Reduzido de 1.6
            "fall_penalty": 1.05,      # Reduzido de 1.15
            "yaw_penalty": 1.0,        # Normal
            "foot_back_penalty": 1.1,  # Reduzido de 1.2
            "effort_square_penalty": 1.0,  # Normal
        }

    def _get_prd_factors(self):
        """Rampa descendente - FOCO EM CONTROLE"""
        return {
            "progress": 0.7,           # Aumentado de 0.5
            "stability_pitch": 1.3,    # Reduzido de 2.5
            "stability_roll": 1.5,     # Reduzido de 3.0
            "alternating_foot_contact": 1.3,  # Reduzido de 2.0
            "foot_clearance": 1.0,     # Aumentado de 0.8
            "gait_rhythm": 1.0,        # Aumentado de 0.75
            "gait_state_change": 1.1,  # Reduzido de 1.2
            "fall_penalty": 1.3,       # Reduzido de 1.7
            "yaw_penalty": 1.2,        # Reduzido de 1.4
            "foot_back_penalty": 1.3,  # Reduzido de 2.5
            "effort_square_penalty": 1.2,  # Reduzido de 1.5
        }

    def _get_pra_factors(self):
        """Rampa ascendente - FOCO EM PROPULSÃO"""
        return {
            "progress": 1.1,           # Reduzido de 1.2
            "stability_pitch": 1.2,    # Reduzido de 1.5
            "stability_roll": 1.3,     # Reduzido de 2.0
            "alternating_foot_contact": 1.0,  # Normal
            "foot_clearance": 1.2,     # Reduzido de 1.5
            "gait_rhythm": 1.3,        # Reduzido de 1.75
            "gait_state_change": 1.0,  # Normal
            "fall_penalty": 1.05,      # Reduzido de 1.15
            "yaw_penalty": 1.0,        # Normal
            "foot_back_penalty": 1.2,  # Reduzido de 1.5
            "effort_square_penalty": 1.0,  # Aumentado de 0.8
        }

    def _apply_terrain_weights(self, sim):
        """Aplica fatores multiplicativos aos pesos base - COM PROTEÇÃO CONTRA OVERFLOW"""
        if not self.enabled:
            return

        # Obter fatores do terreno atual
        terrain_key = self.current_terrain
        if terrain_key not in self.terrain_factors:
            terrain_key = "PR"  # Fallback para pista normal
        
        terrain_factors = self.terrain_factors[terrain_key]
        
        # Ajustar fatores baseado na fase atual
        phase_adjustment = self._get_phase_adjustment()
        
        # NOVO: Limitar learning_rate_factor para evitar crescimento exponencial
        self.learning_rate_factor = np.clip(self.learning_rate_factor, 0.1, 3.0)
        
        # Aplicar fatores a cada componente habilitado
        for component_name, component in self.reward_system.components.items():
            if component.enabled and component_name in terrain_factors:
                # Obter fator do terreno para este componente
                terrain_factor = terrain_factors[component_name]
                
                # NOVO: Usar peso base capturado inicialmente em vez do atual
                base_weight = self.base_weights.get(component_name, component.weight)
                
                # Calcular peso ajustado com limites
                adjusted_weight = base_weight * terrain_factor * phase_adjustment * self.learning_rate_factor
                
                # NOVO: Limitar pesos para evitar overflow
                max_weight = 100.0  # Limite máximo absoluto
                adjusted_weight = np.clip(adjusted_weight, -max_weight, max_weight)
                
                # Aplicar o peso ajustado
                component.weight = adjusted_weight

    def _get_base_weight(self, component_name):
        """Retorna o peso base do componente do default.json"""
        # NOVO: Usar base_weights em vez do peso atual
        return self.base_weights.get(component_name, 0.0)

    def _get_phase_adjustment(self):
        """Retorna fator de ajuste baseado na fase atual"""
        if self.current_phase == 1:  # Estabilidade
            return 0.9  # Aumentado de 0.8
        elif self.current_phase == 2:  # Progresso
            return 1.1  # Reduzido de 1.2
        else:  # Otimização (fase 3)
            return 1.3  # Reduzido de 1.5

    def _update_phase_progression(self, episode_results):
        """Atualiza fase baseada no desempenho"""
        distance = max(episode_results.get("distance", 0), 0)
        stability = 1.0 - min((abs(episode_results.get("roll", 0)) + 
                              abs(episode_results.get("pitch", 0))) / 1.0, 1.0)
        
        # Critérios de progressão de fase
        if self.current_phase == 1 and distance > 3.0 and stability > 0.7:
            self.current_phase = 2
            
        elif self.current_phase == 2 and distance > 6.0 and stability > 0.8:
            self.current_phase = 3  

    def calculate_reward(self, sim, action) -> float:
        if not self.enabled:
            return self.reward_system.calculate_reward(sim, action)
            
        # Aplica fatores específicos do terreno e fase
        self._apply_terrain_weights(sim)
        
        # Usa o cálculo padrão do RewardSystem com pesos ajustados
        reward = self.reward_system.calculate_reward(sim, action)
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
        """Progressão inteligente por fase e terreno - COM PROTEÇÃO CONTRA OVERFLOW"""
        self.episode_count += 1
        
        distance = max(episode_results.get("distance", 0), 0)
        stability = 1.0 - min((abs(episode_results.get("roll", 0)) + 
                              abs(episode_results.get("pitch", 0))) / 1.0, 1.0)
        
        # Atualizar progressão de fase
        self._update_phase_progression(episode_results)
        
        # Aprendizado adaptativo por terreno
        terrain_factor = self._get_terrain_learning_factor()
        performance = (min(distance / 9.0, 1.0) * 0.6 + stability * 0.4)
        
        # NOVO: Ajuste mais conservador da taxa de aprendizado
        if performance > 0.6:  
            self.learning_rate_factor = min(2.0, self.learning_rate_factor + 0.02 * terrain_factor)  # Reduzido de 0.05
        else: 
            self.learning_rate_factor = max(0.5, self.learning_rate_factor - 0.01 * terrain_factor)  # Aumentado de 0.3
         
        if self.episode_count % 500 == 0:
            self._generate_comprehensive_report()
            
        # Limpeza periódica do buffer
        if self.episode_count % 100 == 0 and len(self.buffer) > 2000:
            self._cleanup_buffer()

    def _get_terrain_learning_factor(self):
        """Fator de dificuldade por terreno"""
        terrain_difficulty = {
            "PR": 1.0,   # Normal
            "PBA": 0.8,  # Aumentado de 0.6
            "PG": 0.9,   # Aumentado de 0.8
            "PRB": 0.85, # Aumentado de 0.7
            "PRD": 0.7,  # Aumentado de 0.5
            "PRA": 0.95, # Aumentado de 0.9
        }
        return terrain_difficulty.get(self.current_terrain, 1.0)
    
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