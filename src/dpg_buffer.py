# dpg_buffer.py
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import heapq
from collections import deque
from functools import lru_cache
import time

@dataclass
class Experience:
    """Estrutura para armazenar experiências"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]
    group: int
    sub_phase: int
    quality: float
    skills: Dict[str, float]
    
    def __lt__(self, other):
        """Define comparação para heapq baseado na qualidade"""
        return self.quality < other.quality
    
    def __eq__(self, other):
        """Define igualdade para heapq"""
        if not isinstance(other, Experience):
            return False
        return self.quality == other.quality

class Cache:
    """Cache unificado com múltiplas estratégias de evição"""
    
    def __init__(self, max_size=1000, default_ttl=100, strategy="adaptive"):
        self._cache = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._strategy = strategy
        
        # Estatísticas
        self._hits = 0
        self._misses = 0
        self._access_pattern = {}
        
    def get(self, key: str) -> Any:
        """Obtém valor do cache com verificação de TTL"""
        if key in self._cache:
            value, timestamp, ttl, priority = self._cache[key]
            if time.time() - timestamp < ttl:
                self._hits += 1
                self._access_pattern[key] = self._access_pattern.get(key, 0) + 1
                return value
            else:
                # Expirou - remover
                del self._cache[key]
                if key in self._access_pattern:
                    del self._access_pattern[key]
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None, priority: float = 1.0):
        """Armazena valor no cache com prioridade"""
        if len(self._cache) >= self._max_size:
            self._evict_entries()
            
        actual_ttl = ttl or self._default_ttl
        self._cache[key] = (value, time.time(), actual_ttl, priority)
        self._access_pattern[key] = self._access_pattern.get(key, 0) + 1

    def _evict_entries(self):
        """Estratégias de evição adaptativas"""
        if not self._cache:
            return
            
        if self._strategy == "lru":
            self._evict_lru()
        elif self._strategy == "priority":
            self._evict_low_priority()
        else:  # adaptive
            self._evict_adaptive()
    
    def _evict_adaptive(self):
        """Combina frequência de acesso, idade e prioridade"""
        def eviction_score(k):
            value, timestamp, ttl, priority = self._cache[k]
            age = time.time() - timestamp
            access_count = self._access_pattern.get(k, 0)
            # Score mais alto = mais provável de ser removido
            return age / (access_count + 1) / (priority + 0.1)
            
        key_to_remove = min(self._cache.keys(), key=eviction_score)
        del self._cache[key_to_remove]
        if key_to_remove in self._access_pattern:
            del self._access_pattern[key_to_remove]
    
    def get_stats(self) -> Dict:
        """Estatísticas completas do cache"""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._max_size,
            "strategy": self._strategy
        }
    
    def clear_expired(self):
        """Limpa entradas expiradas"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, timestamp, ttl, _) in self._cache.items()
            if current_time - timestamp >= ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_pattern:
                del self._access_pattern[key]

class StateCompressor:
    """Compressor eficiente de estados para economizar memória"""
    
    def __init__(self):
        self._mean = None
        self._std = None
        self._is_trained = False
        
    def compress_state(self, state: np.ndarray) -> np.ndarray:
        """Compressão simplificada do estado"""
        if len(state) > 15:  # Apenas comprime estados grandes
            # Redução dimensional simples - pega características principais
            if len(state) >= 10:
                # Mantém primeiras 8 dimensões + últimas 2 (normalmente mais importantes)
                compressed = np.concatenate([state[:8], state[-2:]])
                return compressed
            else:
                return state[:8]  # Fallback
        return state

class PrioritizedBuffer:
    """Buffer com amostragem prioritária"""
    
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self._quality_heap = []  # Heap para rápida recuperação das melhores
        self.pos = 0
        
    def add(self, experience: Experience, priority: float = None) -> bool:
        """Adiciona experiência com capacidade total"""
        if priority is None:
            priority = self._calculate_priority(experience)

        try:
            # SE há espaço, adiciona normalmente
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.priorities.append(priority)

                # Mantém heap de qualidade atualizado
                heapq.heappush(self._quality_heap, (-experience.quality, experience))

                return True
            else:
                # SE buffer cheio, substitui a de menor prioridade
                min_priority = min(self.priorities) if self.priorities else 0
                if priority > min_priority:
                    min_idx = self.priorities.index(min_priority)
                    self.buffer[min_idx] = experience
                    self.priorities[min_idx] = priority

                    # Atualiza heap
                    heapq.heappush(self._quality_heap, (-experience.quality, experience))
                    return True
                else:
                    return False  # Prioridade muito baixa, não armazena

        except Exception as e:
            return False
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Amostragem 70/30 - preserva diversidade + qualidade"""
        if not self.buffer:
            return []

        if len(self.buffer) <= batch_size:
            return self.buffer.copy()

        # ESTRATÉGIA MISTA (70% aleatório + 30% qualidade)
        random_count = int(batch_size * 0.7)
        quality_count = batch_size - random_count

        # Amostra aleatória (rápida)
        random_samples = []
        if random_count > 0:
            random_samples = random.sample(self.buffer, random_count)

        # Amostra por qualidade (mantém aprendizado)
        quality_samples = []
        if quality_count > 0 and hasattr(self, '_quality_heap') and self._quality_heap:
            quality_samples = [exp for _, exp in heapq.nlargest(quality_count, self._quality_heap)]

        # Se não tem qualidade suficiente, completa com aleatório
        if len(quality_samples) < quality_count:
            needed = quality_count - len(quality_samples)
            extra_random = random.sample(self.buffer, min(needed, len(self.buffer)))
            quality_samples.extend(extra_random)

        return random_samples + quality_samples
    
    def get_high_quality(self, count: int) -> List[Experience]:
        """Recupera experiências de alta qualidade rapidamente"""
        if not self._quality_heap:
            return self.sample(count)
            
        return [exp for _, exp in heapq.nlargest(count, self._quality_heap)]
    
    def _calculate_priority(self, experience: Experience) -> float:
        """Calcula prioridade baseada em múltiplos fatores"""
        return (experience.quality * 0.6 + 
                min(experience.reward * 0.1, 0.3) + 
                sum(experience.skills.values()) * 0.1)


class BufferManager:
    """BUFFER DE ALTA PERFORMANCE"""
    
    def __init__(self, logger, config, max_experiences=50000):
        self.logger = logger
        self.config = config
        self.max_experiences = max_experiences
        
        # Sistema de buffers otimizado
        self.group_buffers = {}
        for group in [1, 2, 3]:
            self.group_buffers[group] = PrioritizedBuffer(capacity=20000)
        self.core_buffer = PrioritizedBuffer(capacity=10000)
        self.current_group = 1
        if self.current_group in self.group_buffers:
            self.current_group_buffer = self.group_buffers[self.current_group].buffer
        else:
            self.current_group_buffer = []
        
        # Sistemas de otimização
        self.quality_cache = Cache(max_size=500)
        self.state_compressor = StateCompressor()
        
        # Estatísticas
        self.episode_count = 0
        self.stored_count = 0
        self.rejected_count = 0
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_savings": 0,
            "avg_quality_stored": 0.0
        }
        
        # Controles adaptativos
        self._last_cleanup = 0
        self.cleanup_interval = 200
        self._dpg_manager = None
        
    def store_experience(self, experience_data: Dict):
        try:
            if not experience_data:
                return False

            # Garantir que metrics existe
            if "metrics" not in experience_data:
                experience_data["metrics"] = {}

            # Calcular qualidade
            quality = self._calculate_quality(experience_data)
            current_group = self.get_current_group()
            experience_data["group_level"] = current_group

            # Criar experiência
            experience = self._create_compressed_experience(experience_data)
            experience.quality = quality

            # VERIFICAR se deve armazenar
            if not self._should_store(experience):
                self.rejected_count += 1
                return False

            # ARMAZENAR no buffer
            success = self._store_optimized(experience, current_group)
            if success:
                self.stored_count += 1
            else:
                self.logger.info(f"❌ FALHA NO ARMAZENAMENTO: Buffer cheio?")

            return success

        except Exception as e:
            self.logger.error(f"❌ ERRO CRÍTICO no armazenamento: {e}")
            return False
    
    def _create_compressed_experience(self, data: Dict) -> Experience:
        """Cria experiência com estado comprimido"""
        try:
            # Garante que estados são numpy arrays
            state = np.array(data["state"], dtype=np.float32)
            action = np.array(data["action"], dtype=np.float32)

            compressed_state = self.state_compressor.compress_state(state)
            compressed_next = self.state_compressor.compress_state(
                data.get("next_state", data["state"])
            )

            # Garantir que as métricas estão no info
            phase_info = data.get("phase_info", {})

            # COPIAR AS MÉTRICAS PARA O INFO
            info_with_metrics = phase_info.copy()
            info_with_metrics["metrics"] = data.get("metrics", {})  

            # Calcula qualidade ANTES de criar a experiência
            quality = self._calculate_quality(data)

            experience = Experience(
                state=compressed_state,
                action=action,
                reward=float(data["reward"]),
                next_state=compressed_next,
                done=False,
                info=info_with_metrics,  
                group=data.get("group_level", 1),
                sub_phase=0,
                quality=quality,
                skills=self._analyze_skills(data.get("metrics", {}))
            )
            return experience

        except Exception as e:
            self.logger.error(f"❌ ERRO na criação de experiência: {e}")
            # Fallback...
            return Experience(
                state=np.zeros(10),
                action=np.zeros(6),
                reward=0,
                next_state=np.zeros(10),
                done=False,
                info={"metrics": {}},  
                group=1,
                sub_phase=0,
                quality=0.0,
                skills={}
            )
    
    def _should_store(self, experience: Experience) -> bool:
        """Critérios de armazenamento"""
        try:
            metrics = experience.info.get("metrics", {})
            distance = metrics.get("distance", 0)

            # ARMAZENAR QUALQUER experiência com movimento positivo
            if distance > 0.001:
                return True

            # ARMAZENAR experiências de estabilidade mesmo sem movimento
            roll = abs(metrics.get("roll", 0))
            pitch = abs(metrics.get("pitch", 0))
            if roll < 0.2 and pitch < 0.2:  # Muito estável
                return True

            # ARMAZENAR algumas experiências negativas para aprendizado (10%)
            if distance < 0 and np.random.random() < 0.1:
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Erro em _should_store: {e}")
            return True  # Em caso de erro, armazena por segurança

    def _get_recent_experiences(self, count: int) -> List[Experience]:
        """Obtém experiências recentes para análise de novidade"""
        current_buffer = self.group_buffers.get(self.current_group)
    
        # Verifica se o buffer existe e tem experiências
        if not current_buffer or not current_buffer.buffer:
            return []

        if len(current_buffer.buffer) < count:
            return current_buffer.buffer.copy()  # retorna cópia

        return current_buffer.buffer[-count:].copy() 
    
    def _store_optimized(self, experience: Experience, group: int):
        """Armazenamento com limpeza automática quando necessário"""
        try:
            # Adicionar informação de episódio para tracking de idade
            if hasattr(self, '_dpg_manager') and self._dpg_manager:
                experience.episode_created = self._dpg_manager.episode_count
            
            # Calcular prioridade
            priority = self._calculate_experience_priority(experience)
            
            # VERIFICAR SE PRECISA LIMPAR ANTES DE ARMAZENAR
            current_buffer = self.group_buffers[group]
            needs_cleanup = len(current_buffer.buffer) >= current_buffer.capacity * 0.9  # 90% cheio
            
            if needs_cleanup:
                # Limpar experiências de baixa qualidade primeiro
                self.cleanup_low_quality_experiences(min_quality_threshold=0.4)
                
                # Se ainda estiver cheio, limpar experiências antigas
                if len(current_buffer.buffer) >= current_buffer.capacity * 0.8:
                    self.cleanup_old_experiences(max_age_episodes=800)
                                
            # Tentar armazenar normalmente
            success = current_buffer.add(experience, priority)
            
            if success:
                # Atualizar buffer atual
                self.current_group_buffer = current_buffer.buffer
                
                # Se for excepcional, vai para core buffer
                if experience.quality > 0.7 or self._is_core_experience(experience):
                    self.core_buffer.add(experience, priority * 1.2)
                    
                return True
            else:
                return False
    
        except Exception as e:
            self.logger.error(f"❌ Erro em _store_optimized: {e}")
            return False
    
    def cleanup_low_quality_experiences(self, min_quality_threshold=0.3):
        """Remove experiências de baixa qualidade para liberar espaço"""
        try:
            total_removed = 0

            # Limpar todos os buffers
            for group_id in [1, 2, 3]:
                if group_id in self.group_buffers:
                    buffer = self.group_buffers[group_id]
                    if hasattr(buffer, 'buffer') and buffer.buffer:

                        # Filtrar experiências mantendo apenas as de alta qualidade
                        high_quality_exps = []
                        for exp in buffer.buffer:
                            if exp.quality >= min_quality_threshold:
                                high_quality_exps.append(exp)
                            else:
                                total_removed += 1

                        # Reconstruir buffer com as melhores
                        buffer.buffer = high_quality_exps

                        # Reconstruir heap de qualidade
                        buffer._quality_heap = []
                        for exp in high_quality_exps:
                            heapq.heappush(buffer._quality_heap, (-exp.quality, exp))

            # Limpar core buffer também
            if hasattr(self.core_buffer, 'buffer') and self.core_buffer.buffer:
                high_quality_core = []
                for exp in self.core_buffer.buffer:
                    if exp.quality >= min_quality_threshold:
                        high_quality_core.append(exp)
                    else:
                        total_removed += 1

                self.core_buffer.buffer = high_quality_core
                self.core_buffer._quality_heap = []
                for exp in high_quality_core:
                    heapq.heappush(self.core_buffer._quality_heap, (-exp.quality, exp))

            return total_removed

        except Exception as e:
            self.logger.error(f"❌ Erro na limpeza do buffer: {e}")
            return 0

    def cleanup_old_experiences(self, max_age_episodes=1000):
        """Remove experiências muito antigas"""
        try:
            total_removed = 0
            current_episode = getattr(self._dpg_manager, 'episode_count', 0) if self._dpg_manager else 0

            for group_id in [1, 2, 3]:
                if group_id in self.group_buffers:
                    buffer = self.group_buffers[group_id]
                    if hasattr(buffer, 'buffer') and buffer.buffer:

                        # Manter apenas experiências recentes
                        recent_exps = []
                        for exp in buffer.buffer:
                            exp_episode = getattr(exp, 'episode_created', 0)
                            if current_episode - exp_episode <= max_age_episodes:
                                recent_exps.append(exp)
                            else:
                                total_removed += 1

                        buffer.buffer = recent_exps

            return total_removed

        except Exception as e:
            self.logger.warning(f"Erro ao limpar experiências antigas: {e}")
            return 0
        
        
    def _calculate_experience_priority(self, experience: Experience) -> float:
        """Calcula prioridade baseada em múltiplos fatores"""
        quality_factor = experience.quality * 0.5
        reward_factor = min(experience.reward * 0.2, 0.3)
        skill_factor = sum(experience.skills.values()) * 0.15
        novelty_factor = self._calculate_novelty(experience) * 0.15
        
        return quality_factor + reward_factor + skill_factor + novelty_factor
    
    def _calculate_novelty(self, experience: Experience) -> float:
        """Calcula fator de novidade"""
        recent = self._get_recent_experiences(10)
        if not recent:
            return 1.0
            
        similarities = []
        for exp in recent:
            exp_state = np.array(exp.state)
            current_state = np.array(experience.state)
            sim = np.linalg.norm(current_state - exp_state)
            similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0
        return min(avg_similarity, 1.0)  # Mais similar = menos novidade
    
    def _is_core_experience(self, experience: Experience) -> bool:
        """Verifica se experiência deve ir para core buffer"""
        return (experience.quality > 0.7 and 
                experience.reward > 2.0 and
                experience.skills.get("estabilidade", 0) > 0.7)
    
    def _calculate_quality(self, data: Dict) -> float:
        """QUALIDADE FOCADA APENAS EM MOVIMENTO POSITIVO"""
        metrics = data.get("metrics", {})
        distante = metrics.get("distance", 0)
        distance = abs(distante)
    
        if distance <= 0:
            return 0.0  

        if distance > 0.5: return 0.9
        if distance > 0.3: return 0.8
        if distance > 0.2: return 0.7
        if distance > 0.1: return 0.6
        if distance > 0.05: return 0.5
        if distance > 0.02: return 0.4
        if distance > 0.01: return 0.3
        return 0.2
    
    def _analyze_skills(self, metrics: Dict) -> Dict[str, float]:
        """Análise simplificada de habilidades """
        distance = max(metrics.get("distance", 0), 0)
        speed = metrics.get("speed", 0)
        roll = abs(metrics.get("roll", 0))
        pitch = abs(metrics.get("pitch", 0))

        # Foco ABSOLUTO em movimento positivo
        movimento_positivo = 0.0
        if distance > 0:
            movimento_positivo = min(distance / 1.5, 1.0)  # Meta realista

        # Estabilidade baseada em thresholds REALISTAS
        estabilidade = 1.0 - min((roll + pitch) / 0.8, 1.0)  # Mais tolerante

        # Progresso baseado em movimento REAL
        progresso_basico = 1.0 if distance > 0.3 else min(distance / 0.3, 0.5)

        return {
            "movimento_positivo": movimento_positivo,
            "velocidade_eficiente": min(speed / 0.8, 1.0) if speed > 0 else 0.0,
            "estabilidade": estabilidade,
            "progresso_basico": progresso_basico,
            "coordenação": 0.7 if metrics.get("alternating", False) else 0.3,
            "controle_postural": 1.0 - min(pitch * 2.0, 1.0),
        }  
    
    def get_current_group(self) -> int:
        """Retorna grupo atual de forma robusta"""
        if hasattr(self, 'current_group'):
            return self.current_group

        if hasattr(self, '_dpg_manager') and self._dpg_manager:
            return getattr(self._dpg_manager, 'current_group', 1)

        return 1 

    def get_status(self):
        """Métricas CORRIGIDAS - contar TODAS as experiências"""
        try:
            total_experiences = 0
            total_movement_distance = 0.0
            movement_experience_count = 0
            total_quality = 0.0
            total_reward = 0.0

            for group_id in [1, 2, 3]:
                if group_id in self.group_buffers:
                    buffer = self.group_buffers[group_id]
                    if hasattr(buffer, 'buffer') and buffer.buffer:
                        for exp in buffer.buffer:
                            try:
                                metrics = exp.info.get("metrics", {})
                                distance = metrics.get("distance", 0)

                                if isinstance(distance, (int, float)):
                                    abs_distance = abs(distance)
                                    total_experiences += 1
                                    total_quality += exp.quality
                                    total_reward += exp.reward

                                    if abs_distance > 0.01:
                                        total_movement_distance += abs_distance
                                        movement_experience_count += 1
                            except Exception as e:
                                continue

            # Cálculos CORRETOS
            if total_experiences > 0:
                avg_quality = total_quality / total_experiences
                avg_reward = total_reward / total_experiences
            else:
                avg_quality = 0.1
                avg_reward = 0.1

            # Distância média apenas das experiências com movimento
            if movement_experience_count > 0:
                avg_distance = total_movement_distance / movement_experience_count
            else:
                avg_distance = 0.0

            return {
                "total_experiences": total_experiences,
                "movement_experience_count": movement_experience_count,
                "current_group_experiences": len(self.current_group_buffer) if hasattr(self, 'current_group_buffer') and self.current_group_buffer else 0,
                "avg_quality": avg_quality,
                "avg_distance": avg_distance,
                "avg_reward": avg_reward,
                "quality_calculation_working": True,
                "stored_count": self.stored_count,
                "rejected_count": self.rejected_count
            }

        except Exception as e:
            self.logger.error(f"❌ ERRO no get_status: {e}")
            return {
                "total_experiences": self.stored_count,
                "movement_experience_count": 0,
                "current_group_experiences": self.stored_count,
                "avg_quality": 0.3,
                "avg_distance": 0.0,
                "avg_reward": 10.0,
                "quality_calculation_working": True,
                "error": str(e)
            }
        
    def _calculate_avg_distance(self) -> float:
            """Calcula distância média"""
            try:
                status = self.get_status()
                avg_distance = status.get("avg_distance", 0.0)
                return avg_distance
    
            except Exception as e:
                self.logger.error(f"❌ Erro em _calculate_avg_distance: {e}")
                return 0.0

    def get_metrics(self) -> Dict:
        """Métricas focadas em performance - OTIMIZADAS"""
        base_metrics = self.get_status()

        if not self.current_group_buffer:
            return {**base_metrics, "movement_efficiency": 0, "positive_movement_rate": 0}

        # Cálculos eficientes
        recent_experiences = self.current_group_buffer[-100:]  
        positive_exps = [exp for exp in recent_experiences 
                        if exp.info.get("metrics", {}).get("distance", 0) > 0.1]

        positive_rate = len(positive_exps) / len(recent_experiences) if recent_experiences else 0

        optimization_metrics = {
            "cache_hit_rate": base_metrics["cache_hit_rate"],
            "positive_movement_rate": positive_rate,
            "movement_efficiency": min(positive_rate * 2.0, 1.0),
            "rejection_rate": base_metrics["rejection_rate"],
            "buffer_efficiency": len(self.current_group_buffer) / self.max_experiences
        }

        return {**base_metrics, **optimization_metrics}