# dpg_buffer.py
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


class SkillTransferMap:
    """Mapeamento de habilidades transferíveis entre grupos"""
    
    def __init__(self):
        self.skill_transfer_rules = {
            # Fundação → Desenvolvimento
            (1, 2): {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico", "coordenação"],
                "skill_weights": {"estabilidade": 0.4, "controle_postural": 0.3, "progresso_basico": 0.2, "coordenação": 0.1},
                "relevance_threshold": 0.5
            },
            # Desenvolvimento → Domínio
            (2, 3): {
                "transferable_skills": ["coordenação", "controle_velocidade", "eficiência", "estabilidade"],
                "skill_weights": {"coordenação": 0.3, "controle_velocidade": 0.3, "eficiência": 0.2, "estabilidade": 0.1},
                "relevance_threshold": 0.6
            },
            # Regressões
            (2, 1): {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico"],
                "skill_weights": {"estabilidade": 0.5, "controle_postural": 0.3, "progresso_basico": 0.2},
                "relevance_threshold": 0.5 
            },
            (3, 2): {
                "transferable_skills": ["coordenação", "eficiência", "estabilidade"],
                "skill_weights": {"coordenação": 0.4, "eficiência": 0.3, "estabilidade": 0.3},
                "relevance_threshold": 0.6 
            }
        }
    
    def get_transfer_rules(self, old_group: int, new_group: int) -> Dict:
        """Obtém regras de transferência para transição"""
        if old_group == new_group:
            return {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico", "coordenação", "eficiência", "controle_velocidade"],
                "skill_weights": {
                    "estabilidade": 0.2, 
                    "controle_postural": 0.2, 
                    "progresso_basico": 0.2, 
                    "coordenação": 0.15, 
                    "eficiência": 0.15,
                    "controle_velocidade": 0.1
                },
                "relevance_threshold": 0.3  
            }
        
        return self.skill_transfer_rules.get((old_group, new_group), {
            "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico"],
            "skill_weights": {"estabilidade": 0.4, "controle_postural": 0.4, "progresso_basico": 0.2},
            "relevance_threshold": 0.5  
        })
    
    def calculate_skill_relevance(self, experience: Experience, target_group: int) -> float:
        """Calcula relevância da experiência para o grupo alvo"""
        rules = self.get_transfer_rules(experience.group, target_group)
        
        if not rules["transferable_skills"]:
            return 0.0
        
        relevance = 0.0
        for skill, weight in rules["skill_weights"].items():
            skill_value = experience.skills.get(skill, 0.0)
            relevance += skill_value * weight
        
        if experience.quality > 0.8:
            relevance *= 1.3  
        elif experience.quality > 0.6:
            relevance *= 1.1  
            
        return min(relevance, 1.0)


class SmartBufferManager:
    """
    ESPECIALISTA EM MEMÓRIA com Preservação Inteligente
    """
    
    def __init__(self, logger, config, max_core_experiences=2000):
        self.logger = logger
        self.config = config
        self.max_core_experiences = max_core_experiences
        
        # Sistema de memória hierárquico
        self.group_buffers = {0: []}
        self.core_buffer = deque(maxlen=max_core_experiences)
        self.current_group_buffer = []
        
        # Sistema de preservação
        self.skill_map = SkillTransferMap()
        self.preservation_stats = {
            "total_transitions": 0,
            "experiences_preserved": 0,
            "preservation_rate": 0.0
        }
        
        # Otimizações
        self._last_sort_episode = 0
        self.sort_interval = 100  # Ordenar a cada 100 episódios
        self._high_quality_cache = None
        self._cache_valid = False
    
        # Estatísticas
        self.experience_count = 0
        self.group_transitions = 0
    
    def store_experience(self, experience_data: Dict):
        """Armazenamento simplificado e consistente"""
        try:
            if not experience_data:
                return

            current_group = self.get_current_group()
            experience_data["group_level"] = current_group
            experience = self._create_enhanced_experience(experience_data)

            self._store_hierarchical_optimized(experience, current_group)

            if self._is_fundamental_experience(experience):
                import time
                heap_item = (-experience.quality, time.time_ns(), experience)

                if len(self.core_buffer_heap) < self.max_core_experiences:
                    heapq.heappush(self.core_buffer_heap, heap_item)
                else:
                    worst_quality, worst_timestamp, worst_exp = self.core_buffer_heap[0]
                    if experience.quality > -worst_quality:
                        heapq.heapreplace(self.core_buffer_heap, heap_item)

            self.experience_count += 1
            self._invalidate_cache()

        except Exception as e:
            self.logger.warning(f"Erro ao armazenar experiência: {e}")
    
    def get_current_group(self) -> int:
        """Retorna o grupo atual baseado no buffer atual"""
        if hasattr(self, '_dpg_manager') and self._dpg_manager:
            return getattr(self._dpg_manager, 'current_group', 1)

        return getattr(self, '_current_group', 1) 

    def _create_enhanced_experience(self, data: Dict) -> Experience:
        """Cria experiência com análise de habilidades"""
        state = data["state"]
        action = data["action"]
        reward = data["reward"]
        phase_info = data["phase_info"]
        metrics = data["metrics"]
        
        quality = self._calculate_experience_quality(state, action, reward, metrics)
        skills = self._analyze_experience_skills(metrics, phase_info)
        
        return Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=state,
            done=False,
            info=phase_info,
            group=data.get("group_level", 1),
            sub_phase=phase_info.get("sub_phase", 0),
            quality=quality,
            skills=skills
        )
    
    def _analyze_experience_skills(self, metrics: Dict, phase_info: Dict) -> Dict[str, float]:
        """Analisa habilidades demonstradas na experiência"""
        skills = {}
        
        # Habilidade de estabilidade
        roll = metrics.get("roll", 0)
        pitch = metrics.get("pitch", 0)
        skills["estabilidade"] = 1.0 - min(abs(roll) + abs(pitch), 1.0)
        
        # Habilidade de progresso
        distance = metrics.get("distance", 0)
        skills["progresso_basico"] = min(distance / 2.0, 1.0)
        
        # Habilidade de coordenação
        left_contact = metrics.get("left_contact", False)
        right_contact = metrics.get("right_contact", False)
        skills["coordenação"] = 1.0 if left_contact != right_contact else 0.3
        
        # Habilidade de eficiência
        steps = metrics.get("steps", 1)
        skills["eficiência"] = min(distance / max(steps, 1), 1.0)
        
        # Habilidade de controle de velocidade
        speed = metrics.get("speed", 0)
        target_speed = phase_info.get('target_speed', 1.0)
        speed_error = abs(speed - target_speed)
        skills["controle_velocidade"] = 1.0 - min(speed_error / target_speed, 1.0)
        
        # Habilidade de controle postural
        skills["controle_postural"] = 1.0 - min(abs(pitch) * 2.0, 1.0)

        return skills

    def transition_with_preservation(self, old_group: int, new_group: int, adaptive_config: Dict):
        """Transição inteligente com preservação de aprendizado """
        self.group_transitions += 1
        if old_group not in self.group_buffers:
            self.group_buffers[old_group] = []
        if new_group not in self.group_buffers:
            self.group_buffers[new_group] = []
        if old_group == new_group:
            preserved_experiences = self.group_buffers[old_group] + list(self.core_buffer)
        else:
            relevant = self._filter_relevant_experiences(self.group_buffers[old_group], new_group)
            preserved_experiences = relevant + list(self.core_buffer)
        max_preserved = 2000 
        if len(preserved_experiences) > max_preserved:
            preserved_experiences.sort(key=lambda x: (x.quality, x.reward), reverse=True)
            preserved_experiences = preserved_experiences[:max_preserved]
        self.group_buffers[new_group] = preserved_experiences
        self.current_group_buffer = preserved_experiences
            
    def _filter_relevant_experiences(self, experiences: List[Experience], new_group: int) -> List[Experience]:
        """Filtra experiências relevantes para o novo grupo"""
        relevant = []
        
        for exp in experiences:
            relevance = self.skill_map.calculate_skill_relevance(exp, new_group)
            rules = self.skill_map.get_transfer_rules(exp.group, new_group)
            
            if relevance >= rules["relevance_threshold"] or exp.quality > 0.8:
                relevant.append(exp)
        
        relevant.sort(key=lambda x: self.skill_map.calculate_skill_relevance(x, new_group) * 0.7 + x.quality * 0.3, reverse=True)
        
        return relevant
    
    def _apply_preservation_policy(self, experiences: List[Experience], policy: str) -> List[Experience]:
        """Aplica política de preservação"""
        policy_limits = {
            "high": 3000,    
            "medium": 2000,  
            "low": 1500     
        }
        
        limit = policy_limits.get(policy, 2000)
        if len(experiences) <= limit:
            return experiences
        high_quality = [exp for exp in experiences if exp.quality > 0.7]
        medium_quality = [exp for exp in experiences if 0.4 <= exp.quality <= 0.7]
        low_quality = [exp for exp in experiences if exp.quality < 0.4]
        
        preserved = []
        
        high_limit = int(limit * 0.7)
        if len(high_quality) > high_limit:
            high_quality.sort(key=lambda x: x.quality, reverse=True)
            preserved.extend(high_quality[:high_limit])
        else:
            preserved.extend(high_quality)
        
        remaining_slots = limit - len(preserved)
        medium_limit = int(remaining_slots * 0.9)  
        
        if len(medium_quality) > medium_limit:
            medium_quality.sort(key=lambda x: x.quality, reverse=True)
            preserved.extend(medium_quality[:medium_limit])
        else:
            preserved.extend(medium_quality)
        
        remaining_slots = limit - len(preserved)
        if remaining_slots > 0 and low_quality:
            low_quality.sort(key=lambda x: x.reward, reverse=True)
            preserved.extend(low_quality[:remaining_slots])
        
        return preserved
    
    def _calculate_experience_quality(self, state, action, reward, metrics) -> float:
        """Calcula qualidade da experiência"""
        quality = 0.0
        
        # Fator de recompensa
        quality += min(abs(reward) * 0.3, 1.0)
        
        # Fator de progresso
        progress = metrics.get("distance", 0)
        if progress > 0:
            quality += min(progress * 2.0, 1.0)
        
        # Fator de estabilidade
        stability = 1.0 - min(metrics.get("roll", 0) + metrics.get("pitch", 0), 1.0)
        quality += stability * 0.4

        # Fator de sucesso
        success = metrics.get("success", False)
        if success:
            quality += 0.5
        
        return min(quality, 1.0)
    
    def _is_fundamental_experience(self, experience: Experience) -> bool:
        """Verifica se experiência é fundamental"""
        return (experience.quality > 0.6 and 
                experience.reward > 0.3 and
                experience.skills.get("estabilidade", 0) > 0.5)
    
    def _store_hierarchical(self, experience: Experience, group: int, sub_phase: int):
        """Armazena experiência na hierarquia"""
        if group not in self.group_buffers:
            self.group_buffers[group] = []
            self.logger.info(f"📁 Criado novo grupo {group} no buffer")

        self.group_buffers[group].append(experience)
        self.current_group_buffer = self.group_buffers[group]

        if len(self.group_buffers[group]) > 3000:
            self.group_buffers[group].sort(key=lambda x: x.quality, reverse=True)
            self.group_buffers[group] = self.group_buffers[group][:2500]
            self.current_group_buffer = self.group_buffers[group]
    
    def get_training_batch(self, batch_size=32):
        """Retorna batch para treinamento"""
        if not self.current_group_buffer:
            return None
        available = []
        group_experiences = self.current_group_buffer[:]
        available.extend(group_experiences)
        core_experiences = [exp for exp in self.core_buffer 
                          if self._is_relevant_for_current_group(exp)]
        available.extend(core_experiences[:len(core_experiences)//3])
        if len(available) < batch_size:
            batch_size = len(available)
        high_quality = [exp for exp in available if exp.quality > 0.7]
        medium_quality = [exp for exp in available if 0.4 <= exp.quality <= 0.7]
        low_quality = [exp for exp in available if exp.quality < 0.4]
        selected = []
        hq_count = min(len(high_quality), int(batch_size * 0.6))
        mq_count = min(len(medium_quality), int(batch_size * 0.3))
        lq_count = batch_size - hq_count - mq_count
        if hq_count > 0:
            selected.extend(high_quality[:hq_count])
        if mq_count > 0:
            selected.extend(medium_quality[:mq_count])
        if lq_count > 0 and len(low_quality) > 0:
            positive_low = [exp for exp in low_quality if exp.reward > 0]
            selected.extend(positive_low[:lq_count])
            
        return selected
    
    def _is_relevant_for_current_group(self, experience: Experience) -> bool:
        """Verifica se experiência do core é relevante para grupo atual"""
        current_group = self.get_current_group()
        if self._is_fundamental_experience(experience):
            return True
        relevance = self.skill_map.calculate_skill_relevance(experience, current_group)
        
        return relevance > 0.5
    
    def get_status(self):
        """Retorna status com estatísticas de preservação"""
        return {
            "total_experiences": self.experience_count,
            "core_experiences": len(self.core_buffer),
            "current_group_experiences": len(self.current_group_buffer),
            "group_transitions": self.group_transitions,
            "preservation_stats": self.preservation_stats,
            "groups_with_buffer": list(self.group_buffers.keys())
        }
    
    def get_metrics(self) -> Dict:
        """Retorna métricas para monitoramento"""
        if not self.current_group_buffer:
            return {
                "buffer_avg_quality": 0,
                "buffer_avg_reward": 0,
                "core_buffer_size": len(self.core_buffer),
                "current_buffer_size": len(self.current_group_buffer),
                "learning_convergence": 0,
                "memory_efficiency": 0,
            }

        avg_quality = np.mean([exp.quality for exp in self.current_group_buffer])
        avg_reward = np.mean([exp.reward for exp in self.current_group_buffer])

        return {
            "buffer_avg_quality": avg_quality,
            "buffer_avg_reward": avg_reward,
            "core_buffer_size": len(self.core_buffer),
            "current_buffer_size": len(self.current_group_buffer),
            "learning_convergence": 0.5,  
            "memory_efficiency": self.preservation_stats.get("preservation_rate", 0.0),
        }
    
class OptimizedBufferManager(SmartBufferManager):
    def __init__(self, logger, config, max_core_experiences=2000):
        super().__init__(logger, config, max_core_experiences)
        
        self.core_buffer_heap = []
        self._high_quality_experiences = set()
        self._relevance_cache = {}
        self._cache_episode = 0
        
        # Estatísticas de performance
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "heap_operations": 0,
            "sort_operations_saved": 0
        }
        
    def store_experience(self, experience_data: Dict):
        """Armazenamento otimizado com inserção O(log n)"""
        try:
            if not experience_data:
                return
                
            current_group = self.get_current_group()
            experience_data["group_level"] = current_group
            experience = self._create_enhanced_experience(experience_data)
            
            self._store_hierarchical_optimized(experience, current_group)
            
            if self._is_fundamental_experience(experience):
                if len(self.core_buffer_heap) < self.max_core_experiences:
                    heapq.heappush(self.core_buffer_heap, (-experience.quality, experience))
                else:
                    # Substituir pior experiência em O(log n)
                    worst_quality, worst_exp = self.core_buffer_heap[0]
                    if experience.quality > -worst_quality:
                        heapq.heapreplace(self.core_buffer_heap, (-experience.quality, experience))
            
            self.experience_count += 1
            self._invalidate_cache()
            
        except Exception as e:
            self.logger.warning(f"Erro ao armazenar experiência: {e}")
    
    def _store_hierarchical_optimized(self, experience: Experience, group: int):
        """Armazenamento otimizado por grupo"""
        if group not in self.group_buffers:
            self.group_buffers[group] = []
            
        self.group_buffers[group].append(experience)
        self.current_group_buffer = self.group_buffers[group]
        
        if len(self.group_buffers[group]) > 3000:
            # Manter apenas as melhores 2500 experiências
            self.group_buffers[group].sort(key=lambda x: x.quality, reverse=True)
            self.group_buffers[group] = self.group_buffers[group][:2500]
            self.current_group_buffer = self.group_buffers[group]
    
    def get_training_batch(self, batch_size=32):
        """Batch otimizado com cache de relevância e métricas"""
        if not self.current_group_buffer:
            return None
            
        current_group = self.get_current_group()
        cache_key = f"group_{current_group}_ep_{self.episode_count//100}"
        
        if cache_key in self._relevance_cache:
            self.performance_stats["cache_hits"] += 1
            available = self._relevance_cache[cache_key]
        else:
            self.performance_stats["cache_misses"] += 1
            available = self._get_relevant_experiences_optimized(current_group)
            self._relevance_cache[cache_key] = available
            # Limpar cache antigo
            old_keys = [k for k in self._relevance_cache.keys() if k != cache_key]
            for k in old_keys:
                del self._relevance_cache[k]
        
        if len(available) < batch_size:
            batch_size = len(available)
            
        return self._stratified_sampling_optimized(available, batch_size)
    
    def _get_relevant_experiences_optimized(self, current_group: int) -> List[Experience]:
        """Experiências relevantes com cache"""
        relevant = []
        
        # Grupo atual
        group_exps = self.group_buffers.get(current_group, [])
        relevant.extend(group_exps[:1000])  # Limitar para performance
        
        # Core buffer (apenas as melhores)
        core_exps = [exp for _, exp in self.core_buffer_heap[-500:]]  # Top 500
        relevant.extend(core_exps[:len(core_exps)//3])
        
        return relevant
    
    def _stratified_sampling_optimized(self, experiences: List[Experience], batch_size: int) -> List[Experience]:
        """Amostragem estratificada otimizada"""
        if not experiences:
            return []
            
        high_quality = []
        medium_quality = []
        low_quality = []
        
        for exp in experiences:
            if exp.quality > 0.7:
                high_quality.append(exp)
            elif exp.quality > 0.4:
                medium_quality.append(exp)
            else:
                low_quality.append(exp)
        
        selected = []
        hq_count = min(len(high_quality), int(batch_size * 0.6))
        mq_count = min(len(medium_quality), int(batch_size * 0.3))
        lq_count = batch_size - hq_count - mq_count
        
        selected.extend(high_quality[:hq_count])
        selected.extend(medium_quality[:mq_count])
        
        if lq_count > 0 and low_quality:
            # Apenas low quality com recompensa positiva
            positive_low = [exp for exp in low_quality if exp.reward > 0]
            selected.extend(positive_low[:lq_count])
            
        return selected
    
    def get_metrics(self) -> Dict:
        """Retorna métricas aprimoradas com estatísticas de performance"""
        base_metrics = super().get_metrics()
        
        # Adicionar métricas de otimização
        optimization_metrics = {
            "cache_hit_rate": self.performance_stats["cache_hits"] / max(
                self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"], 1
            ),
            "heap_size": len(self.core_buffer_heap),
            "relevance_cache_size": len(self._relevance_cache),
            "estimated_efficiency_gain": self._calculate_efficiency_gain()
        }
        
        return {**base_metrics, **optimization_metrics}
    
    def _calculate_efficiency_gain(self) -> float:
        """Calcula ganho estimado de eficiência"""
        total_operations = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        if total_operations == 0:
            return 0.0
            
        cache_efficiency = self.performance_stats["cache_hits"] / total_operations
        heap_efficiency = min(self.performance_stats["heap_operations"] / max(self.experience_count, 1), 1.0)
        
        # Estimativa conservadora: 15-25% de ganho
        estimated_gain = (cache_efficiency * 0.15 + heap_efficiency * 0.10)
        return min(estimated_gain, 0.25)
    
    def _invalidate_cache(self):
        """Invalidar cache quando o buffer muda significativamente"""
        self._high_quality_experiences.clear()
        self._cache_episode += 1


class IntelligentCache:
    """Sistema de cache inteligente para cálculos repetitivos"""
    
    def __init__(self, max_size=1000, default_ttl=100):
        self._cache = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        
    def get(self, key: str) -> Any:
        """Obtém valor do cache com estatísticas"""
        if key in self._cache:
            value, timestamp, ttl = self._cache[key]
            if time.time() - timestamp < ttl:
                self._hits += 1
                return value
            else:
                del self._cache[key]  # Expirou
                
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Define valor no cache com TTL"""
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
            
        self._cache[key] = (value, time.time(), ttl or self._default_ttl)
    
    def _evict_oldest(self):
        """Remove entradas mais antigas"""
        if not self._cache:
            return
            
        oldest_key = min(self._cache.keys(), 
                        key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
    
    def get_hit_rate(self) -> float:
        """Taxa de acerto do cache"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

