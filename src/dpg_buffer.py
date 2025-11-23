# dpg_buffer.py
import random
import numpy as np
from typing import Deque, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
import heapq
from collections import deque, OrderedDict
import time

@dataclass
class Experience:
    """Estrutura para armazenar experiências com métricas brutas"""
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
    episode_created: int = 0
    # MÉTRICAS BRUTAS PARA REAVALIAÇÃO
    raw_metrics: Dict[str, float] = None
    quality_version: int = 1  

class AdaptiveQualityEvaluator:
    """Avaliador de qualidade adaptativo que se ajusta aos pesos do crítico"""
    
    def __init__(self):
        self.quality_functions = {
            "default": self._default_quality,
            "movement_focus": self._movement_focus_quality,
            "stability_focus": self._stability_focus_quality,
            "coordination_focus": self._coordination_focus_quality,
            "efficiency_focus": self._efficiency_focus_quality
        }
        self.current_strategy = "default"
        self.critic_weights = None
        self.quality_version = 1
        
    def update_critic_weights(self, weights: Dict[str, float]):
        """Atualiza pesos do crítico e ajusta estratégia"""
        self.critic_weights = weights
        
        # Determina estratégia baseada nos pesos do crítico
        if weights:
            max_component = max(weights.items(), key=lambda x: x[1])[0]
            if max_component == "stability" and weights["stability"] > 0.4:
                self.current_strategy = "stability_focus"
            elif max_component == "propulsion" and weights["propulsion"] > 0.4:
                self.current_strategy = "movement_focus"
            elif max_component == "coordination" and weights["coordination"] > 0.4:
                self.current_strategy = "coordination_focus"
            elif max_component == "efficiency" and weights["efficiency"] > 0.4:
                self.current_strategy = "efficiency_focus"
            else:
                self.current_strategy = "default"
        
        self.quality_version += 1
        
    def evaluate_quality(self, metrics: Dict[str, float], current_group: int) -> float:
        """Avalia qualidade adaptativamente baseado na estratégia atual"""
        if self.current_strategy in self.quality_functions:
            base_quality = self.quality_functions[self.current_strategy](metrics, current_group)
        else:
            base_quality = self._default_quality(metrics, current_group)
            
        # APLICA PESOS DO CRÍTICO SE DISPONÍVEIS
        if self.critic_weights:
            weighted_quality = self._apply_critic_weights(metrics, base_quality)
            return min(weighted_quality, 1.0)
            
        return base_quality
    
    def _default_quality(self, metrics: Dict, group: int) -> float:
        """Qualidade padrão - balanceada"""
        distance = max(metrics.get("distance", 0), 0)
        stability = 1.0 - min((abs(metrics.get("roll", 0)) + abs(metrics.get("pitch", 0))) / 1.0, 1.0)
        coordination = 1.0 if metrics.get("alternating", False) else 0.3
        
        if distance <= 0:
            return 0.0
            
        # Grupo influencia thresholds
        group_factor = 1.0 + (group - 1) * 0.3
        
        base_score = (
            min(distance / (1.0 * group_factor), 0.5) +
            stability * 0.3 +
            coordination * 0.2
        )
        
        return min(base_score, 0.9)
    
    def _movement_focus_quality(self, metrics: Dict, group: int) -> float:
        """Foco em movimento e propulsão"""
        distance = max(metrics.get("distance", 0), 0)
        velocity = metrics.get("speed", 0)
        
        if distance <= 0:
            return 0.0
            
        group_factor = 1.0 + (group - 1) * 0.4
        
        movement_score = min(distance / (0.8 * group_factor), 0.7)
        velocity_score = min(velocity / (0.6 * group_factor), 0.3)
        
        return movement_score + velocity_score
    
    def _stability_focus_quality(self, metrics: Dict, group: int) -> float:
        """Foco em estabilidade e controle postural"""
        distance = max(metrics.get("distance", 0), 0)
        roll = abs(metrics.get("roll", 0))
        pitch = abs(metrics.get("pitch", 0))
        stability = 1.0 - min((roll + pitch) / 0.6, 1.0)  
        
        if distance < 0.05: 
            return 0.0
            
        base_stability = stability * 0.8
        distance_bonus = min(distance / 2.0, 0.2)  
        
        return base_stability + distance_bonus
    
    def _coordination_focus_quality(self, metrics: Dict, group: int) -> float:
        """Foco em coordenação e padrão de marcha"""
        distance = max(metrics.get("distance", 0), 0)
        alternating = metrics.get("alternating", False)
        clearance = metrics.get("clearance_score", 0.0)
        gait_score = metrics.get("gait_pattern_score", 0.0)
        
        if distance < 0.1 or not alternating:
            return 0.0
            
        coordination_base = 0.4 if alternating else 0.0
        clearance_bonus = min(clearance / 0.1, 0.3)  
        gait_bonus = gait_score * 0.2
        distance_bonus = min(distance / 3.0, 0.1)
        
        return coordination_base + clearance_bonus + gait_bonus + distance_bonus
    
    def _efficiency_focus_quality(self, metrics: Dict, group: int) -> float:
        """Foco em eficiência energética"""
        distance = max(metrics.get("distance", 0), 0)
        efficiency = metrics.get("propulsion_efficiency", 0.5)
        energy_used = metrics.get("energy_used", 1.0)
        
        if distance <= 0:
            return 0.0
            
        # Eficiência é o componente principal
        efficiency_score = efficiency * 0.6
        
        # Distância com eficiência
        distance_score = min(distance / 2.0, 0.3)
        
        # Penalidade por alto consumo energético
        energy_penalty = max(0, (energy_used - 0.8) * 0.5) if energy_used > 0.8 else 0
        
        return max(efficiency_score + distance_score - energy_penalty, 0.1)
    
    def _apply_critic_weights(self, metrics: Dict, base_quality: float) -> float:
        """Aplica pesos do crítico para ajuste fino da qualidade"""
        if not self.critic_weights:
            return base_quality
            
        # Calcula componentes individuais
        movement_component = min(metrics.get("distance", 0) / 2.0, 0.4)
        stability_component = 1.0 - min((abs(metrics.get("roll", 0)) + abs(metrics.get("pitch", 0))) / 1.0, 1.0)
        coordination_component = 1.0 if metrics.get("alternating", False) else 0.2
        efficiency_component = metrics.get("propulsion_efficiency", 0.5)
        
        # Aplica pesos
        weighted_score = (
            self.critic_weights.get("propulsion", 0.25) * movement_component +
            self.critic_weights.get("stability", 0.25) * stability_component +
            self.critic_weights.get("coordination", 0.25) * coordination_component +
            self.critic_weights.get("efficiency", 0.25) * efficiency_component
        )
        
        # Combina com qualidade base
        return (base_quality * 0.7) + (weighted_score * 0.3)

class Cache:
    """Cache unificado com LRU + TTL"""
    
    def __init__(self, max_size=1000, base_ttl=100):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._base_ttl = base_ttl
        self._hits = 0
        self._misses = 0
        
    def get(self, key: str) -> Any:
        if key in self._cache:
            value, timestamp, ttl = self._cache[key]
            if time.time() - timestamp < ttl:
                self._hits += 1
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
                
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            
        actual_ttl = ttl or self._base_ttl
        self._cache[key] = (value, time.time(), actual_ttl)
        
    def get_stats(self):
        total = self._hits + self._misses
        return {
            "hits": self._hits, "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache)
        }

class AdaptiveBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)  # Especificar tipo
        self.quality_heap = []  
        self.quality_threshold = 0.3
        self.current_quality_version = 1
        
    def add(self, experience: Experience) -> bool:
        """Adiciona experiência com versionamento de qualidade"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            return True
        else:
            # Substitui a de menor qualidade
            min_idx = self._find_min_quality_index()
            if experience.quality > self._get_effective_quality(self.buffer[min_idx]):
                self.buffer[min_idx] = experience
                return True
            return False
    
    def _get_effective_quality(self, experience: Experience) -> float:
        """Qualidade efetiva considerando versionamento"""
        # Se a qualidade está desatualizada, penaliza temporariamente
        if experience.quality_version < self.current_quality_version:
            return experience.quality * 0.7 
        return experience.quality
    
    def reevaluate_experiences(self, quality_evaluator: AdaptiveQualityEvaluator, 
                             current_group: int, batch_size: int = 200):
        """Reavalia qualidade de experiências desatualizadas em lotes"""
        reevaluated_count = 0
        
        for i, exp in enumerate(self.buffer):
            if exp.quality_version < quality_evaluator.quality_version:
                # Recalcula qualidade com métricas brutas
                new_quality = quality_evaluator.evaluate_quality(
                    exp.raw_metrics, current_group
                )
                
                # Atualiza experiência
                exp.quality = new_quality
                exp.quality_version = quality_evaluator.quality_version
                reevaluated_count += 1
                
                # Limita por batch para não travar o sistema
                if reevaluated_count >= batch_size:
                    break
        
        # Atualiza versão corrente
        self.current_quality_version = quality_evaluator.quality_version
        
        # Reconstrói heap com qualidades atualizadas
        self._rebuild_quality_heap()
        
        return reevaluated_count
    
    def _rebuild_quality_heap(self):
        """Reconstrói o heap de qualidade completamente"""
        self.quality_heap = []
        for i, exp in enumerate(self.buffer):
            effective_quality = self._get_effective_quality(exp)
            if effective_quality > self.quality_threshold:
                heapq.heappush(self.quality_heap, 
                              (-effective_quality, i, exp.quality_version))
    
    def sample(self, batch_size: int, current_group: int) -> List[Experience]:
        """Amostra com consciência de grupo e qualidade adaptativa"""
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
            
        # Estratégia adaptativa baseada no grupo
        if current_group == 1:
            # Grupo 1: 90% aleatório, 10% qualidade (exploração)
            random_ratio = 0.9
        elif current_group == 2:
            # Grupo 2: 70% aleatório, 30% qualidade (balanceado)
            random_ratio = 0.7
        else:
            # Grupo 3+: 50% aleatório, 50% qualidade (exploração qualificada)
            random_ratio = 0.5
            
        random_count = int(batch_size * random_ratio)
        quality_count = batch_size - random_count
        
        # Amostra aleatória
        samples = random.sample(self.buffer, random_count) if random_count > 0 else []
        
        # Amostra de qualidade (com versão atual)
        if self.quality_heap and quality_count > 0:
            quality_samples = self._get_quality_samples(quality_count)
            samples.extend(quality_samples)
            
        return samples
    
    def _get_quality_samples(self, count: int) -> List[Experience]:
        """Obtém amostras de alta qualidade (priorizando versões atualizadas)"""
        # Separa por versão de qualidade
        current_version_samples = []
        outdated_version_samples = []
        
        temp_heap = self.quality_heap.copy()
        sampled_indices = set()
        
        while temp_heap and len(sampled_indices) < count * 2:
            neg_quality, idx, quality_version = heapq.heappop(temp_heap)
            
            if idx >= len(self.buffer) or idx in sampled_indices:
                continue
                
            sampled_indices.add(idx)
            exp = self.buffer[idx]
            
            if quality_version == self.current_quality_version:
                current_version_samples.append(exp)
            else:
                outdated_version_samples.append(exp)
        
        # Prioriza experiências com qualidade atualizada
        result = []
        result.extend(current_version_samples[:count])
        
        # Completa com experiências desatualizadas se necessário
        if len(result) < count:
            result.extend(outdated_version_samples[:count - len(result)])
            
        return result[:count]
    
    def _find_min_quality_index(self) -> int:
        """Encontra índice da experiência com menor qualidade efetiva"""
        if not self.buffer:
            return 0
            
        min_quality = float('inf')
        min_idx = 0
        
        for i, exp in enumerate(self.buffer):
            effective_quality = self._get_effective_quality(exp)
            if effective_quality < min_quality:
                min_quality = effective_quality
                min_idx = i
                
        return min_idx

class AdaptiveBufferManager:
    def __init__(self, logger, config, max_experiences=5000):
        self.logger = logger
        self.config = config
        self.max_experiences = max_experiences
        self.capacity = max_experiences  # ADICIONAR ESTE ATRIBUTO

        # SISTEMA ADAPTATIVO
        self.quality_evaluator = AdaptiveQualityEvaluator()
        self.main_buffer = AdaptiveBuffer(capacity=max_experiences)
        self.cache = Cache(max_size=200)
        
        # CONTROLE DE REAVALIAÇÃO
        self.last_reevaluation_episode = 0
        self.reevaluation_interval = 50  
        self.reevaluation_batch_size = 100  
        
        # ESTATÍSTICAS ADAPTATIVAS
        self.stored_count = 0
        self.rejected_count = 0
        self.reevaluated_count = 0
        self._dpg_manager = None
        self.episode_count = 0
        
    def store_experience(self, experience_data: Dict) -> bool:
        """Armazenamento com critérios mais permissivos"""
        try:
            # Calcula qualidade com avaliador atual
            metrics = experience_data.get("metrics", {})
            current_group = experience_data.get("group_level", 1)
            quality = self.quality_evaluator.evaluate_quality(metrics, current_group)
            
            # FILTRO ADAPTATIVO MAIS PERMISSIVO
            min_threshold = 0.05  
            buffer_size = len(self.main_buffer.buffer)
            
            # No início, aceita quase tudo para construir buffer
            if buffer_size < 100:
                min_threshold = 0.03
            elif buffer_size < 500:
                min_threshold = 0.05
                
            if quality < min_threshold:
                self.rejected_count += 1
                return False
                
            # Cria experiência
            experience = self._create_adaptive_experience(experience_data, quality, current_group)
            
            # Armazena no buffer principal
            success = self.main_buffer.add(experience)
            if success:
                self.stored_count += 1
            else:
                self.rejected_count += 1
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ ERRO no armazenamento adaptativo: {e}")
            return False

    def _create_adaptive_experience(self, data: Dict, quality: float, group: int) -> Experience:
        """Cria experiência com métricas otimizadas"""
        state = np.array(data["state"], dtype=np.float32)
        action = np.array(data["action"], dtype=np.float32)
        
        # MÉTRICAS BRUTAS COM VALORES PADRÃO
        raw_metrics = data.get("metrics", {}).copy()
        
        # Garante métricas essenciais
        essential_metrics = ["distance", "speed", "roll", "pitch", "alternating"]
        for metric in essential_metrics:
            if metric not in raw_metrics or raw_metrics[metric] is None:
                raw_metrics[metric] = 0.0

        # Garante que next_state existe
        next_state = data.get("next_state", data["state"])
        if next_state is None:
            next_state = data["state"]

        return Experience(
            state=state,  
            action=action,
            reward=float(data["reward"]),
            next_state=np.array(next_state, dtype=np.float32),
            done=data.get("done", False),
            info=data.get("phase_info", {}),
            group=group,
            sub_phase=0,
            quality=quality,
            skills=self._analyze_adaptive_skills(raw_metrics),
            episode_created=self.episode_count,
            raw_metrics=raw_metrics,
            quality_version=self.quality_evaluator.quality_version
        )

    def get_adaptive_status(self) -> Dict:
        """Status com métricas adaptativas"""
        buffer_size = len(self.main_buffer.buffer)
        
        # Estatísticas de qualidade
        quality_stats = {
            "current_quality_version": self.quality_evaluator.quality_version,
            "quality_strategy": self.quality_evaluator.current_strategy,
            "avg_quality": 0.0,
            "outdated_experiences": 0,
            "high_quality_count": 0,
            "buffer_utilization": buffer_size / self.capacity if self.capacity > 0 else 0
        }
        
        if buffer_size > 0:
            qualities = []
            for exp in self.main_buffer.buffer:
                qualities.append(exp.quality)
                if exp.quality_version < quality_stats["current_quality_version"]:
                    quality_stats["outdated_experiences"] += 1
                if exp.quality > 0.7:
                    quality_stats["high_quality_count"] += 1
            
            quality_stats["avg_quality"] = sum(qualities) / len(qualities) if qualities else 0.0
        
        base_status = {
            "total_experiences": buffer_size,
            "capacity": self.capacity,
            "stored_count": self.stored_count,
            "rejected_count": self.rejected_count,
            "reevaluated_count": self.reevaluated_count,
            "cache_stats": self.cache.get_stats()
        }
        
        critic_weights = self.quality_evaluator.critic_weights or {}
        return {**base_status, **quality_stats, **critic_weights}
    
    def update_quality_criteria(self, critic_weights: Dict, current_group: int, 
                              current_episode: int, force_reevaluation: bool = False):
        """Atualiza critérios de qualidade e reavalia se necessário"""
        # Atualiza avaliador
        self.quality_evaluator.update_critic_weights(critic_weights)
        
        # Verifica se deve reavaliar
        should_reevaluate = (
            force_reevaluation or
            (current_episode - self.last_reevaluation_episode) >= self.reevaluation_interval or
            len(self.main_buffer.buffer) > self.max_experiences * 0.8
        )
        
        if should_reevaluate:
            reevaluated = self.main_buffer.reevaluate_experiences(
                self.quality_evaluator, current_group, self.reevaluation_batch_size
            )
            self.reevaluated_count += reevaluated
            self.last_reevaluation_episode = current_episode
            
    def sample(self, batch_size: int, current_group: int) -> List[Experience]:
        """Amostra com consciência de grupo"""
        return self.main_buffer.sample(batch_size, current_group)
    
    def _analyze_adaptive_skills(self, metrics: Dict) -> Dict[str, float]:
        """Análise de habilidades para reavaliação"""
        distance = max(metrics.get("distance", 0), 0)
        roll = abs(metrics.get("roll", 0))
        pitch = abs(metrics.get("pitch", 0))
        alternating = metrics.get("alternating", False)
        efficiency = metrics.get("propulsion_efficiency", 0.5)
        
        return {
            "movement": min(distance / 2.0, 1.0),
            "stability": 1.0 - min((roll + pitch) / 0.8, 1.0),
            "coordination": 0.8 if alternating else 0.2,
            "efficiency": efficiency,
            "clearance": metrics.get("clearance_score", 0.0),
            "gait_quality": metrics.get("gait_pattern_score", 0.5)
        }