# dpg_valence.py (VERSÃO OTIMIZADA)
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ValenceState(Enum):
    INACTIVE = "inative"
    LEARNING = "learning"
    CONSOLIDATING = "consolidating"
    MASTERED = "mastered"

@dataclass
class ValenceConfig:
    name: str
    target_level: float
    metrics: List[str]
    reward_components: List[str]
    dependencies: List[str]
    activation_threshold: float = 0.3
    mastery_threshold: float = 0.7

class ValenceTracker:
    def __init__(self, valence_name: str):
        self.valence_name = valence_name
        self.current_level = 0.0
        self.history = []  
        self.episodes_active = 0
        self.state = ValenceState.INACTIVE
    
    def update_level(self, new_level: float, episode: int):
        self.current_level = new_level
        self.history.append((episode, new_level))
        if len(self.history) > 50:  # Histórico menor
            self.history.pop(0)

class ValenceManager:
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config or {}
        
        # Sistema de valências SIMPLIFICADO
        self.valences = self._initialize_valences()
        self.valence_performance = {}
        self.active_valences = set()
        self.valence_weights = {}
        
        # Estado do sistema
        self.episode_count = 0
        self.overall_progress = 0.0

        # Cache otimizado
        self._cached_levels = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100  # Cache menor

        # Adaptação de terreno SIMPLIFICADA
        self.terrain_adaptation = {
            "normal": {"focus_valences": ["movimento_basico", "propulsao_basica"], "priority_boost": 1.2},
            "ramp_up": {"focus_valences": ["propulsao_basica", "estabilidade_postural"], "priority_boost": 1.3},
            "ramp_down": {"focus_valences": ["estabilidade_postural", "coordenacao_fundamental"], "priority_boost": 1.3},
            "uneven": {"focus_valences": ["coordenacao_fundamental", "estabilidade_postural"], "priority_boost": 1.2},
        }
        self.current_terrain = "normal"
        
        self._ensure_valence_trackers()

    def _initialize_valences(self) -> Dict[str, ValenceConfig]:
        """Apenas 4 valências essenciais - remove as avançadas"""
        return {
            "movimento_basico": ValenceConfig(
                name="movimento_basico",
                target_level=0.8,  
                metrics=["distance", "speed"],
                reward_components=["movement_priority", "basic_progress"],
                dependencies=[],
                activation_threshold=0.001,
                mastery_threshold=0.5
            ),

            "estabilidade_postural": ValenceConfig(
                name="estabilidade_postural",
                target_level=0.75,
                metrics=["roll", "pitch", "stability"],
                reward_components=["stability", "posture"],
                dependencies=["movimento_basico"],
                activation_threshold=0.01,
                mastery_threshold=0.4
            ),

            "propulsao_basica": ValenceConfig(
                name="propulsao_basica",
                target_level=0.7,
                metrics=["speed", "distance"],
                reward_components=["velocity", "propulsion"],
                dependencies=["movimento_basico"],
                activation_threshold=0.01,
                mastery_threshold=0.4
            ),

            "coordenacao_fundamental": ValenceConfig(
                name="coordenacao_fundamental",
                target_level=0.5,
                metrics=["alternating", "clearance_score"],
                reward_components=["coordination", "gait_pattern"],
                dependencies=["movimento_basico", "estabilidade_postural"],
                activation_threshold=0.05,
                mastery_threshold=0.4
            )
        }
    
    def set_current_terrain(self, terrain_type: str):
        if terrain_type in self.terrain_adaptation:
            self.current_terrain = terrain_type

    def _apply_terrain_adaptation(self, valence_levels: Dict[str, float]) -> Dict[str, float]:
        """Adaptação de terreno SIMPLIFICADA - apenas boost básico"""
        if self.current_terrain not in self.terrain_adaptation:
            return valence_levels

        terrain_config = self.terrain_adaptation[self.current_terrain]
        adapted_levels = valence_levels.copy()

        for valence_name in terrain_config.get("focus_valences", []):
            if valence_name in adapted_levels and adapted_levels[valence_name] > 0:
                boost = terrain_config.get("priority_boost", 1.0)
                adapted_levels[valence_name] = min(adapted_levels[valence_name] * boost, 1.0)

        return adapted_levels

    def _calculate_terrain_aware_weights(self, valence_levels: Dict[str, float]) -> Dict[str, float]:
        """ÚNICA versão - cálculo simplificado de pesos"""
        weights = {}
        total_weight = 0.0
        
        for valence_name in self.active_valences:
            current_level = valence_levels.get(valence_name, 0.0)
            config = self.valences[valence_name]
            
            # Cálculo SIMPLES baseado apenas no déficit
            deficit = max(0.1, config.target_level - current_level)  # Mínimo 0.1
            weights[valence_name] = deficit
            total_weight += deficit
        
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights

    def update_valences(self, episode_results: Dict) -> Dict[str, float]:
        """Atualização OTIMIZADA"""
        self.episode_count += 1

        # 1. Cálculo rápido das valências
        valence_levels = self._quick_valence_update(episode_results)
        
        # 2. Aplicação SIMPLES da adaptação ao terreno
        valence_levels = self._apply_terrain_adaptation(valence_levels)
        
        # 3. Atualização de pesos
        self.valence_weights = self._calculate_terrain_aware_weights(valence_levels)
        
        # 4. Atualizar trackers
        for valence_name, level in valence_levels.items():
            if valence_name in self.valence_performance:
                self.valence_performance[valence_name].update_level(level, self.episode_count)
                # Ativa valência se atingiu threshold
                if (level >= self.valences[valence_name].activation_threshold and 
                    valence_name not in self.active_valences):
                    self.active_valences.add(valence_name)
                    self.valence_performance[valence_name].state = ValenceState.LEARNING
        
        # 5. Progresso geral
        self.overall_progress = self._calculate_overall_progress(valence_levels)
        
        return valence_levels

    def _quick_valence_update(self, episode_results: Dict) -> Dict[str, float]:
        """Cálculo OTIMIZADO com early termination"""
        valence_levels = {}
        distance = max(episode_results.get("distance", 0), 0)
        
        # APENAS valências essenciais
        for valence_name in ["movimento_basico", "estabilidade_postural", "propulsao_basica", "coordenacao_fundamental"]:
            if self._should_calculate_valence(valence_name, distance, episode_results):
                valence_levels[valence_name] = self._calculate_valence_level(valence_name, episode_results)

        return valence_levels

    def _should_calculate_valence(self, valence_name: str, distance: float, results: Dict) -> bool:
        """Early termination OTIMIZADO"""
        conditions = {
            "movimento_basico": distance > 0.001,
            "estabilidade_postural": distance > 0.01,
            "propulsao_basica": distance > 0.02 and results.get("speed", 0) > 0,
            "coordenacao_fundamental": distance > 0.03 and results.get("alternating", False)
        }
        return conditions.get(valence_name, False)

    def _calculate_valence_level(self, valence_name: str, results: Dict) -> float:
        """Cache INTELIGENTE com chaves simplificadas"""
        # Chave baseada em buckets para melhor hit rate
        distance_bucket = int(results.get("distance", 0) * 20)  # Buckets de 5cm
        speed_bucket = int(results.get("speed", 0) * 10)        # Buckets de 0.1 m/s
        alternating = 1 if results.get("alternating", False) else 0
        
        cache_key = f"{valence_name}_{distance_bucket}_{speed_bucket}_{alternating}"

        if cache_key in self._cached_levels:
            self._cache_hits += 1
            return self._cached_levels[cache_key]

        self._cache_misses += 1
        level = self._compute_valence_level(valence_name, results)

        # Gerenciamento de cache
        self._cached_levels[cache_key] = level
        if len(self._cached_levels) > self._max_cache_size:
            oldest_key = next(iter(self._cached_levels))
            del self._cached_levels[oldest_key]

        return level

    def _compute_valence_level(self, valence_name: str, results: Dict) -> float:
        """Cálculo SIMPLIFICADO das valências"""
        try:
            distance = max(results.get("distance", 0), 0)
            speed = results.get("speed", 0)
            roll = abs(results.get("roll", 0))
            pitch = abs(results.get("pitch", 0))
            alternating = results.get("alternating", False)

            if valence_name == "movimento_basico":
                if distance <= 0: return 0.0
                return min(distance / 2.0, 0.8)  # Linear até 2m

            elif valence_name == "estabilidade_postural":
                instability = (roll + pitch) / 1.0
                stability = max(0.0, 1.0 - instability)
                # Bônus por movimento estável
                if distance > 0.1 and instability < 0.3:
                    stability = min(stability + 0.2, 0.9)
                return stability

            elif valence_name == "propulsao_basica":
                if speed <= 0 or distance < 0.02:
                    return 0.0
                return min(speed / 1.0, 0.7)  # Linear até 1.0 m/s

            elif valence_name == "coordenacao_fundamental":
                if distance < 0.03 or not alternating:
                    return 0.0
                base = 0.3 if alternating else 0.0
                # Bônus por distância coordenada
                if distance > 0.2:
                    base += 0.3
                if distance > 0.5:
                    base += 0.2
                return min(base, 0.8)

            return 0.0

        except Exception:
            return 0.0

    def _calculate_overall_progress(self, valence_levels: Dict[str, float]) -> float:
        """Progresso geral SIMPLIFICADO"""
        if not valence_levels:
            return 0.0

        total_progress = 0.0
        count = 0
        
        for valence_name, level in valence_levels.items():
            config = self.valences[valence_name]
            progress = min(level / config.target_level, 1.0)
            total_progress += progress
            count += 1

        return total_progress / count if count > 0 else 0.0

    # Mantém os métodos de status/get (já estão otimizados)
    def get_active_reward_components(self) -> List[str]:
        components = set()
        for valence_name in self.active_valences:
            components.update(self.valences[valence_name].reward_components)
        return list(components)

    def get_valence_status(self) -> Dict:
        status = {
            "overall_progress": self.overall_progress,
            "episode_count": self.episode_count,
            "active_valences": list(self.active_valences),
            "valence_details": {}
        }
        
        for valence_name, perf in self.valence_performance.items():
            config = self.valences[valence_name]
            status["valence_details"][valence_name] = {
                "current_level": perf.current_level,
                "target_level": config.target_level,
                "state": perf.state.value,
                "episodes_active": perf.episodes_active
            }
        
        return status

    def get_valence_weights_for_reward(self) -> Dict[str, float]:
        component_weights = {}
        for valence_name, valence_weight in self.valence_weights.items():
            for component in self.valences[valence_name].reward_components:
                component_weights[component] = component_weights.get(component, 0.0) + valence_weight
        
        total = sum(component_weights.values())
        if total > 0:
            component_weights = {k: v/total for k, v in component_weights.items()}
        
        return component_weights

    def _ensure_valence_trackers(self):
        for valence_name in self.valences.keys():
            if valence_name not in self.valence_performance:
                self.valence_performance[valence_name] = ValenceTracker(valence_name)

    def get_cache_stats(self) -> Dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self._cached_levels)
        }
    
    def get_terrain_status(self) -> Dict:
        """Retorna status do terreno atual"""
        if self.current_terrain not in self.terrain_adaptation:
            return {
                "current_terrain": self.current_terrain,
                "focus_valences": [],
                "priority_boost": 1.0,
                "activation_thresholds": {},
                "target_adjustments": {}, 
                "terrain_specific_metrics": []
            }

        terrain_config = self.terrain_adaptation[self.current_terrain]

        return {
            "current_terrain": self.current_terrain,
            "focus_valences": terrain_config.get("focus_valences", []),
            "priority_boost": terrain_config.get("priority_boost", 1.0),
            "activation_thresholds": terrain_config.get("activation_thresholds", {}),
            "target_adjustments": terrain_config.get("target_adjustments", {}),
            "terrain_specific_metrics": terrain_config.get("terrain_specific_metrics", [])
        }