# dpg_valence.py (versão unificada)
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time

class ValenceState(Enum):
    INACTIVE = "inative"
    LEARNING = "learning"
    CONSOLIDATING = "consolidating"
    MASTERED = "mastered"
    REGRESSING = "regressing"

@dataclass
class ValenceConfig:
    """Configuração de uma valência individual"""
    name: str
    target_level: float
    metrics: List[str]
    reward_components: List[str]
    dependencies: List[str]
    activation_threshold: float = 0.3
    mastery_threshold: float = 0.85
    regression_threshold: float = 0.6
    max_learning_rate: float = 0.1
    min_episodes: int = 10

class ValenceTracker:
    """Rastreamento de performance por valência"""
    
    def __init__(self, valence_name: str):
        self.valence_name = valence_name
        self.current_level = 0.0
        self.history = []  
        self.learning_rate = 0.0
        self.consistency_score = 0.5
        self.episodes_active = 0
        self.state = ValenceState. INACTIVE
    
    def update_level(self, new_level: float, episode: int):
        """Atualiza nível com cálculo de taxa de aprendizado"""
        old_level = self.current_level
        self.current_level = new_level
        self.history.append((episode, new_level))
        
        # Calcular taxa de aprendizado 
        if len(self.history) > 1:
            recent_growth = new_level - old_level
            self.learning_rate = 0.8 * self.learning_rate + 0.2 * recent_growth
        
        # Manter histórico limitado
        if len(self.history) > 100:
            self.history.pop(0)

class ValenceManager:
    """SISTEMA DE VALÊNCIAS OTIMIZADO"""
    
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config or {}
        
        # Sistema de valências
        self.valences = self._initialize_valences()
        self.valence_performance = {}
        self.active_valences = set()
        self.valence_weights = {}
        self._ensure_valence_trackers()
        
        # Estado do sistema
        self.episode_count = 0
        self.overall_progress = 0.0
        self.performance_history = []

        # Otimizações 
        self._performance_stagnation_count = 0
        self._last_overall_progress = 0.0
        self._cached_levels = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 200
        self._last_key_metrics = {}
        self._valence_change_threshold = 0.10  
        self._last_full_update = 0
        self._full_update_interval = 20

        # CONFIGURAÇÃO CROSS-TERRENO
        self.terrain_adaptation = {
            "low_friction": {"focus": ["estabilidade_postural", "coordenacao_fundamental"]},
            "ramp": {"focus": ["propulsao_basica", "estabilidade_postural"]}, 
            "uneven": {"focus": ["coordenacao_fundamental", "marcha_robusta"]},
            "normal": {"focus": ["movimento_basico", "propulsao_basica"]}
        }
        
        # Inicializar valence_performance
        for valence_name in self.valences.keys():
            self.valence_performance[valence_name] = ValenceTracker(valence_name)
        
    def _initialize_valences(self) -> Dict[str, ValenceConfig]:
        return {
            # FASE 1: Fundamentos → ativa quase imediatamente
            "movimento_basico": ValenceConfig(
                name="movimento_basico",
                target_level=0.7,
                metrics=["distance", "speed", "success", "positive_movement_rate"],
                reward_components=["movement_priority", "basic_progress"],
                dependencies=[],
                activation_threshold=0.01,   
                mastery_threshold=0.6,
                regression_threshold=0.3,
                max_learning_rate=0.4,
                min_episodes=3 
            ),

            # FASE 2: Estabilidade Postural → ativa logo após movimento
            "estabilidade_postural": ValenceConfig(
                name="estabilidade_postural",
                target_level=0.6,
                metrics=["roll", "pitch", "stability", "com_height_consistency", "lateral_stability"],
                reward_components=["stability", "posture", "dynamic_balance"],
                dependencies=["movimento_basico"],
                activation_threshold=0.05,   
                mastery_threshold=0.5,
                regression_threshold=0.25,
                min_episodes=5                
            ),

            # FASE 3: Propulsão Básica
            "propulsao_basica": ValenceConfig(
                name="propulsao_basica",
                target_level=0.6,
                metrics=["x_velocity", "velocity_consistency", "acceleration_smoothness", "distance"],
                reward_components=["velocity", "propulsion", "basic_progress"],
                dependencies=["movimento_basico"],
                activation_threshold=0.1,    
                mastery_threshold=0.5,
                regression_threshold=0.25,
                min_episodes=8                
            ),

            # FASE 4: Coordenação Fundamental → crítica para progresso
            "coordenacao_fundamental": ValenceConfig(
                name="coordenacao_fundamental",
                target_level=0.6,
                metrics=["alternating_consistency", "step_length_consistency", "gait_pattern_score", "clearance_score"],
                reward_components=["coordination", "rhythm", "gait_pattern"],
                dependencies=["movimento_basico", "estabilidade_postural"],                 activation_threshold=0.1,    
                mastery_threshold=0.5,
                regression_threshold=0.3,
                min_episodes=10              
            ),

            # FASE 5+: mantidas, mas com thresholds ligeiramente reduzidos
            "eficiencia_biomecanica": ValenceConfig(
                name="eficiencia_biomecanica",
                target_level=0.5,
                metrics=["energy_efficiency", "stride_efficiency", "propulsion_efficiency"],
                reward_components=["efficiency", "biomechanics", "smoothness"],
                dependencies=["coordenacao_fundamental"],
                activation_threshold=0.2,   
                mastery_threshold=0.5,
                regression_threshold=0.3,
                min_episodes=20
            ),
            "propulsao_avancada": ValenceConfig(
                name="propulsao_avancada",
                target_level=0.5,
                metrics=["x_velocity", "velocity_consistency", "acceleration_smoothness", "distance"],
                reward_components=["velocity", "propulsion", "smoothness"],
                dependencies=["eficiencia_biomecanica"],
                activation_threshold=0.25,   
                mastery_threshold=0.5,
                regression_threshold=0.35,
                min_episodes=25
            ),
            "marcha_robusta": ValenceConfig(
                name="marcha_robusta",
                target_level=0.5,
                metrics=["gait_robustness", "recovery_success", "speed_adaptation", "terrain_handling", "distance"],
                reward_components=["robustness", "adaptation", "recovery", "velocity", "propulsion"],
                dependencies=["propulsao_avancada", "coordenacao_fundamental"],
                activation_threshold=0.3,   
                mastery_threshold=0.5,
                regression_threshold=0.4,
                min_episodes=35
            )
        }

    def update_valences(self, episode_results: Dict) -> Dict[str, float]:
        """Atualização otimizada das valências — mais frequente no início"""
        self.episode_count += 1

        # ATUALIZAÇÃO A CADA EPISÓDIO se < 2000
        force_full_update = self.episode_count < 2000
        if force_full_update or (self.episode_count - self._last_full_update >= self._full_update_interval):
            valence_levels = self._quick_valence_update(episode_results)
            self._update_terrain_focused_states(valence_levels)
            self.valence_weights = self._calculate_terrain_aware_weights(valence_levels)
            self._last_full_update = self.episode_count
        else:
            valence_levels = {}

        # Sempre forçar estado mínimo de valências ativas se houver progresso
        distance = max(episode_results.get("distance", 0), 0)
        roll = abs(episode_results.get("roll", 0))
        pitch = abs(episode_results.get("pitch", 0))
        alternating = episode_results.get("alternating", False)

        # Força movimento_basico se andou >1 cm
        if distance > 0.01:
            self._ensure_valence_active("movimento_basico", min_level=min(distance / 1.5, 0.5))

        # Força estabilidade_postural se estável (>80% estável) e movimento >5cm
        stability_score = 1.0 - min((roll + pitch) / 1.0, 1.0)
        if distance > 0.05 and stability_score > 0.6:
            self._ensure_valence_active("estabilidade_postural", min_level=0.3)

        # Força coordenação se alternância OK + clearance detectada
        clearance_score = episode_results.get("clearance_score", 0.0)
        if alternating and clearance_score > 0.05 and distance > 0.15:
            self._ensure_valence_active("coordenacao_fundamental", min_level=0.25)

        # HARD GATING: impedir masterização precoce
        self._enforce_valence_dependencies()

        # Atualizar progresso geral
        self.overall_progress = self._calculate_overall_progress(valence_levels or {v: t.current_level for v, t in self.valence_performance.items()})
        return valence_levels or {v: t.current_level for v, t in self.valence_performance.items()}

    def _ensure_valence_active(self, valence_name: str, min_level: float = 0.1):
        """Garante que valência esteja ativa com nível mínimo"""
        if valence_name not in self.valence_performance:
            return
        tracker = self.valence_performance[valence_name]
        if tracker.state == ValenceState.INACTIVE or tracker.current_level < min_level:
            tracker.state = ValenceState.LEARNING
            tracker.current_level = max(tracker.current_level, min_level)
            self.active_valences.add(valence_name)
            tracker.episodes_active = max(tracker.episodes_active, 1)

    def _enforce_valence_dependencies(self):
        """Hard gating: uma valência só pode estar em CONSOLIDATING/MASTERED se dependências ≥ mastery_threshold"""
        for valence_name, tracker in self.valence_performance.items():
            config = self.valences[valence_name]
            dep_names = config.dependencies
            if not dep_names:
                continue
            
            dep_levels_ok = True
            for dep in dep_names:
                if dep not in self.valence_performance:
                    dep_levels_ok = False
                    break
                dep_tracker = self.valence_performance[dep]
                if dep_tracker.current_level < self.valences[dep].mastery_threshold:
                    dep_levels_ok = False
                    break
                
            if not dep_levels_ok and tracker.state in (ValenceState.CONSOLIDATING, ValenceState.MASTERED):
                tracker.state = ValenceState.LEARNING
                # Suavizar nível para evitar saltos
                tracker.current_level = min(tracker.current_level, config.mastery_threshold * 0.95)
    
    def _quick_valence_update(self, episode_results: Dict) -> Dict[str, float]:
        """Atualização rápida — mais sensível à coordenação funcional"""
        valence_levels = {}
        distance = max(episode_results.get("distance", 0), 0)

        # SEMPRE atualizar movimento_basico
        movement_level = self._calculate_valence_level("movimento_basico", episode_results)
        valence_levels["movimento_basico"] = movement_level

        # Forçar ativação se há movimento
        if distance > 0.01:
            tracker = self.valence_performance["movimento_basico"]
            if tracker.state == ValenceState.INACTIVE:
                tracker.state = ValenceState.LEARNING
                self.active_valences.add("movimento_basico")
                tracker.current_level = max(tracker.current_level, 0.05)

        # Estabilidade: ativa com d > 2cm e estabilidade mínima
        if distance > 0.02:
            stability_level = self._calculate_valence_level("estabilidade_postural", episode_results)
            valence_levels["estabilidade_postural"] = stability_level
            if stability_level > 0.02:  
                self.active_valences.add("estabilidade_postural")

        # Propulsão: com d > 5cm
        if distance > 0.05:
            propulsion_level = self._calculate_valence_level("propulsao_basica", episode_results)
            valence_levels["propulsao_basica"] = propulsion_level
            if propulsion_level > 0.05:  
                self.active_valences.add("propulsao_basica")

        # Coordenação: só com alternating + clearance + d > 10cm
        alternating = episode_results.get("alternating", False)
        clearance_score = episode_results.get("clearance_score", 0.0)
        if alternating and clearance_score > 0.05 and distance > 0.10:
            coordination_level = self._calculate_valence_level("coordenacao_fundamental", episode_results)
            valence_levels["coordenacao_fundamental"] = coordination_level
            if coordination_level > 0.1:  
                self.active_valences.add("coordenacao_fundamental")

        # Completar com zeros
        for valence in ["movimento_basico", "estabilidade_postural", "propulsao_basica", "coordenacao_fundamental"]:
            if valence not in valence_levels:
                valence_levels[valence] = 0.0

        return valence_levels

    def _update_terrain_focused_states(self, valence_levels: Dict[str, float]):
        """Ativa valências baseadas no terreno atual"""
        terrain = getattr(self, '_current_terrain', 'normal')
        focus_valences = self.terrain_adaptation.get(terrain, {}).get("focus", [])
        
        for valence_name, current_level in valence_levels.items():
            current_level = valence_levels.get(valence_name, 0.0)
            perf = self.valence_performance.get(valence_name)
        
            if perf is None:
                continue
                
            if current_level > 0.01:
                perf.state = ValenceState.LEARNING 
                self.active_valences.add(valence_name)
                perf.current_level = current_level
                perf.episodes_active += 1

            # Atualizar nível mesmo se já estava ativa
            elif valence_name in self.active_valences:
                perf.current_level = current_level

    def _ensure_valence_trackers(self):
        """Garante que todos os trackers existam"""
        for valence_name in self.valences.keys():
            if valence_name not in self.valence_performance:
                self.valence_performance[valence_name] = ValenceTracker(valence_name)
            
    def _calculate_terrain_aware_weights(self, valence_levels: Dict[str, float]) -> Dict[str, float]:
        """Pesos que priorizam valências relevantes para o terreno atual"""
        terrain = getattr(self, '_current_terrain', 'normal')
        focus_valences = self.terrain_adaptation.get(terrain, {}).get("focus", [])
        
        weights = {}
        total_weight = 0.0
        
        for valence_name in self.active_valences:
            current_level = valence_levels[valence_name]
            config = self.valences[valence_name]
            
            # BÔNUS para valências focadas no terreno atual
            terrain_bonus = 2.0 if valence_name in focus_valences else 1.0
            
            deficit = max(0, config.target_level - current_level)
            weight = deficit * terrain_bonus
            
            weights[valence_name] = weight
            total_weight += weight
        
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights

    def _calculate_valence_level(self, valence_name: str, results: Dict) -> float:
        """Cache mais agressivo - mesmo cálculo"""
        # CHAVE MAIS ESTÁVEL (menos sensível a pequenas variações)
        metrics_hash = self._calculate_stable_metrics_hash(results)
        cache_key = f"{valence_name}_{metrics_hash}"

        if cache_key in self._cached_levels:
            self._cache_hits += 1
            return self._cached_levels[cache_key]

        self._cache_misses += 1

        # CÁLCULO ORIGINAL (zero alteração)
        level = self._compute_valence_level(valence_name, results)

        # CACHE MAIOR
        self._cached_levels[cache_key] = level
        if len(self._cached_levels) > self._max_cache_size:
            # Remove o mais antigo (FIFO simples)
            oldest_key = next(iter(self._cached_levels))
            del self._cached_levels[oldest_key]

        return level

    def _calculate_stable_metrics_hash(self, results: Dict) -> int:
        """Hash menos sensível a pequenas variações"""
        try:
            # Arredondar métricas para reduzir variação
            stable_metrics = {}
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    # Arredondar para reduzir pequenas variações
                    if abs(v) < 10:
                        stable_metrics[k] = round(v, 2)  
                    else:
                        stable_metrics[k] = round(v, 1)  

            return hash(frozenset(stable_metrics.items()))
        except:
            return hash(str(results))

    def _compute_valence_level(self, valence_name: str, results: Dict) -> float:
        try:
            raw_distance = results.get("distance", 0)
            if not isinstance(raw_distance, (int, float)):
                distance = 0.0
            else:
                distance = float(raw_distance)
    
            roll = abs(results.get("roll", 0))
            pitch = abs(results.get("pitch", 0))
            alternating = results.get("alternating", False)
            clearance_score = results.get("clearance_score", 0.0)
            gait_score = results.get("gait_pattern_score", 0.0)
            velocity = results.get("speed", 0)
    
            # MOVIMENTO BÁSICO (mantido quase igual)
            if valence_name == "movimento_basico":
                if distance <= 0:
                    return 0.0
                if distance > 2.0:  return 0.9
                if distance > 1.0:  return 0.7
                if distance > 0.5:  return 0.5
                if distance > 0.3:  return 0.35
                if distance > 0.15: return 0.2
                if distance > 0.05: return 0.1
                return 0.05
    
            # ESTABILIDADE POSTURAL — mais sensível
            elif valence_name == "estabilidade_postural":
                # Penaliza mais roll/pitch combinados
                instability = (roll + pitch) / 1.0  
                stability_score = max(0.0, 1.0 - instability)
                # Bônus por movimento estável
                if distance > 0.1 and instability < 0.4:
                    stability_score = min(stability_score + 0.3, 0.9)
                return stability_score
    
            # PROPULSÃO BÁSICA — condicional à estabilidade
            elif valence_name == "propulsao_basica":
                if velocity <= 0 or distance < 0.05:
                    return 0.0
                # Só conta se estiver razoavelmente estável
                instability = (roll + pitch) / 1.0
                if instability > 0.7:
                    return 0.0
                if velocity > 1.0: return 0.85
                if velocity > 0.6: return 0.65
                if velocity > 0.3: return 0.4
                if velocity > 0.1: return 0.15
                return 0.05
    
            # COORDENAÇÃO FUNDAMENTAL
            elif valence_name == "coordenacao_fundamental":
                if distance < 0.1:
                    return 0.0
    
                base = 0.1
    
                # Alternância: obrigatória
                if alternating:
                    base += 0.25
                else:
                    return 0.0  
    
                # Clearance funcional (pé levanta >6cm)
                if clearance_score > 0.06:
                    base += 0.2
    
                # Gait pattern
                if gait_score > 0.5:
                    base += 0.15
    
                # Bônus por distância com coordenação
                if distance > 0.5:
                    base += 0.1
    
                return min(base, 0.9)
    
            # Demais valências (mantidas com pequena otimização)
            elif valence_name == "eficiencia_biomecanica":
                efficiency = results.get("propulsion_efficiency", 0.5)
                coord_level = self.valence_performance["coordenacao_fundamental"].current_level
                if coord_level < 0.3:
                    return 0.0
                return max(0.0, min(efficiency * 0.8, 0.8))
    
            elif valence_name == "propulsao_avancada":
                eficiencia_level = self.valence_performance["eficiencia_biomecanica"].current_level
                if eficiencia_level < 0.4 or velocity < 0.5:
                    return 0.0
                if velocity > 2.0: return 0.9
                if velocity > 1.5: return 0.7
                if velocity > 1.0: return 0.5
                return 0.2
    
            elif valence_name == "marcha_robusta":
                propulsao_level = self.valence_performance["propulsao_avancada"].current_level
                coord_level = self.valence_performance["coordenacao_fundamental"].current_level
                if propulsao_level < 0.5 or coord_level < 0.4:
                    return 0.0
                if distance > 3.0: return 0.9
                if distance > 2.0: return 0.7
                if distance > 1.0: return 0.4
                return 0.1
    
            return 0.0
    
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de {valence_name}: {e}")
            return 0.0

    def _calculate_overall_progress(self, valence_levels: Dict[str, float]) -> float:
        """Calcula progresso geral considerando todas as valências"""
        if not valence_levels:
            return 0.0

        total_weighted = 0.0
        total_weights = 0.0

        for valence_name, level in valence_levels.items():
            config = self.valences[valence_name]
            weight = 1.0
            if valence_name in ["estabilidade_dinamica", "propulsao_eficiente"]:
                weight = 1.5  
            normalized_progress = min(level / config.target_level, 1.0)
            total_weighted += normalized_progress * weight
            total_weights += weight

        return total_weighted / total_weights if total_weights > 0 else 0.0

    def get_active_reward_components(self) -> List[str]:
        """Retorna componentes de recompensa das valências ativas"""
        components = set()
        
        for valence_name in self.active_valences:
            valence_config = self.valences[valence_name]
            components.update(valence_config.reward_components)
        
        return list(components)

    def get_valence_status(self) -> Dict:
        """Retorna status detalhado de todas as valências"""
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
                "episodes_active": perf.episodes_active,
                "learning_rate": perf.learning_rate,
                "consistency": perf.consistency_score,
                "dependencies": config.dependencies
            }
        
        return status

    def get_valence_weights_for_reward(self) -> Dict[str, float]:
        """Retorna pesos formatados para o sistema de recompensa"""
        component_weights = {}
        
        for valence_name, valence_weight in self.valence_weights.items():
            valence_config = self.valences[valence_name]
            
            for component in valence_config.reward_components:
                if component not in component_weights:
                    component_weights[component] = 0.0
                component_weights[component] += valence_weight
        
        # Normalizar
        total = sum(component_weights.values())
        if total > 0:
            component_weights = {k: v/total for k, v in component_weights.items()}
        
        return component_weights

    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache para monitoramento"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self._cached_levels)
        }