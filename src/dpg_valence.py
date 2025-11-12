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
        self.mastery_callback = None
        self._ensure_valence_trackers()
        
        # Estado do sistema
        self.episode_count = 0
        self.overall_progress = 0.0
        self.performance_history = []
        self.irl_system = LightValenceIRL(logger)
        
        # Otimizações
        self._last_irl_update = 0
        self._irl_update_interval = 100  
        self._performance_stagnation_count = 0
        self._last_overall_progress = 0.0
        self._cached_levels = {}
        self._cache_hits = 0
        self._cache_misses = 0
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
            # FASE 1: Fundamentos 
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
                min_episodes=5              
            ),

            # FASE 2: Estabilidade Postural 
            "estabilidade_postural": ValenceConfig(
                name="estabilidade_postural", 
                target_level=0.6,           
                metrics=["roll", "pitch", "stability", "com_height_consistency", "lateral_stability"],
                reward_components=["stability", "posture", "dynamic_balance"],
                dependencies=["movimento_basico"],
                activation_threshold=0.1,   
                mastery_threshold=0.5,
                regression_threshold=0.25,      
                min_episodes=10
            ),

            # FASE 3: Propulsão Básica
            "propulsao_basica": ValenceConfig(
                name="propulsao_basica",
                target_level=0.6,           
                metrics=["x_velocity", "velocity_consistency", "acceleration_smoothness", "distance"],
                reward_components=["velocity", "propulsion", "basic_progress"],
                dependencies=["movimento_basico"],
                activation_threshold=0.2,  
                mastery_threshold=0.5,
                regression_threshold=0.25,      
                min_episodes=15
            ),

            # FASE 4: Coordenação Fundamental
            "coordenacao_fundamental": ValenceConfig(
                name="coordenacao_fundamental",
                target_level=0.6,
                metrics=["alternating_consistency", "step_length_consistency", "gait_pattern_score"],
                reward_components=["coordination", "rhythm", "gait_pattern"],
                dependencies=["movimento_basico"],  
                activation_threshold=0.25,  
                mastery_threshold=0.5,
                regression_threshold=0.3,
                min_episodes=15  
            ),

            # FASE 5: Eficiência Biomecânica
            "eficiencia_biomecanica": ValenceConfig(
                name="eficiencia_biomecanica",
                target_level=0.5,
                metrics=["energy_efficiency", "stride_efficiency", "propulsion_efficiency"],
                reward_components=["efficiency", "biomechanics", "smoothness"],
                dependencies=["coordenacao_fundamental"],
                activation_threshold=0.3,
                mastery_threshold=0.5,
                regression_threshold=0.3,
                min_episodes=30
            ),

            # FASE 6: Propulsão Avançada
            "propulsao_avancada": ValenceConfig(
                name="propulsao_avancada",
                target_level=0.5,
                metrics=["x_velocity", "velocity_consistency", "acceleration_smoothness", "distance"],
                reward_components=["velocity", "propulsion", "smoothness"],
                dependencies=["eficiencia_biomecanica"],
                activation_threshold=0.35,
                mastery_threshold=0.5,
                regression_threshold=0.35,
                min_episodes=35
            ),

            # FASE 7: Marcha Robusta
            "marcha_robusta": ValenceConfig(
                name="marcha_robusta", 
                target_level=0.5,
                metrics=["gait_robustness", "recovery_success", "speed_adaptation", 
                        "terrain_handling", "distance", "velocity_consistency"],
                reward_components=["robustness", "adaptation", "recovery", "velocity", "propulsion"],
                dependencies=["propulsao_avancada", "coordenacao_fundamental"],
                activation_threshold=0.4,
                mastery_threshold=0.5,
                regression_threshold=0.4,
                min_episodes=50
            )
        }

    def update_valences(self, episode_results: Dict) -> Dict[str, float]:
        """Atualização otimizada das valências"""
        self.episode_count += 1
        
        # ATUALIZAÇÃO RÁPIDA: Apenas valências relevantes
        valence_levels = self._quick_valence_update(episode_results)
        
        # ATUALIZAÇÃO DE ESTADOS com foco no terreno
        self._update_terrain_focused_states(valence_levels)
        
        # PESOS ADAPTATIVOS ao terreno
        self.valence_weights = self._calculate_terrain_aware_weights(valence_levels)
        self.overall_progress = self._calculate_overall_progress(valence_levels)
        
        return self.valence_weights, 1.0

    def _quick_valence_update(self, episode_results: Dict) -> Dict[str, float]:
        """Atualização rápida das valências mais importantes"""
        valence_levels = {}
        
        # SEMPRE atualizar movimento_basico 
        valence_levels["movimento_basico"] = self._calculate_valence_level(
            "movimento_basico", episode_results
        )
        
        # Atualizar outras valências baseadas em dependências simples
        movimento_level = valence_levels["movimento_basico"]
        
        if movimento_level > 0.01:
            valence_levels["estabilidade_postural"] = self._calculate_valence_level(
                "estabilidade_postural", episode_results
            )
            
        if movimento_level > 0.15:  
            valence_levels["propulsao_basica"] = self._calculate_valence_level(
                "propulsao_basica", episode_results
            )
            
        if movimento_level > 0.2:
            valence_levels["coordenacao_fundamental"] = self._calculate_valence_level(
                "coordenacao_fundamental", episode_results 
            )
            
        basic_valences = ["movimento_basico", "estabilidade_postural", "propulsao_basica", "coordenacao_fundamental"]
        for valence in basic_valences:
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
            
            # Ativação MAIS RÁPIDA para valências focadas no terreno
            if valence_name in focus_valences:
                activation_threshold = 0.02
            else:
                activation_threshold = 0.05
                
            if current_level > activation_threshold:
                perf.state = ValenceState.LEARNING 
                self.active_valences.add(valence_name)
                perf.current_level = current_level
                perf.episodes_active += 1
            else:
                perf.state = ValenceState.INACTIVE
                self.active_valences.discard(valence_name)

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
        """Cálculo otimizado de nível de valência com cache"""
        metrics_hash = self._calculate_metrics_hash(results)
        cache_key = f"{valence_name}_{metrics_hash}"
        
        if cache_key in self._cached_levels:
            self._cache_hits += 1
            return self._cached_levels[cache_key]
        
        self._cache_misses += 1
        level = self._compute_valence_level(valence_name, results)
        self._cached_levels[cache_key] = level
        
        if len(self._cached_levels) > 100:
            oldest_key = next(iter(self._cached_levels))
            del self._cached_levels[oldest_key]
            
        return level

    def _compute_valence_level(self, valence_name: str, results: Dict) -> float:
        """Cálculo real do nível da valência"""
        try:
            raw_distance = results.get("distance", 0)
            if not isinstance(raw_distance, (int, float)):
                distance = 0.0
            else:
                distance = float(raw_distance)

            # MOVIMENTO BÁSICO
            if valence_name == "movimento_basico":
                if distance <= 0:
                    return 0.0 
                if distance > 2.0: return 0.9
                if distance > 1.5: return 0.8
                if distance > 1.0: return 0.7
                if distance > 0.7: return 0.6
                if distance > 0.5: return 0.5
                if distance > 0.4: return 0.4
                if distance > 0.3: return 0.3
                if distance > 0.2: return 0.2
                if distance > 0.1: return 0.15
                return 0.1

            # ESTABILIDADE POSTURAL
            elif valence_name == "estabilidade_postural":
                roll = abs(results.get("roll", 0))
                pitch = abs(results.get("pitch", 0))
                stability = 1.0 - min((roll + pitch) / 1.0, 1.0)
                if distance > 0.05:  
                    return stability * 0.9
                return 0.0

            # PROPULSÃO BÁSICA 
            elif valence_name == "propulsao_basica":
                velocity = results.get("speed", 0)
                if velocity <= 0:
                    return 0.0
                if distance > 0.1:  
                    if velocity > 1.2: return 0.9
                    if velocity > 0.8: return 0.7
                    if velocity > 0.5: return 0.5
                    if velocity > 0.3: return 0.3
                    if velocity > 0.1: return 0.15
                return 0.0

            # COORDENAÇÃO FUNDAMENTAL
            elif valence_name == "coordenacao_fundamental":
                alternating = results.get("alternating", False)

                if distance > 0.15: 
                    base_level = 0.4
                    if alternating:
                        base_level += 0.4
                    gait_score = results.get("gait_pattern_score", 0)
                    if gait_score > 0.5:
                        base_level += 0.2
                    return min(base_level, 0.9)
                return 0.0

            # EFICIÊNCIA BIOMECÂNICA
            elif valence_name == "eficiencia_biomecanica":
                efficiency = results.get("propulsion_efficiency", 0.5)
                coordenacao_level = self.valence_performance["coordenacao_fundamental"].current_level
                if coordenacao_level < 0.4:
                    return 0.0
                return efficiency * 0.8

            # PROPULSÃO AVANÇADA  
            elif valence_name == "propulsao_avancada":
                velocity = results.get("speed", 0)
                eficiencia_level = self.valence_performance["eficiencia_biomecanica"].current_level
                if eficiencia_level < 0.5:
                    return 0.0
                if velocity > 2.0: return 0.9
                if velocity > 1.5: return 0.7
                if velocity > 1.0: return 0.5
                return 0.2

            # MARCHA ROBUSTA
            elif valence_name == "marcha_robusta":
                distance = max(results.get("distance", 0), 0)
                propulsao_level = self.valence_performance["propulsao_avancada"].current_level
                coordenacao_level = self.valence_performance["coordenacao_fundamental"].current_level
                if propulsao_level < 0.6 or coordenacao_level < 0.5:
                    return 0.0
                if distance > 3.0: return 0.9
                if distance > 2.0: return 0.7
                if distance > 1.0: return 0.5
                return 0.2

            return 0.0
    
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de {valence_name}: {e}")
            return 0.0

    def _calculate_metrics_hash(self, results: Dict) -> int:
        """Calcula hash eficiente para cache"""
        try:
            numeric_items = {k: v for k, v in results.items() 
                           if isinstance(v, (int, float))}
            return hash(frozenset(numeric_items.items()))
        except:
            return hash(str(results))

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

    def get_irl_weights(self):
        """Retorna pesos IRL atuais"""
        try:
            if hasattr(self, 'irl_weights') and self.irl_weights:
                return self.irl_weights
            else:
                self.irl_weights = {
                    'progress': 0.3,
                    'stability': 0.4, 
                    'efficiency': 0.2,
                    'coordination': 0.1
                }
                return self.irl_weights
        except Exception as e:
            self.logger.warning(f"Erro ao obter pesos IRL: {e}")
            return {'progress': 0.3, 'stability': 0.4, 'efficiency': 0.2, 'coordination': 0.1}

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

# Manter a classe LightValenceIRL 
class LightValenceIRL:
    """Sistema IRL leve integrado com valências"""
    
    def __init__(self, logger):
        self.logger = logger
        self.demonstration_buffer = []
        self.learned_weights = {}
        self.sample_count = 0
        self._active = False
        
    def should_activate(self, valence_status):
        """Ativa quando valências base estão consolidadas"""
        try:
            if self.sample_count < 50:
                return False
                
            base_valences = ['estabilidade_dinamica', 'propulsao_eficiente']
            struggling_valences = 0
            
            for v in base_valences:
                if v in valence_status['valence_details']:
                    details = valence_status['valence_details'][v]
                    if (details['state'] == 'regressing' or 
                        details['current_level'] < 0.3 or
                        (details['learning_rate'] < 0.005 and details['current_level'] < 0.5)):
                        struggling_valences += 1
            
            if struggling_valences >= 1:
                self._active = True
                return True
                
            overall_progress = valence_status.get('overall_progress', 0)
            if self.sample_count > 100 and overall_progress < 0.3:
                self._active = True
                return True
                
            return self._active
            
        except Exception as e:
            self.logger.warning(f"Erro ao verificar ativação IRL: {e}")
            return False
    
    def collect_demonstration(self, episode_results, valence_status):
        """Coleta demonstrações com critérios liberais"""
        quality = self._calculate_demo_quality(episode_results)
        
        if quality > 0.3:  
            self.demonstration_buffer.append({
                'results': episode_results,
                'quality': quality,
                'valence_status': valence_status,
                'timestamp': self.sample_count
            })
            self.sample_count += 1
            
            if len(self.demonstration_buffer) > 300:  
                self.demonstration_buffer.pop(0)
    
    def _calculate_demo_quality(self, results):
        """Critérios de qualidade mais restritivos"""
        quality = 0.0
        
        if results.get('success', False):
            quality += 0.5 
        elif max(results.get('distance', 0), 0) > 1.0:
            quality += 0.4 
        elif results.get('speed', 0) > 0.5:
            quality += 0.3 
            
        roll = abs(results.get('roll', 0))
        pitch = abs(results.get('pitch', 0))
        stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
        if stability > 0.6:
            quality += 0.3
            
        return min(quality, 1.0)
    
    def get_irl_weights(self, valence_status):
        """Retorna pesos IRL se disponíveis e relevantes"""
        if len(self.demonstration_buffer) < 10:
            return self.learned_weights
            
        high_quality_demos = [d for d in self.demonstration_buffer if d['quality'] > 0.6]
        if not high_quality_demos:
            return self.learned_weights
            
        new_weights = self._learn_simple_weights(high_quality_demos)
        if new_weights:
            self.learned_weights = new_weights
            
        return self.learned_weights
    
    def _learn_simple_weights(self, demonstrations):
        """Aprendizado simples de pesos IRL"""
        feature_scores = {
            'progress': 0.0,
            'stability': 0.0, 
            'efficiency': 0.0,
            'coordination': 0.0
        }
        feature_counts = {k: 0 for k in feature_scores.keys()}
        
        for demo in demonstrations:
            results = demo['results']
            
            if max(results.get('distance', 0), 0) > 0.5:
                feature_scores['progress'] += results['distance']
                feature_counts['progress'] += 1
                
            roll = abs(results.get('roll', 0))
            pitch = abs(results.get('pitch', 0))
            stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
            feature_scores['stability'] += stability
            feature_counts['stability'] += 1
            
            if results.get('propulsion_efficiency', 0) > 0:
                feature_scores['efficiency'] += results['propulsion_efficiency']
                feature_counts['efficiency'] += 1
                
            if results.get('alternating', False):
                feature_scores['coordination'] += 1.0
            feature_counts['coordination'] += 1
        
        for feature in feature_scores:
            if feature_counts[feature] > 0:
                feature_scores[feature] /= feature_counts[feature]
        
        total = sum(feature_scores.values())
        if total > 0:
            return {k: v/total for k, v in feature_scores.items()}
        return {}