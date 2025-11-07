# dpg_valence.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class ValenceState(Enum):
    INACTIVE = "inactive"
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
        self.state = ValenceState.INACTIVE
    
    def update_level(self, new_level: float, episode: int):
        """Atualiza nível com cálculo de taxa de aprendizado"""
        old_level = self.current_level
        self.current_level = new_level
        self.history.append((episode, new_level))
        
        # Calcular taxa de aprendizado (suavizada)
        if len(self.history) > 1:
            recent_growth = new_level - old_level
            self.learning_rate = 0.8 * self.learning_rate + 0.2 * recent_growth
        
        # Manter histórico limitado
        if len(self.history) > 100:
            self.history.pop(0)
    
    def calculate_consistency(self) -> float:
        """Calcula consistência baseada na variância recente"""
        if len(self.history) < 8:
            return 0.3
            
        recent_levels = [level for _, level in self.history[-8:]]
        variance = np.std(recent_levels)
        consistency = 1.0 - min(variance * 3.0, 1.0)
        self.consistency_score = consistency
        return consistency


class Mission:
    """Missão com proteção contra progresso negativo"""
    
    def __init__(self, valence_name: str, target_improvement: float, duration_episodes: int):
        self.valence_name = valence_name
        self.target_improvement = target_improvement
        self.duration_episodes = duration_episodes
        self.start_level = 0.0
        self.episodes_remaining = duration_episodes
        self.completed = False
        self.bonus_multiplier = 1.5
        self.max_level_achieved = 0.0  # ✅ Rastrear máximo alcançado
    
    def update(self, current_level: float) -> float:
        """Atualiza missão com proteção contra regressão"""
        if self.completed or self.episodes_remaining <= 0:
            return 1.0
            
        self.episodes_remaining -= 1
        
        # ✅ USAR o máximo alcançado para evitar regressão no cálculo
        self.max_level_achieved = max(self.max_level_achieved, current_level)
        improvement = self.max_level_achieved - self.start_level
        
        progress_ratio = improvement / self.target_improvement if self.target_improvement > 0 else 0
        
        if progress_ratio >= 1.0:
            self.completed = True
            return self.bonus_multiplier
        elif self.episodes_remaining <= 0:
            return 0.8  # Penalidade leve por falha
            
        return 1.0


class ValenceManager:
    """SISTEMA DE VALÊNCIAS ADAPTATIVAS"""
    
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config or {}
        
        # Sistema de valências
        self.valences = self._initialize_valences()
        self.valence_performance = {}
        self.active_valences = set()
        self.valence_weights = {}
        self.mastery_callback = None
        
        # Sistema de missões
        self.current_missions = []
        self.mission_history = []
        
        # Estado do sistema
        self.episode_count = 0
        self.overall_progress = 0.0
        self.performance_history = []
        self.irl_system = LightValenceIRL(logger)
        
        # Inicializar valence_performance
        for valence_name in self.valences.keys():
            self.valence_performance[valence_name] = ValenceTracker(valence_name)
        
    def _initialize_valences(self) -> Dict[str, ValenceConfig]:
        return {
            # FASE 1: Fundamentos - thresholds MAIS BAIXOS
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
                activation_threshold=0.15,   
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

            # FASE 4: Coordenação Fundamental (Episódios 1500-5000)
            "coordenacao_fundamental": ValenceConfig(
                name="coordenacao_fundamental",
                target_level=0.6,
                metrics=["alternating_consistency", "step_length_consistency", "gait_pattern_score"],
                reward_components=["coordination", "rhythm", "gait_pattern"],
                dependencies=["propulsao_basica", "estabilidade_postural"],
                activation_threshold=0.25,
                mastery_threshold=0.5,
                regression_threshold=0.3,
                min_episodes=25
            ),

            # FASE 5: Eficiência Biomecânica (Episódios 2500-6000)
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

            # FASE 6: Propulsão Avançada (Episódios 3500-7000)
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

            # FASE 7: Marcha Robusta (Episódios 5000-10000)
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
        """Atualiza valências com DEBUG EXPANDIDO"""
        self.episode_count += 1
        
        # ✅ DEBUG CRÍTICO: Verificar dados de entrada
        distance_raw = episode_results.get("distance", "N/A")
        self.logger.info(f"🔍 VALENCE_INPUT - Ep {self.episode_count}, Distance: {distance_raw}, Type: {type(distance_raw)}")
        
        self.performance_history.append(episode_results)
        self.update_irl_system(episode_results)
        
        valence_levels = {}
        for valence_name, valence_config in self.valences.items():
            level = self._calculate_valence_level(valence_name, episode_results)
            valence_levels[valence_name] = level
            
            # ✅ DEBUG expandido para movimento_basico
            if valence_name == "movimento_basico" and self.episode_count % 20 == 0:
                self.logger.info(f"🔍 movimento_basico RESULT: dist={episode_results.get('distance', 'N/A')} -> level={level:.1%}")
            
            perf = self.valence_performance[valence_name]
            old_level = perf.current_level
            perf.update_level(level, self.episode_count)
            perf.episodes_active += 1 if valence_name in self.active_valences else 0
            
            # ✅ Log de mudanças significativas
            if abs(level - old_level) > 0.1 and self.episode_count % 50 == 0:
                self.logger.info(f"📈 VALENCE_UPDATE: {valence_name} {old_level:.1%} -> {level:.1%}")
        
        self._update_valence_states(valence_levels)
        self.valence_weights = self._calculate_valence_weights(valence_levels)
        mission_bonus = self._update_missions(valence_levels)
        self.overall_progress = self._calculate_overall_progress(valence_levels)
        
        return self.valence_weights, mission_bonus
    
    def update_irl_system(self, episode_results):
        """Atualiza sistema IRL com resultados do episódio"""
        valence_status = self.get_valence_status()
        self.irl_system.collect_demonstration(episode_results, valence_status)
        
        if self.episode_count % 50 == 0:
            new_irl_weights = self.irl_system.get_irl_weights(valence_status)
            if hasattr(self, 'irl_weights'):
                if new_irl_weights: 
                    self.irl_weights.update(new_irl_weights)
            else:
                self.irl_weights = {
                'progress': 0.3,
                'stability': 0.4, 
                'efficiency': 0.2,
                'coordination': 0.1
                }
                if new_irl_weights:  
                    self.irl_weights.update(new_irl_weights)

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
    
    def _calculate_valence_level(self, valence_name: str, results: Dict) -> float:
        """VALÊNCIA - COM SMOOTHING para evitar oscilações"""
        try:
            raw_distance = results.get("distance", 0)
            if not isinstance(raw_distance, (int, float)):
                distance = 0.0
            else:
                distance = float(raw_distance)

            # IGNORAR distâncias negativas para movimento_basico
            if valence_name == "movimento_basico" and distance < 0:
                return 0.0

        except:
            distance = 0.0

        # MOVIMENTO BÁSICO 
        if valence_name == "movimento_basico":
            if distance <= 0:
                return 0.0 

            # ESCALA MAIS SUAVE E CONSISTENTE
            if distance > 1.0: return 0.95
            if distance > 0.8: return 0.85
            if distance > 0.6: return 0.75
            if distance > 0.5: return 0.65
            if distance > 0.4: return 0.55
            if distance > 0.3: return 0.45
            if distance > 0.25: return 0.35
            if distance > 0.2: return 0.25
            if distance > 0.15: return 0.18
            if distance > 0.1: return 0.12
            if distance > 0.05: return 0.07
            if distance > 0.02: return 0.04
            return 0.02

        # ESTABILIDADE POSTURAL
        elif valence_name == "estabilidade_postural":
            roll = abs(results.get("roll", 0))
            pitch = abs(results.get("pitch", 0))
            stability = 1.0 - min((roll + pitch) / 1.0, 1.0)

            # ATIVAÇÃO MAIS FÁCIL
            movimento_level = self.valence_performance["movimento_basico"].current_level
            if movimento_level >= 0.2:  
                return stability * 0.9
            return 0.0

        # PROPULSÃO BÁSICA
        elif valence_name == "propulsao_basica":
            velocity = results.get("speed", 0)
            if velocity <= 0:
                return 0.0

            # ATIVAÇÃO MAIS FÁCIL
            movimento_level = self.valence_performance["movimento_basico"].current_level
            if movimento_level >= 0.25:  
                # Escala de velocidade MAIS PERMISSIVA
                if velocity > 1.2: return 0.9
                if velocity > 0.8: return 0.7
                if velocity > 0.5: return 0.5
                if velocity > 0.3: return 0.3
                if velocity > 0.1: return 0.15
            return 0.0

        # COORDENAÇÃO FUNDAMENTAL - BASEADA EM PADRÃO ALTERNADO
        elif valence_name == "coordenacao_fundamental":
            alternating = results.get("alternating", False)
            movimento_level = self.valence_performance["movimento_basico"].current_level
            propulsao_level = self.valence_performance["propulsao_basica"].current_level

            if movimento_level < 0.5 or propulsao_level < 0.3:
                return 0.0

            base_level = 0.3
            if alternating:
                base_level += 0.4
            if results.get("gait_pattern_score", 0) > 0.6:
                base_level += 0.2

            return min(base_level, 0.9)

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
          
    def _update_valence_states(self, valence_levels: Dict[str, float]):
        """Ativação das valências"""
        for valence_name, current_level in valence_levels.items():
            perf = self.valence_performance[valence_name]
            config = self.valences[valence_name]

            # MOVIMENTO BÁSICO - SEMPRE ATIVO
            if valence_name == "movimento_basico":
                perf.state = ValenceState.LEARNING
                self.active_valences.add(valence_name)
                continue

            # Para outras valências, verificar dependências 
            dependencies_met = True
            for dep in config.dependencies:
                if dep in self.valence_performance:
                    dep_level = self.valence_performance[dep].current_level
                    if dep_level < 0.15:  
                        dependencies_met = False
                        break
                else:
                    dependencies_met = False
                    break

            if not dependencies_met:
                perf.state = ValenceState.INACTIVE
                self.active_valences.discard(valence_name)
                continue

            # SE dependências atendidas, ATIVAR imediatamente
            perf.state = ValenceState.LEARNING
            self.active_valences.add(valence_name)
    
    def _calculate_valence_weights(self, valence_levels: Dict[str, float]) -> Dict[str, float]:
        """Calcula pesos dinâmicos baseados em déficit de performance"""
        weights = {}
        total_weight = 0.0
        for valence_name in self.active_valences:
            config = self.valences[valence_name]
            current_level = valence_levels[valence_name]
            perf = self.valence_performance[valence_name]
            deficit = max(0, config.target_level - current_level)
            state_multiplier = {
                ValenceState.LEARNING: 2.0,
                ValenceState.REGRESSING: 1.8,
                ValenceState.CONSOLIDATING: 1.2,
                ValenceState.MASTERED: 0.3,
                ValenceState.INACTIVE: 0.0
            }.get(perf.state, 1.0)
            weight = deficit * state_multiplier
            if perf.consistency_score < 0.5:
                weight *= 1.5
            weights[valence_name] = weight
            total_weight += weight
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _update_missions(self, valence_levels: Dict[str, float]) -> float:
        """Atualiza missões sem progresso negativo"""
        total_bonus = 1.0

        # Atualizar missões existentes
        for mission in self.current_missions[:]:
            current_level = valence_levels.get(mission.valence_name, 0.0)
            bonus = mission.update(current_level)
            total_bonus *= bonus

            # Evitar progresso negativo nas missões
            improvement = current_level - mission.start_level
            if improvement < 0:
                # Se regrediu, ajusta o start_level para evitar progresso negativo
                mission.start_level = current_level

            if mission.completed or mission.episodes_remaining <= 0:
                self.current_missions.remove(mission)
                self.mission_history.append(mission)

        # Gerar novas missões se necessário
        if len(self.current_missions) < 2:  
            new_mission = self._generate_mission(valence_levels)
            if new_mission:
                self.current_missions.append(new_mission)

        return total_bonus
    
    def _generate_mission(self, valence_levels: Dict[str, float]) -> Optional[Mission]:
        """MISSÕES COM METAS REALISTAS E PROGRESSIVAS"""
        episode = self.episode_count

        # MISSÃO MOVIMENTO BÁSICO - sempre que estiver baixo
        movimento_level = valence_levels.get('movimento_basico', 0)
        if movimento_level < 0.4:  
            if not any(m.valence_name == 'movimento_basico' for m in self.current_missions):
                mission = Mission('movimento_basico', 0.4, 150)  
                mission.start_level = movimento_level
                mission.bonus_multiplier = 2.0
                return mission

        # MISSÃO ESTABILIDADE - quando movimento_basico estiver OK
        elif episode < 2000 and movimento_level >= 0.3:
            estabilidade_level = valence_levels.get('estabilidade_postural', 0)
            if estabilidade_level < 0.4:
                if not any(m.valence_name == 'estabilidade_postural' for m in self.current_missions):
                    mission = Mission('estabilidade_postural', 0.4, 200)
                    mission.start_level = estabilidade_level
                    mission.bonus_multiplier = 1.8
                    return mission

        # MISSÃO PROPULSÃO - quando estabilidade estiver OK
        elif episode < 5000:
            estabilidade_level = valence_levels.get('estabilidade_postural', 0)
            if estabilidade_level >= 0.3:
                propulsao_level = valence_levels.get('propulsao_basica', 0)
                if propulsao_level < 0.5:
                    if not any(m.valence_name == 'propulsao_basica' for m in self.current_missions):
                        mission = Mission('propulsao_basica', 0.5, 250)
                        mission.start_level = propulsao_level
                        mission.bonus_multiplier = 1.6
                        return mission

        return None
    
    def _calculate_overall_progress(self, valence_levels: Dict[str, float]) -> float:
        """Calcula progresso geral considerando todas as valências"""
        if not valence_levels:
            return 0.0

        total_weighted = 0.0
        total_weights = 0.0

        for valence_name, level in valence_levels.items():
            config = self.valences[valence_name]
            perf = self.valence_performance[valence_name]
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
            "current_missions": [
                {
                    "valence": mission.valence_name,
                    "progress": f"{mission.target_improvement:.2f}",
                    "episodes_remaining": mission.episodes_remaining
                }
                for mission in self.current_missions
            ],
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
        # Converter pesos de valência em pesos de componentes
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
    
    def _debug_valence_calculation(self, episode_results):
        """DEBUG AVANÇADO para movimento positivo/negativo"""
        distance = episode_results.get('distance', 0)
        movimento_level = self.valence_manager._calculate_valence_level("movimento_basico", episode_results)

        # Log detalhado a cada 20 episódios
        if self.episode_count % 20 == 0:
            status = "🟢 POSITIVO" if distance > 0 else "🔴 NEGATIVO" if distance < 0 else "⚪ ZERO"
            self.logger.info(f"🔍 DEBUG EPISODE {self.episode_count}: {status} distance={distance:.3f}, movimento_level={movimento_level:.3f}")

        # FORÇAR ativação se houver movimento positivo
        if distance > 0.01 and "movimento_basico" not in self.valence_manager.active_valences:
            self.logger.warning(f"🚨 MOVIMENTO POSITIVO DETECTADO mas valência inativa! distance={distance:.3f}")
            # Forçar ativação
            self.valence_manager.active_valences.add("movimento_basico")
            self.valence_manager.valence_performance["movimento_basico"].state = ValenceState.LEARNING

        # ALERTA CRÍTICO para movimento negativo
        if distance < -0.1:
            self.logger.error(f"🚨🚨 MOVIMENTO NEGATIVO CRÍTICO: {distance:.3f} - PENALIDADE APLICADA")
    
class OptimizedValenceManager(ValenceManager):
    def __init__(self, logger, config=None):
        super().__init__(logger, config)
        self._last_irl_update = 0
        self._irl_update_interval = 100  
        self._performance_stagnation_count = 0
        self._last_overall_progress = 0.0
        self._normalization_cache = {}
        self._last_metrics_hash = None
        self._cached_levels = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_key_metrics = {}
        self._valence_change_threshold = 0.10  
        self._last_full_update = 0
        self._full_update_interval = 20
        
    def update_valences(self, episode_results: Dict) -> Dict[str, float]:
        """Update OTIMIZADO - apenas valências que mudaram significativamente"""
        self.episode_count += 1
        
        if not self._should_recalculate_valences(episode_results):
            return self._cached_valence_weights, 1.0  
        
        valences_to_update = self._get_valences_that_matter(episode_results)
        valence_levels = {}
        
        for valence_name in valences_to_update:
            level = self._calculate_valence_level(valence_name, episode_results)
            valence_levels[valence_name] = level
            
            perf = self.valence_performance[valence_name]
            perf.update_level(level, self.episode_count)
            perf.episodes_active += 1 if valence_name in self.active_valences else 0
        
        self._update_valence_states(valence_levels)
        self.valence_weights = self._calculate_valence_weights(valence_levels)
        mission_bonus = self._update_missions(valence_levels)
        
        if self._should_update_irl():
            self.update_irl_system(episode_results)
            self._last_irl_update = self.episode_count
        
        self.overall_progress = self._calculate_overall_progress(valence_levels)
        self._cached_valence_weights = self.valence_weights
        self._last_full_update = self.episode_count
        
        return self.valence_weights, mission_bonus
    
    def _should_update_irl(self) -> bool:
        """Determina se IRL deve ser executado baseado em critérios inteligentes"""
        if self.episode_count - self._last_irl_update < self._irl_update_interval:
            return False
            
        progress_change = abs(self.overall_progress - self._last_overall_progress)
        if progress_change < 0.02: 
            self._performance_stagnation_count += 1
        else:
            self._performance_stagnation_count = 0
            
        self._last_overall_progress = self.overall_progress
        
        valence_status = self.get_valence_status()
        mastered_count = sum(1 for d in valence_status['valence_details'].values() 
                           if d['state'] == 'mastered')
        
        return (self._performance_stagnation_count >= 20 or 
                mastered_count != getattr(self, '_last_mastered_count', 0))
    
    def _calculate_valence_level(self, valence_name: str, results: Dict) -> float:
        metrics_hash = self._calculate_metrics_hash(results)
        cache_key = f"{valence_name}_{metrics_hash}"
        
        if cache_key in self._cached_levels:
            self._cache_hits += 1
            return self._cached_levels[cache_key]
        
        self._cache_misses += 1
        level = super()._calculate_valence_level(valence_name, results)
        self._cached_levels[cache_key] = level
        
        if len(self._cached_levels) > 100:
            oldest_key = next(iter(self._cached_levels))
            del self._cached_levels[oldest_key]
            
        return level
    
    def _calculate_metrics_hash(self, results: Dict) -> int:
        """Calcula hash eficiente para cache"""
        try:
            numeric_items = {k: v for k, v in results.items() 
                           if isinstance(v, (int, float))}
            return hash(frozenset(numeric_items.items()))
        except:
            return hash(str(results))
    
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
    
    def _should_recalculate_valences(self, episode_results) -> bool:
        """Decide se precisa recalcular valências (80% menos cálculos)"""
        key_metrics = ['distance', 'speed', 'roll', 'pitch', 'success']
        current_values = {metric: episode_results.get(metric, 0) for metric in key_metrics}
        
        if not self._last_key_metrics:
            self._last_key_metrics = current_values
            return True
        
        if (self.episode_count - self._last_full_update) >= self._full_update_interval:
            self._last_key_metrics = current_values
            return True
        
        significant_change = False
        for metric in key_metrics:
            current_val = current_values[metric]
            last_val = self._last_key_metrics.get(metric, current_val)
            
            if metric == 'success':
                if current_val != last_val:
                    significant_change = True
                    break
            else:
                change_pct = abs(current_val - last_val) / max(abs(last_val), 0.1)
                if change_pct > self._valence_change_threshold:
                    significant_change = True
                    break
        
        self._last_key_metrics = current_values
        return significant_change
    
    def _get_valences_that_matter(self, episode_results) -> List[str]:
        """Retorna apenas valências que precisam ser atualizadas"""
        valences_to_update = set()
        
        valences_to_update.update(self.active_valences)
        
        for valence_name, config in self.valences.items():
            if valence_name not in self.active_valences:
                # Verificar se dependências foram atendidas recentemente
                dependencies_met = all(
                    self.valence_performance[dep].current_level >= config.activation_threshold
                    for dep in config.dependencies
                )
                if dependencies_met and valence_name not in self.active_valences:
                    valences_to_update.add(valence_name)
        
        for valence_name, perf in self.valence_performance.items():
            if perf.state == ValenceState.REGRESSING:
                valences_to_update.add(valence_name)
        
        return list(valences_to_update)
    

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
            if self.sample_count < 50:  # Aumentamos o mínimo de amostras
                return False
                
            base_valences = ['estabilidade_dinamica', 'propulsao_eficiente']
            struggling_valences = 0
            
            for v in base_valences:
                if v in valence_status['valence_details']:
                    details = valence_status['valence_details'][v]
                    if (details['state'] == 'regressing' or 
                        details['current_level'] < 0.3 or  # Limite mais baixo
                        (details['learning_rate'] < 0.005 and details['current_level'] < 0.5)): # Taxa de aprendizado mais baixa
                        struggling_valences += 1
            
            if struggling_valences >= 1:
                self._active = True
                return True
                
            overall_progress = valence_status.get('overall_progress', 0)
            if self.sample_count > 100 and overall_progress < 0.3:  # Mais amostras e progresso mais baixo
                self._active = True
                return True
                
            return self._active
            
        except Exception as e:
            self.logger.warning(f"Erro ao verificar ativação IRL: {e}")
            return False
    
    def collect_demonstration(self, episode_results, valence_status):
        """COLETA MAIS DEMONSTRAÇÕES - Critérios mais liberais"""
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
        """CRITÉRIOS DE QUALIDADE MAIS RESTRITIVOS"""
        quality = 0.0
        
        # Progresso básico já é suficiente
        if results.get('success', False):
            quality += 0.5 
        elif max(results.get('distance', 0), 0) > 1.0:  # Aumentamos a distância mínima
            quality += 0.4 
        elif results.get('speed', 0) > 0.5:  # Aumentamos a velocidade mínima
            quality += 0.3 
            
        # Estabilidade mínima
        roll = abs(results.get('roll', 0))
        pitch = abs(results.get('pitch', 0))
        stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
        if stability > 0.6:  # Aumentamos a estabilidade mínima
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
            
            # Progresso
            if max(results.get('distance', 0), 0) > 0.5:
                feature_scores['progress'] += results['distance']
                feature_counts['progress'] += 1
                
            # Estabilidade
            roll = abs(results.get('roll', 0))
            pitch = abs(results.get('pitch', 0))
            stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
            feature_scores['stability'] += stability
            feature_counts['stability'] += 1
            
            # Eficiência
            if results.get('propulsion_efficiency', 0) > 0:
                feature_scores['efficiency'] += results['propulsion_efficiency']
                feature_counts['efficiency'] += 1
                
            # Coordenação
            if results.get('alternating', False):
                feature_scores['coordination'] += 1.0
            feature_counts['coordination'] += 1
        
        # Calcular médias
        for feature in feature_scores:
            if feature_counts[feature] > 0:
                feature_scores[feature] /= feature_counts[feature]
        
        # Normalizar
        total = sum(feature_scores.values())
        if total > 0:
            return {k: v/total for k, v in feature_scores.items()}
        return {}