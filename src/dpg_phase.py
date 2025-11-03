# dpg_phase.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PhaseTransitionResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure" 
    REGRESSION = "regression"
    STAGNATION = "stagnation"
    VALIDATION_REQUIRED = "validation_required"


class ValidationMode(Enum):
    LIGHT = "light"
    MEDIUM = "medium" 
    HEAVY = "heavy"


@dataclass
class SubPhaseConfig:
    """Configuração de uma sub-fase dentro de um grupo"""
    name: str
    target_speed: float
    enabled_components: List[str]
    component_weights: Dict[str, float]
    min_episodes: int
    transition_conditions: Dict[str, float]
    focus_skills: List[str]
    complexity_level: int
    validation_requirements: Dict
    validation_requirements: Dict[str, float] = None


@dataclass
class PhaseGroup:
    """Grupo de fases com validação adaptativa"""
    name: str
    group_level: int
    base_complexity: int
    focus_areas: List[str]
    adaptive_config: Dict
    sub_phases: List[SubPhaseConfig]


class AdaptiveValidator:
    """Especialista em Validação Distribuída"""
    
    def __init__(self, logger):
        self.logger = logger
        self.validation_modes = {
            "light": {"frequency": 0.3, "depth": "basic", "components": ["success_rate", "distance"]},
            "medium": {"frequency": 0.6, "depth": "standard", "components": ["success_rate", "distance", "stability", "consistency"]},
            "heavy": {"frequency": 0.9, "depth": "comprehensive", "components": ["success_rate", "distance", "stability", "consistency", "efficiency", "coordination"]}
        }
        
        self.validation_triggers = {
            "performance_drop": {"threshold": 0.2, "action": "light_validation"},
            "stagnation": {"episodes": 15, "action": "medium_validation"},
            "regression": {"count": 2, "action": "heavy_validation"},
            "phase_transition": {"always": True, "action": "comprehensive"}
        }
        
        self.validation_history = []
    
    def should_validate(self, group_level: int, performance_trend: float, 
                  consecutive_failures: int, episodes_in_phase: int) -> Tuple[bool, str]:
        """Decide se deve validar baseado em múltiplos gatilhos """

        # Gatilho: Queda de performance 
        if performance_trend < -0.5:  
            return True, "performance_drop"

        # Gatilho: Estagnação 
        if episodes_in_phase > 25 and performance_trend < 0.02:  
            return True, "stagnation"

        # Gatilho: Muitas falhas consecutivas 
        if consecutive_failures > 25:  
            return True, "regression"

        # Gatilho periódico baseado no grupo
        validation_freq = self._get_validation_frequency(group_level)* 2
        if len(self.validation_history) > 0:
            last_validation = self.validation_history[-1]
            episodes_since_last = episodes_in_phase - last_validation.get('episode', 0)
            if episodes_since_last > validation_freq:
                return True, "periodic"

        return False, ""
    
    def distributed_validation(self, group_config: PhaseGroup, sub_phase_config: SubPhaseConfig,
                             performance_history: List[Dict]) -> Dict[str, bool]:
        """Executa validação distribuída por componentes"""
        validations = {}
        
        # Validação básica sempre
        validations["basic"] = self._basic_validation(performance_history)
        
        # Validações condicionais baseadas no grupo e sub-fase
        if group_config.group_level >= 1:
            validations["stability"] = self._stability_validation(performance_history)
            validations["progress"] = self._progress_validation(performance_history, sub_phase_config)
        
        if group_config.group_level >= 2:
            validations["coordination"] = self._coordination_validation(performance_history)
            validations["consistency"] = self._consistency_validation(performance_history)
        
        if group_config.group_level >= 3:
            validations["efficiency"] = self._efficiency_validation(performance_history)
            validations["adaptation"] = self._adaptation_validation(performance_history)
        
        # Registrar validação
        self.validation_history.append({
            'episode': len(performance_history),
            'group': group_config.group_level,
            'sub_phase': sub_phase_config.name,
            'results': validations,
            'timestamp': np.datetime64('now')
        })
        
        return validations
    
    def _basic_validation(self, performance_history: List[Dict]) -> bool:
        """Validação básica de sucesso e progresso """
        if len(performance_history) < 3:  
            return True  

        recent = performance_history[-3:]  
        success_rate = sum(1 for r in recent if r.get("success", False)) / len(recent)
        avg_distance = np.mean([r.get("distance", 0) for r in recent])

        # Condições mais flexíveis para fase inicial
        return success_rate > 0.1 and avg_distance > 0.05
    
    def _stability_validation(self, performance_history: List[Dict]) -> bool:
        """Validação de estabilidade """
        if len(performance_history) < 3:  
            return True

        recent = performance_history[-3:]  
        avg_roll = np.mean([abs(r.get("roll", 0)) for r in recent])
        return avg_roll < 1.5
    
    def _progress_validation(self, performance_history: List[Dict], sub_phase_config: SubPhaseConfig) -> bool:
        """Validação de progresso específica da sub-fase"""
        if len(performance_history) < 3:  
            return True

        recent = performance_history[-3:]  
        avg_distance = np.mean([r.get("distance", 0) for r in recent])
        target_distance = sub_phase_config.transition_conditions.get("min_avg_distance", 0.5)

        return avg_distance >= target_distance * 0.1
    
    def _coordination_validation(self, performance_history: List[Dict]) -> bool:
        """Validação de coordenação"""
        if len(performance_history) < 8:
            return False
        
        recent = performance_history[-8:]
        alternations = sum(1 for r in recent if r.get("alternating", False))
        return alternations / len(recent) > 0.4
    
    def _consistency_validation(self, performance_history: List[Dict]) -> bool:
        """Validação de consistência"""
        if len(performance_history) < 5:
            return False
        
        recent = performance_history[-5:]
        distances = [r.get("distance", 0) for r in recent]
        consistency = np.std(distances) / (np.mean(distances) + 1e-8)
        return consistency < 0.5
    
    def _efficiency_validation(self, performance_history: List[Dict]) -> bool:
        """Validação de eficiência energética"""
        if len(performance_history) < 5:
            return False
        
        recent = performance_history[-5:]
        efficiencies = []
        for r in recent:
            distance = r.get("distance", 0)
            steps = r.get("steps", 1)
            if steps > 0:
                efficiencies.append(distance / steps)
        
        return np.mean(efficiencies) > 0.05 if efficiencies else False
    
    def _adaptation_validation(self, performance_history: List[Dict]) -> bool:
        """Validação de capacidade de adaptação"""
        if len(performance_history) < 10:
            return False
        
        # Verificar recuperação de instabilidades
        recovery_events = 0
        for i in range(1, min(5, len(performance_history))):
            prev_roll = abs(performance_history[-i-1].get("roll", 0))
            curr_roll = abs(performance_history[-i].get("roll", 0))
            if prev_roll > 0.5 and curr_roll < 0.3:
                recovery_events += 1
        
        return recovery_events >= 2
    
    def _get_validation_frequency(self, group_level: int) -> int:
        """Retorna frequência de validação baseada no grupo"""
        frequencies = {1: 20, 2: 15, 3: 10}  # episódios entre validações
        return frequencies.get(group_level, 15)
    
    def get_validation_status(self) -> Dict:
        """Retorna status do sistema de validação"""
        return {
            "total_validations": len(self.validation_history),
            "last_validation": self.validation_history[-1] if self.validation_history else None,
            "validation_modes": list(self.validation_modes.keys()),
            "active_triggers": list(self.validation_triggers.keys())
        }


class PhaseManager:
    """
    ESPECIALISTA EM FASES com Validação Distribuída
    """
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.current_group = 0
        self.current_sub_phase = 0
        self.episodes_in_sub_phase = 0
        self.groups = self._initialize_groups()
        self.performance_history = []
        
        # Sistema de validação
        self.validator = AdaptiveValidator(logger)
        self.last_validation_episode = 0
        self.validation_required = False
        
        # Histórico de performance
        self.performance_history = []
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.stagnation_counter = 0
        
        # Estado de transição
        self.transition_active = False
        self.transition_episodes = 0
    
    def _initialize_groups(self):
        """Inicializa os 3 grupos principais com sub-fases dinâmicas"""
        return [
            # GRUPO 1: Fundação
            PhaseGroup(
                name="fundacao",
                group_level=1,
                base_complexity=1,
                focus_areas=["estabilidade", "postura", "progresso_basico"],
                adaptive_config={
                    "validation_frequency": "medium",
                    "irl_requirement": "light", 
                    "critic_complexity": "basic",
                    "learning_preservation": "high"
                },
                sub_phases=[
                    # Sub-fase 1.1: Estabilidade Inicial
                    SubPhaseConfig(
                        name="estabilidade_inicial",
                        target_speed=0.3,
                        enabled_components=["stability", "basic_progress", "posture"],
                        component_weights={"stability": 0.6, "basic_progress": 0.3, "posture": 0.1},
                        min_episodes=8,
                        transition_conditions={
                            "min_success_rate": 0.4,
                            "min_avg_distance": 0.2,
                            "max_avg_roll": 1.0,
                            "min_avg_steps": 3,
                            "min_positive_movement_rate": 0.6
                        },
                        focus_skills=["basic_balance", "postural_stability"],
                        validation_requirements={},
                        complexity_level=1
                    ),
                    # Sub-fase 1.2: Controle Postural
                    SubPhaseConfig(
                        name="controle_postural",
                        target_speed=0.5,
                        enabled_components=["stability", "basic_progress", "posture", "velocity"],
                        component_weights={"stability": 0.4, "basic_progress": 0.3, "posture": 0.2, "velocity": 0.1},
                        min_episodes=10,
                        transition_conditions={
                            "min_success_rate": 0.5,
                            "min_avg_distance": 0.5,
                            "max_avg_roll": 0.8,
                            "min_avg_speed": 0.1
                        },
                        focus_skills=["postural_stability", "gait_initiation"],
                        validation_requirements={},
                        complexity_level=1
                    )
                ]
            ),
            
            # GRUPO 2: Desenvolvimento
            PhaseGroup(
                name="desenvolvimento",
                group_level=2,
                base_complexity=2,
                focus_areas=["coordenação", "velocidade", "eficiência"],
                adaptive_config={
                    "validation_frequency": "high",
                    "irl_requirement": "standard",
                    "critic_complexity": "standard", 
                    "learning_preservation": "medium"
                },
                sub_phases=[
                    # Sub-fase 2.1: Marcha Básica
                    SubPhaseConfig(
                        name="marcha_basica",
                        target_speed=0.8,
                        enabled_components=["velocity", "stability", "phase_angles", "propulsion"],
                        component_weights={"velocity": 0.3, "stability": 0.25, "phase_angles": 0.25, "propulsion": 0.2},
                        min_episodes=12,
                        transition_conditions={
                            "min_success_rate": 0.5,
                            "min_avg_distance": 0.8,
                            "max_avg_roll": 0.7,
                            "min_avg_speed": 0.2,
                            "min_alternating_score": 0.3
                        },
                        focus_skills=["step_consistency", "dynamic_balance"],
                        validation_requirements={},
                        complexity_level=2
                    ),
                    # Sub-fase 2.2: Coordenação Rítmica
                    SubPhaseConfig(
                        name="coordenacao_ritmica", 
                        target_speed=1.2,
                        enabled_components=["velocity", "stability", "propulsion", "coordination"],
                        component_weights={"velocity": 0.25, "stability": 0.2, "propulsion": 0.25, "coordination": 0.3},
                        min_episodes=15,
                        transition_conditions={
                            "min_success_rate": 0.6,
                            "min_avg_distance": 1.5,
                            "max_avg_roll": 0.6,
                            "min_avg_speed": 0.4,
                            "min_gait_coordination": 0.4
                        },
                        focus_skills=["gait_coordination", "rhythmic_pattern"],
                        validation_requirements={},
                        complexity_level=2
                    )
                ]
            ),
            
            # GRUPO 3: Domínio
            PhaseGroup(
                name="domínio",
                group_level=3, 
                base_complexity=3,
                focus_areas=["otimização", "adaptação", "robustez"],
                adaptive_config={
                    "validation_frequency": "light",
                    "irl_requirement": "advanced",
                    "critic_complexity": "advanced",
                    "learning_preservation": "low"
                },
                sub_phases=[
                    # Sub-fase 3.1: Marcha Eficiente
                    SubPhaseConfig(
                        name="marcha_eficiente",
                        target_speed=1.5,
                        enabled_components=["velocity", "stability", "propulsion", "clearance", "coordination", "efficiency"],
                        component_weights={"velocity": 0.2, "stability": 0.2, "propulsion": 0.2, "clearance": 0.15, "coordination": 0.15, "efficiency": 0.1},
                        min_episodes=18,
                        transition_conditions={
                            "min_success_rate": 0.7,
                            "min_avg_distance": 2.0,
                            "max_avg_roll": 0.5,
                            "min_avg_speed": 0.6,
                            "min_gait_coordination": 0.5
                        },
                        focus_skills=["energy_efficiency", "gait_coordination"],
                        validation_requirements={},
                        complexity_level=3
                    ),
                    # Sub-fase 3.2: Otimização Final
                    SubPhaseConfig(
                        name="otimizacao_final",
                        target_speed=2.0,
                        enabled_components=["velocity", "stability", "propulsion", "clearance", "coordination", "efficiency", "adaptation"],
                        component_weights={"velocity": 0.15, "stability": 0.15, "propulsion": 0.15, "clearance": 0.15, "coordination": 0.15, "efficiency": 0.15, "adaptation": 0.1},
                        min_episodes=20,
                        transition_conditions={
                            "min_success_rate": 0.8,
                            "min_avg_distance": 3.0,
                            "max_avg_roll": 0.3,
                            "min_avg_speed": 0.8,
                            "min_gait_coordination": 0.6,
                            "min_energy_efficiency": 0.5
                        },
                        focus_skills=["energy_efficiency", "speed_maintenance", "adaptation"],
                        validation_requirements={},
                        complexity_level=3
                    )
                ]
            )
        ]
    
    def update_phase(self, episode_results: Dict) -> PhaseTransitionResult:
        """Atualiza progressão com validação distribuída"""
        if self.transition_active:
            return self._update_transition_progress()
        
        self.episodes_in_sub_phase += 1

        essential_data = {
            "distance": episode_results.get("distance", 0),
            "success": episode_results.get("success", False),
            "steps": episode_results.get("steps", 0),
            "roll": episode_results.get("roll", 0),
            "speed": episode_results.get("speed", 0),
            "left_contact": episode_results.get("left_contact", False),
            "right_contact": episode_results.get("right_contact", False),
            "alternating": episode_results.get("alternating", False)
        }
        
        self.performance_history.append(essential_data)
        
        # Manter histórico limitado
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Atualizar contadores
        self._update_success_counters(episode_results)
        
        should_regress = False
        if self.episodes_in_sub_phase >= 20:  
            should_regress = self._should_regress()

        # Verificar avanço primeiro, depois regressão
        should_advance = self._should_advance_sub_phase()
        if should_advance:
            return self._advance_to_next_sub_phase()

        # Verificar outras transições
        if self._should_advance_group():
            return self._start_group_advancement()

        if should_regress:
            return self._start_regression()
        
        return PhaseTransitionResult.SUCCESS
    
    def execute_validation(self) -> Dict[str, bool]:
        """Executa validação distribuída completa"""
        self.last_validation_episode = self.episodes_in_sub_phase
        
        return self.validator.distributed_validation(
            self.current_group_config,
            self.current_sub_phase_config,
            self.performance_history
        )
    
    def _calculate_performance_trend(self) -> float:
        """Calcula tendência de performance para gatilhos de validação"""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent = self.performance_history[-5:]
        distances = [r.get("distance", 0) for r in recent]
        
        if len(distances) >= 3:
            return (distances[-1] - distances[0]) / max(distances[0], 0.1)

        return 0.0

    def _should_advance_sub_phase(self) -> bool:
        """Verifica se pode avançar para próxima sub-fase"""
        current_sub_phase = self.current_sub_phase_config

        if self.episodes_in_sub_phase < current_sub_phase.min_episodes * 2:
            return False

        if self.current_sub_phase >= len(self.current_group_config.sub_phases) - 1:
            return False

        return self._check_all_conditions(current_sub_phase.transition_conditions)

    def _check_all_conditions(self, conditions: Dict) -> bool:
        """Verifica todas as condições de transição - COM LOGS DETALHADOS"""

        # Sucess rate
        success_rate = self._calculate_success_rate()
        min_success = conditions.get("min_success_rate", 0.3)
        success_ok = success_rate >= min_success
        if not success_ok:
            return False

        # Average distance
        avg_distance = self._calculate_avg_distance()
        min_distance = conditions.get("min_avg_distance", 0.5)
        distance_ok = avg_distance >= min_distance
        if not distance_ok:
            return False

        # Average roll
        avg_roll = self._calculate_avg_roll()
        max_roll = conditions.get("max_avg_roll", 1.0)
        roll_ok = avg_roll <= max_roll
        if not roll_ok:
            return False

        # Average steps
        avg_steps = self._calculate_avg_steps()
        min_steps = conditions.get("min_avg_steps", 3)
        steps_ok = avg_steps >= min_steps
        if not steps_ok:
            return False

        # Positive movement rate
        positive_rate = self._calculate_positive_movement_rate()
        min_positive = conditions.get("min_positive_movement_rate", 0.6)
        positive_ok = positive_rate >= min_positive
        if not positive_ok:
            return False

        # Additional conditions that might be present
        if "min_avg_speed" in conditions:
            avg_speed = self._calculate_avg_speed()
            min_speed = conditions["min_avg_speed"]
            speed_ok = avg_speed >= min_speed
            if not speed_ok:
                return False

        if "min_alternating_score" in conditions:
            alternating_score = self._calculate_alternating_score()
            min_alternating = conditions["min_alternating_score"]
            alternating_ok = alternating_score >= min_alternating
            if not alternating_ok:
                return False

        return True
    
    def _should_advance_group(self) -> bool:
        """Verifica se pode avançar para próximo grupo"""
        # Só avança grupo se completou todas sub-fases
        if self.current_sub_phase < len(self.current_group_config.sub_phases) - 1:
            return False
        
        current_sub_phase = self.current_sub_phase_config
        
        if self.episodes_in_sub_phase < current_sub_phase.min_episodes:
            return False
        
        conditions = current_sub_phase.transition_conditions
        
        success_rate = self._calculate_success_rate()
        if success_rate < conditions.get("min_success_rate", 0.3) + 0.1:  
            return False
        
        if not self._check_advanced_consistency():
            return False
        
        if self.current_group >= len(self.groups) - 1:
            return False
        
        return True
    
    def _advance_to_next_sub_phase(self) -> PhaseTransitionResult:
        """Avança para próxima sub-fase dentro do mesmo grupo"""
        old_sub_phase = self.current_sub_phase
        self.current_sub_phase += 1
        self.episodes_in_sub_phase = 0
        
        return PhaseTransitionResult.SUCCESS
    
    def _start_group_advancement(self) -> PhaseTransitionResult:
        """Inicia transição para próximo grupo"""
        self.transition_active = True
        self.transition_episodes = 0
        self.transition_total_episodes = 12  
        
        return PhaseTransitionResult.SUCCESS
    
    def _should_regress(self) -> bool:
        """Verifica se precisa regredir (sub-fase ou grupo)"""
        regression_thresholds = {
            1: {"max_failures": 80, "min_success_rate": 0.05, "stagnation_episodes": 20},
            2: {"max_failures": 60, "min_success_rate": 0.1, "stagnation_episodes": 15},
            3: {"max_failures": 45, "min_success_rate": 0.15, "stagnation_episodes": 12}
        }
        
        thresholds = regression_thresholds.get(self.current_group_config.group_level, regression_thresholds[1])
        
        if self.episodes_in_sub_phase < 30: 
            return False
        
        if self.consecutive_failures > thresholds["max_failures"]:
            return True

        if self.stagnation_counter > thresholds["stagnation_episodes"]:
            return True

        success_rate = self._calculate_success_rate()
        if success_rate < thresholds["min_success_rate"] and self.episodes_in_sub_phase > 45:
            return True

        return False
    
    def _start_regression(self) -> PhaseTransitionResult:
        """Inicia processo de regressão"""
        if self.current_sub_phase > 0:
            old_phase = self.current_sub_phase
            self.current_sub_phase -= 1
            self.episodes_in_sub_phase = 0
            self.consecutive_failures = 0
            self.stagnation_counter = 0
            return PhaseTransitionResult.REGRESSION
        elif self.current_group > 0 and self.episodes_in_sub_phase > 50:
            self.transition_active = True
            self.transition_episodes = 0
            self.transition_total_episodes = 10
            return PhaseTransitionResult.REGRESSION
        else:
            self.consecutive_failures = 0  
            self.stagnation_counter = 0
            return PhaseTransitionResult.FAILURE
    
    def _update_transition_progress(self) -> PhaseTransitionResult:
        """Atualiza progresso da transição entre grupos"""
        self.transition_episodes += 1
        
        if self.transition_episodes >= self.transition_total_episodes:
            return self._complete_group_transition()
        
        return PhaseTransitionResult.SUCCESS
    
    def _complete_group_transition(self) -> PhaseTransitionResult:
        """Completa transição entre grupos"""
        old_group = self.current_group
        is_regression = (self._calculate_success_rate() < 0.3)
        
        if is_regression:
            self.current_group = max(0, self.current_group - 1)
            self.regression_count += 1
            result = PhaseTransitionResult.REGRESSION
        else:
            self.current_group += 1
            result = PhaseTransitionResult.SUCCESS
        
        # Sempre começa na primeira sub-fase do grupo
        self.current_sub_phase = 0
        self.episodes_in_sub_phase = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.stagnation_counter = 0
        self.transition_active = False
        self.transition_episodes = 0
        
        old_group_name = self.groups[old_group].name
        new_group_name = self.current_group_config.name
        
        if is_regression:
            self.logger.info(f"📉 Transição de grupo concluída: {old_group_name} → {new_group_name} (regressão)")
        else:
            self.logger.info(f"📈 Transição de grupo concluída: {old_group_name} → {new_group_name} (progressão)")
        
        return result
    
    def _is_final_sub_phase(self) -> bool:
        """Verifica se está na sub-fase final do grupo final"""
        return (self.current_group == len(self.groups) - 1 and 
                self.current_sub_phase == len(self.current_group_config.sub_phases) - 1)
    
    # Métricas de performance (mantidas do original)
    def _update_success_counters(self, episode_results: Dict):
        """Atualiza contadores de sucesso e estagnação"""
        success = episode_results.get("success", False)
        distance = episode_results.get("distance", 0)
        
        if success and distance > 0.05:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        # Detectar estagnação
        if len(self.performance_history) >= 12: 
            recent_distances = [r.get("distance", 0) for r in self.performance_history[-12:]]
            current_distance = episode_results.get("distance", 0)

            avg_recent = np.mean(recent_distances)
            std_recent = np.std(recent_distances)

            if (std_recent < 0.15 and  
                abs(current_distance - avg_recent) < 0.25 and  
                current_distance < 1.2):  
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
    
    def _check_basic_conditions(self, conditions: Dict) -> bool:
        """Verifica condições básicas obrigatórias"""
        success_rate = self._calculate_success_rate()
        if success_rate < conditions.get("min_success_rate", 0.3):
            return False
        
        avg_distance = self._calculate_avg_distance()
        if avg_distance < conditions.get("min_avg_distance", 0.5):
            return False
        
        avg_roll = self._calculate_avg_roll()
        if avg_roll > conditions.get("max_avg_roll", 1.0):
            return False
        
        return True
    
    def _check_advanced_consistency(self) -> bool:
        """Verifica consistência avançada para transição de grupo"""
        if len(self.performance_history) < 8:
            return False
        
        recent_results = self.performance_history[-8:]
        
        # Consistência nas distâncias
        distances = [r.get("distance", 0) for r in recent_results]
        distance_std = np.std(distances)
        distance_mean = np.mean(distances) if np.mean(distances) > 0 else 0.1
        
        if (distance_std / distance_mean) > 0.3:  
            return False
        
        # Consistência no sucesso
        successes = [1 if r.get("success", False) else 0 for r in recent_results]
        success_rate = np.mean(successes)
        
        if success_rate < 0.6:  
            return False
        
        return True
    
    # Métricas de cálculo 
    def _calculate_success_rate(self) -> float:
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-10:]
        successes = 0
        for r in recent_history:
            if r.get("success", False):
                successes += 1
            elif r.get("episode_success", False):
                successes += 1
            elif r.get("distance", 0) > 0.5:  
                successes += 1

        return successes / len(recent_history)
    
    def _calculate_avg_distance(self) -> float:
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-10:]
        distances = [r.get("distance", 0) for r in recent_history]
        return np.mean(distances)
    
    def _calculate_avg_roll(self) -> float:
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-10:]
        rolls = [abs(r.get("roll", 0)) for r in recent_history]
        return np.mean(rolls)
    
    def _calculate_avg_speed(self) -> float:
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-10:]
        speeds = [r.get("speed", 0) for r in recent_history]
        return np.mean(speeds)
    
    def _calculate_alternating_score(self) -> float:
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-8:]
        alternations = sum(1 for r in recent_history if r.get("alternating", False))
        return alternations / len(recent_history)
    
    def _calculate_gait_coordination(self) -> float:
        if not self.performance_history:
            return 0.3
        recent_history = self.performance_history[-10:]
        coordination_scores = []
        for result in recent_history:
            left_contact = result.get("left_contact", False)
            right_contact = result.get("right_contact", False)
            alternation = 1.0 if left_contact != right_contact else 0.0
            coordination_scores.append(alternation)
        return np.mean(coordination_scores) if coordination_scores else 0.3
    
    def _calculate_positive_movement_rate(self) -> float:
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-10:]
        positive_movements = sum(1 for r in recent_history if r.get("distance", 0) > 0.1)
        return positive_movements / len(recent_history)
    
    def _calculate_avg_steps(self) -> float:
        """Calcula média de passos"""
        if not self.performance_history:
            return 0.0
        recent_history = self.performance_history[-10:]
        steps = [r.get("steps", 0) for r in recent_history]
        return np.mean(steps) if steps else 0.0

    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance consolidadas"""
        return {
            "success_rate": self._calculate_success_rate(),
            "avg_distance": self._calculate_avg_distance(),
            "avg_roll": self._calculate_avg_roll(),
            "avg_speed": self._calculate_avg_speed(),
            "positive_movement_rate": self._calculate_positive_movement_rate(),
            "avg_steps": self._calculate_avg_steps(),
            "alternating_score": self._calculate_alternating_score(),
            "gait_coordination": self._calculate_gait_coordination()
        }

    def get_current_phase_info(self) -> Dict:
        """Retorna informações da sub-fase atual"""
        group_config = self.current_group_config
        sub_phase_config = self.current_sub_phase_config
        
        info = {
            'group': self.current_group,
            'group_level': self.current_group,
            'group_name': group_config.name,
            'sub_phase': self.current_sub_phase,
            'sub_phase_name': sub_phase_config.name,
            'target_speed': sub_phase_config.target_speed,
            'enabled_components': sub_phase_config.enabled_components,
            'component_weights': sub_phase_config.component_weights,
            'focus_skills': sub_phase_config.focus_skills,
            'episodes_in_sub_phase': self.episodes_in_sub_phase,
            'group_level': group_config.group_level,
            'adaptive_config': group_config.adaptive_config
        }

        return info
    
    def get_status(self) -> Dict:
        """Retorna status com informações de validação"""
        performance_metrics = self.get_performance_metrics()

        status = {
            "current_group": self.current_group,
            "current_sub_phase": self.current_sub_phase,
            "episodes_in_sub_phase": self.episodes_in_sub_phase,
            "validation_required": self.validation_required,
            "last_validation_episode": self.last_validation_episode,
            "consecutive_failures": self.consecutive_failures,
            "performance_trend": self._calculate_performance_trend(),
            "success_rate": performance_metrics["success_rate"],
            "avg_distance": performance_metrics["avg_distance"],
            "avg_roll": performance_metrics["avg_roll"],
        }
        
        # Adicionar status do validador
        status.update(self.validator.get_validation_status())
        
        return status
    
    def get_current_sub_phase_info(self) -> Dict:
        """Retorna informações da sub-fase atual """
        return {
            "current_phase": self.current_group,  
            "phase_index": self.current_sub_phase,  
            "target_speed": self.current_sub_phase_config.target_speed,
            "episodes_in_phase": self.episodes_in_sub_phase,
            "performance_metrics": {
                "success_rate": self._calculate_success_rate(),
                "avg_distance": self._calculate_avg_distance(),
                "avg_roll": self._calculate_avg_roll(),
                "avg_speed": self._calculate_avg_speed(),
                "positive_movement_rate": self._calculate_positive_movement_rate()
            }
        }
        
    @property
    def current_group_config(self) -> PhaseGroup:
        return self.groups[self.current_group]
    
    @property
    def current_sub_phase_config(self) -> SubPhaseConfig:
        return self.current_group_config.sub_phases[self.current_sub_phase]