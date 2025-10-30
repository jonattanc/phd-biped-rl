# dpg_gait_phases.py
from datetime import datetime
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import utils
from enum import Enum

class PhaseTransitionResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure" 
    REGRESSION = "regression"
    STAGNATION = "stagnation"

@dataclass
class GaitPhaseConfig:
    """Configuração de uma fase da marcha para DPG com validação robusta"""
    name: str
    target_speed: float  # m/s
    enabled_components: List[str]
    component_weights: Dict[str, float]
    phase_duration: int  # episódios mínimos na fase
    transition_conditions: Dict[str, float]  # condições para transição
    skill_requirements: Dict[str, float]  # habilidades específicas necessárias
    regression_thresholds: Dict[str, float]  # limites para regressão

class GaitPhaseDPG:
    """
    Sistema DPG avançado com validação robusta, fallback adaptativo e métricas detalhadas
    """
    
    def __init__(self, logger, reward_system):
        self.logger = logger
        self.reward_system = reward_system
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.phases = []
        self.performance_history = []
        self.skill_assessment_history = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.phase_validation_history = []
        self.stagnation_counter = 0
        self.last_avg_distance = 0.0
        self.regression_count = 0
        
        self._initialize_enhanced_gait_phases()
        
    def _initialize_enhanced_gait_phases(self):
        """Inicializa as fases da marcha com requisitos mais rigorosos"""
        
        # FASE 0: ESTABILIDADE BÁSICA E CONTROLE POSTURAL
        phase0 = GaitPhaseConfig(
            name="estabilidade_postural",
            target_speed=0.3,  
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "alternating_foot_contact",
                "success_bonus", "distance_bonus", "height_deviation_penalty"
            ],
            component_weights={
                "progress": 0.4,           
                "stability_roll": 0.2,   
                "stability_pitch": 0.15,
                "alternating_foot_contact": 0.15,  
                "success_bonus": 0.05,     
                "distance_bonus": 0.04, 
                "height_deviation_penalty": 0.01 
            },
            phase_duration=5,  
            transition_conditions={
                "min_success_rate": 0.02,       
                "min_avg_distance": 0.15,       
                "max_avg_roll": 0.8,          
                "min_avg_steps": 10,           
                "consistency_count": 2        
            },
            skill_requirements={
                "basic_balance": 0.4,       
                "postural_stability": 0.3,     
                "gait_initiation": 0.1        
            },
            regression_thresholds={
                "max_failures": 20,           
                "min_success_rate": 0.02,      
                "stagnation_episodes": 30     
            }
        )
        
        # FASE 1: MARCHA LENTA COM PADRÃO ALTERNADO
        phase1 = GaitPhaseConfig(
            name="marcha_lenta_alternada",
            target_speed=0.6,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "foot_clearance", "success_bonus", "distance_bonus",
                "effort_penalty", "y_axis_deviation_penalty"
            ],
            component_weights={
                "progress": 0.35,         
                "stability_roll": 0.2,     
                "stability_pitch": 0.15,   
                "alternating_foot_contact": 0.12,  
                "gait_pattern_cross": 0.08, 
                "foot_clearance": 0.05,    
                "success_bonus": 0.02,
                "distance_bonus": 0.02,
                "effort_penalty": 0.005,   
                "y_axis_deviation_penalty": 0.01  
            },
            phase_duration=25,
            transition_conditions={
                "min_success_rate": 0.5,      
                "min_avg_distance": 1.5,       
                "max_avg_roll": 0.3,           
                "min_avg_speed": 0.2,          
                "min_alternating_score": 0.5, 
                "min_gait_coordination": 0.3,  
                "consistency_count": 5
            },
            skill_requirements={
                "rhythmic_gait": 0.5,          
                "foot_clearance_control": 0.4,  
                "step_consistency": 0.6,
            },
            regression_thresholds={
                "max_failures": 12,
                "min_success_rate": 0.5,
                "stagnation_episodes": 15
            }
        )
        
        # FASE 2: MARCHA RÁPIDA COM PROPULSÃO
        phase2 = GaitPhaseConfig(
            name="marcha_rapida_propulsiva",
            target_speed=1.2,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "effort_square_penalty", "y_axis_deviation_square_penalty",
                "jerk_penalty"
            ],
            component_weights={
                "progress": 0.4,           
                "stability_roll": 0.15,   
                "stability_pitch": 0.12,   
                "alternating_foot_contact": 0.08,
                "gait_pattern_cross": 0.1, 
                "foot_clearance": 0.06,   
                "pitch_forward_bonus": 0.04, 
                "success_bonus": 0.02,
                "distance_bonus": 0.02,
                "effort_square_penalty": 0.008, 
                "y_axis_deviation_square_penalty": 0.01,
                "jerk_penalty": 0.005      
            },
            phase_duration=30,
            transition_conditions={
                "min_success_rate": 0.7,       
                "min_avg_distance": 3.5,       
                "min_avg_speed": 0.6,          
                "max_avg_roll": 0.15,          
                "min_propulsion_efficiency": 0.5, 
                "min_gait_coordination": 0.75,
                "consistency_count": 12
            },
            skill_requirements={
                "balance_recovery": 0.6,        
                "propulsive_phase": 0.5,        
                "dynamic_balance": 0.7,     
            },
            regression_thresholds={
                "max_failures": 10,
                "min_success_rate": 0.6,
                "stagnation_episodes": 12
            }
        )
        
        # FASE 3: CORRIDA COM FASE DE VOO
        phase3 = GaitPhaseConfig(
            name="corrida_com_voo",
            target_speed=1.5,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "foot_clearance",
                "pitch_forward_bonus", "success_bonus", "distance_bonus", 
                "gait_pattern_cross", "effort_square_penalty", "jerk_penalty",
                "center_bonus", "warning_penalty"
            ],
            component_weights={
                "progress": 0.45,         
                "stability_roll": 0.12,    
                "stability_pitch": 0.1,   
                "foot_clearance": 0.08,    
                "pitch_forward_bonus": 0.06,
                "success_bonus": 0.03,
                "distance_bonus": 0.03,
                "gait_pattern_cross": 0.07, 
                "effort_square_penalty": 0.01,
                "jerk_penalty": 0.008,     
                "center_bonus": 0.02,     
                "warning_penalty": 0.015   
            },
            phase_duration=25,
            transition_conditions={
                "min_success_rate": 0.85,     
                "min_avg_distance": 7.0,       
                "min_avg_speed": 1.4,          
                "max_avg_roll": 0.12,          
                "min_flight_phase_quality": 0.6, 
                "min_propulsion_efficiency": 0.7,
                "consistency_count": 15
            },
            skill_requirements={
                "flight_phase_control": 0.7,   
                "impact_absorption": 0.75,     
                "high_speed_stability": 0.8   
            },
            regression_thresholds={
                "max_failures": 8,
                "min_success_rate": 0.7,
                "stagnation_episodes": 10
            }
        )
        
        # FASE 4: CORRIDA AVANÇADA E EFICIENTE
        phase4 = GaitPhaseConfig(
            name="corrida_avancada_eficiente",
            target_speed=2.0,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "foot_clearance",
                "pitch_forward_bonus", "success_bonus", "distance_bonus", 
                "gait_pattern_cross", "center_bonus", "warning_penalty",
                "effort_square_penalty", "jerk_penalty", "direction_change_penalty"
            ],
            component_weights={
                "progress": 0.5,           
                "stability_roll": 0.1,     
                "stability_pitch": 0.08,   
                "foot_clearance": 0.07,    
                "pitch_forward_bonus": 0.08, 
                "success_bonus": 0.04,
                "distance_bonus": 0.04,
                "gait_pattern_cross": 0.05, 
                "center_bonus": 0.02,
                "warning_penalty": 0.015,
                "effort_square_penalty": 0.012, 
                "jerk_penalty": 0.01,      
                "direction_change_penalty": 0.005 
            },
            phase_duration=20,
            transition_conditions={
                "min_success_rate": 0.9,       
                "min_avg_distance": 9.0,       
                "min_avg_speed": 2.0,          
                "max_avg_roll": 0.1,           
                "min_energy_efficiency": 0.8,  
                "min_gait_consistency": 0.85,  
                "consistency_count": 18
            },
            skill_requirements={
                "energy_efficiency": 0.8,      
                "high_speed_maneuver": 0.75,   
                "adaptive_gait": 0.7           
            },
            regression_thresholds={
                "max_failures": 5,
                "min_success_rate": 0.8,
                "stagnation_episodes": 8
            }
        )
        
        self.phases = [phase0, phase1, phase2, phase3, phase4]
        
    def update_phase(self, episode_results: Dict) -> PhaseTransitionResult:
        """
        Atualiza a fase atual com validação robusta e fallback adaptativo
        """    
        # VERIFICAÇÃO MAIS FLEXÍVEL para episódios
        episode_duration = episode_results.get('duration', 0)
        episode_distance = episode_results.get('distance', 0)
        episode_steps = episode_results.get('steps', 0)

        # Critérios mais flexíveis para identificar um episódio real
        is_valid_episode = (
            episode_duration >= 0.1 and      # Duração mínima reduzida
            episode_steps >= 5 and           # Mínimo de 5 steps
            episode_distance >= 0            # Distância pode ser 0
        )

        if not is_valid_episode:
            self.logger.debug(f"Episódio ignorado - dados insuficientes: duration={episode_duration:.2f}s, steps={episode_steps}, distance={episode_distance:.2f}m")
            return PhaseTransitionResult.FAILURE

        # INCREMENTAR CONTADOR DE EPISÓDIOS NA FASE
        self.episodes_in_phase += 1

        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS 

        # Adicionar resultado ao histórico
        enhanced_results = self._enhance_episode_results(episode_results)
        self.performance_history.append(enhanced_results)

        current_phase_success = enhanced_results.get("phase_success", False)
        self.logger.debug(f"Episódio {len(self.performance_history)} - phase_success: {current_phase_success}")

        # Atualizar contadores de sucesso/fracasso
        self._update_success_failure_counters(enhanced_results)

        if len(self.performance_history) < 3:
            return PhaseTransitionResult.FAILURE

        # Verificar regressão ou estagnação PRIMEIRO
        regression_result = self._check_regression_or_stagnation()
        if regression_result != PhaseTransitionResult.SUCCESS:
            return regression_result

        # VERIFICAR SE PODE AVANÇAR DE FASE
        can_advance = self._check_phase_advancement()
        if can_advance:
            self.logger.info("CONDIÇÕES ATENDIDAS - Avançando para próxima fase!")
            return self._advance_to_next_phase()

        return PhaseTransitionResult.FAILURE
        
    def _check_phase_advancement(self) -> bool:
        """Verifica se todas as condições para avançar de fase foram atendidas"""
        if self.current_phase >= len(self.phases) - 1:
            return False

        current_phase_config = self.phases[self.current_phase]

        # Verificar requisitos mínimos PRIMEIRO
        if not self._meets_minimum_requirements():
            return False

        # Verificar métricas de desempenho
        success_rate = self._calculate_success_rate()
        avg_distance = self._calculate_average_distance()
        avg_roll = self._calculate_average_roll()
        avg_steps = np.mean([r.get('steps', 0) for r in self.performance_history])

        # Verificar condições básicas
        basic_conditions_met = (
            success_rate >= current_phase_config.transition_conditions["min_success_rate"] and
            avg_distance >= current_phase_config.transition_conditions["min_avg_distance"] and 
            avg_roll <= current_phase_config.transition_conditions["max_avg_roll"] and
            avg_steps >= current_phase_config.transition_conditions.get("min_avg_steps", 5)
        )

        if not basic_conditions_met:
            return False

        # Verificar condições adicionais
        additional_conditions_met = (
            self._check_performance_consistency() and
            self._validate_phase_skills()
        )

        return additional_conditions_met

    def _advance_to_next_phase(self) -> PhaseTransitionResult:
        """Avança para a próxima fase"""
        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS

        old_phase = self.current_phase
        self.current_phase += 1
        self.episodes_in_phase = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.stagnation_counter = 0
        self.regression_count = 0
        # Manter o histórico de performance para continuidade

        new_phase_config = self.phases[self.current_phase]

        self.logger.info(f"🎉 AVANÇO DE FASE: {self.phases[old_phase].name} → {new_phase_config.name}")

        self._apply_phase_config()
        return PhaseTransitionResult.SUCCESS

    def _enhance_episode_results(self, episode_results: Dict) -> Dict:
        """Adiciona métricas calculadas aos resultados do episódio"""
        enhanced = episode_results.copy()
            
        # Calcular métricas adicionais
        if "distance" in episode_results and "duration" in episode_results:
            enhanced["speed"] = episode_results["distance"] / max(episode_results["duration"], 0.1)
            enhanced["efficiency"] = episode_results["distance"] / max(episode_results.get("energy_used", 1), 1)
        
        # Score de padrão alternado
        left_contact = episode_results.get("left_contact", False)
        right_contact = episode_results.get("right_contact", False)
        
        # Padrão alternado = um pé em contato, outro não
        if left_contact != right_contact:
            enhanced["alternating_score"] = 1.0
        else:
            enhanced["alternating_score"] = 0.0
        
        # Score de coordenação
        enhanced["coordination_score"] = episode_results.get("gait_pattern_score", 0.5)
        
        current_phase = self.current_phase
        episode_distance = episode_results.get("distance", 0)
        episode_steps = episode_results.get("steps", 0)
        episode_roll = abs(episode_results.get("roll", 0))
        episode_speed = enhanced.get("speed", 0)

        if current_phase == 0:  # estabilidade_postural
            phase_success = (episode_distance > 0.1 and 
                       episode_roll < 0.8 and 
                       episode_steps > 5)
        elif current_phase == 1:  # marcha_lenta_alternada
             phase_success = (episode_distance > 1.5 and 
                        episode_steps > 30 and
                        episode_speed > 0.2 and
                        episode_roll < 0.3)
        elif current_phase == 2:  # marcha_rapida_propulsiva
            phase_success = (episode_distance > 3.5 and episode_speed > 0.6)
        elif current_phase == 3:  # corrida_com_voo
            phase_success = (episode_distance > 7.0 and episode_speed > 1.4)
        else:  # Fase 4+ - usar sucesso original
            phase_success = episode_results.get("success", False)

        enhanced["phase_success"] = phase_success
        
        # DEBUG: Log para verificar cálculo
        if current_phase == 1 and episode_steps > 10:
            self.logger.debug(f"Fase 1 - phase_success: {phase_success} | dist: {episode_distance:.2f} > 1.5? | steps: {episode_steps} > 30? | speed: {episode_speed:.2f} > 0.2? | roll: {episode_roll:.2f} < 0.3?")
        
        return enhanced
        
    def _update_success_failure_counters(self, episode_results: Dict):
        """Atualiza contadores de sucesso e fracasso"""
        phase_success = episode_results.get("phase_success", False)
        if phase_success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
        # Verificar estagnação de progresso
        current_avg_distance = self._calculate_average_distance()
        if abs(current_avg_distance - self.last_avg_distance) < 0.1:  # Menos de 10cm de melhoria
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_avg_distance = current_avg_distance
        
    def _check_regression_or_stagnation(self) -> PhaseTransitionResult:
        """Verifica se há regressão ou estagnação que requer ação"""
        current_phase = self.phases[self.current_phase]
        regression_thresholds = current_phase.regression_thresholds

        if self.current_phase == 0:
            if self.consecutive_failures >= 50:  
                return PhaseTransitionResult.FAILURE
            return PhaseTransitionResult.SUCCESS
        if self.current_phase <= 1:  
            if self.consecutive_failures >= 50:
                return PhaseTransitionResult.FAILURE
            return PhaseTransitionResult.SUCCESS
        
        # Relaxar outros thresholds
        if self.consecutive_failures >= regression_thresholds["max_failures"] * 2:  
            self.regression_count += 1
            if self._should_regress_phase():
                return self._regress_to_previous_phase()
            return PhaseTransitionResult.FAILURE

        # Relaxar threshold de estagnação
        if self.stagnation_counter >= regression_thresholds["stagnation_episodes"] * 2:
            return PhaseTransitionResult.STAGNATION

        # Relaxar taxa de sucesso mínima
        success_rate = self._calculate_success_rate()
        if success_rate < regression_thresholds["min_success_rate"] * 0.5:
            self.regression_count += 1
            if self._should_regress_phase():
                return self._regress_to_previous_phase()
            return PhaseTransitionResult.SUCCESS

        return PhaseTransitionResult.SUCCESS
        
    def _should_regress_phase(self) -> bool:
        """Decide se deve regredir para fase anterior"""
        if self.current_phase == 0:
            return False
    
        should_regress = (
        self.current_phase > 0 and 
        self.regression_count >= 3 and  
        self.episodes_in_phase > self.phases[self.current_phase].phase_duration * 2 
    )
        return should_regress
        
    def _regress_to_previous_phase(self) -> PhaseTransitionResult:
        """Regride para a fase anterior"""
        if self.current_phase == 0:
            return PhaseTransitionResult.FAILURE
    
        old_phase = self.current_phase
        self.current_phase = max(0, self.current_phase - 1)
        self.episodes_in_phase = 0
        self.consecutive_failures = 0
        self.regression_count = 0
        self.stagnation_counter = 0
        self.performance_history = []
        
        # Manter os últimos 10-15 episódios para preservar aprendizado
        if len(self.performance_history) > 0:
            self.logger.info(f"Manteve todo o histórico ({len(self.performance_history)} episódios) após regressão")
        else:
            self.logger.info("Sem histórico para manter após regressão")

        self.logger.warning(f"🔄 REGRESSÃO DE FASE: {self.phases[old_phase].name} → {self.phases[self.current_phase].name}")

        self._apply_phase_config()
        return PhaseTransitionResult.REGRESSION
        
    def _meets_minimum_requirements(self) -> bool:
        """Verifica requisitos mínimos para transição"""
        current_phase_config = self.phases[self.current_phase]
        has_minimum_duration = self.episodes_in_phase >= current_phase_config.phase_duration
    
        # Verificar histórico suficiente
        has_sufficient_history = len(self.performance_history) >= 5

        if not has_minimum_duration:
            self.logger.debug(f"Aguardando duração mínima: {self.episodes_in_phase}/{current_phase_config.phase_duration}")

        if not has_sufficient_history:
            self.logger.debug(f"Aguardando histórico suficiente: {len(self.performance_history)}/5")

        return has_minimum_duration and has_sufficient_history
        
    def _validate_phase_skills(self) -> bool:
        """Valida apenas as habilidades REQUERIDAS pela fase atual"""
        current_phase = self.phases[self.current_phase]
        skill_requirements = current_phase.skill_requirements

        all_skills = self._assess_phase_skills()
        
        all_skills_met = True
        for skill, required_score in skill_requirements.items():
            current_score = all_skills.get(skill, 0)
            skill_met = current_score >= required_score
            all_skills_met = all_skills_met and skill_met

        return all_skills_met
        
    def _assess_phase_skills(self) -> Dict[str, float]:
        """Calcula TODAS as habilidades baseadas no histórico"""
        if len(self.performance_history) < 5:
            return {}

        recent_results = self.performance_history[-10:]
        avg_roll = self._calculate_average_roll()
        success_rate = self._calculate_success_rate()
        avg_speed = self._calculate_average_speed()

        all_skills = {
            # Habilidades BÁSICAS
            "postural_stability": 1.0 - min(avg_roll / 1.0, 1.0),
            "gait_initiation": success_rate,
            "basic_balance": 1.0 - min(avg_roll / 0.8, 1.0),

            # Habilidades de MARCHA  
            "rhythmic_gait": np.mean([r.get("alternating_score", 0) for r in recent_results]),
            "foot_clearance_control": np.mean([r.get("clearance_score", 0.5) for r in recent_results]),
            "step_consistency": 1.0 - np.std([r.get("distance", 0) for r in recent_results]) / 2.0,

            # Habilidades AVANÇADAS
            "balance_recovery": self._calculate_balance_recovery_score(recent_results),
            "propulsive_phase": self._calculate_propulsion_efficiency(),
            "dynamic_balance": 1.0 - min(avg_roll / 0.3, 1.0),
            "flight_phase_control": np.mean([r.get("flight_quality", 0.5) for r in recent_results]),
        }

        return all_skills
        
    def _check_performance_consistency(self) -> bool:
        """Verifica consistência do desempenho"""
        current_phase = self.phases[self.current_phase]
        required_consistency = current_phase.transition_conditions.get("consistency_count", 5)

        if len(self.performance_history) < required_consistency:
            self.logger.debug(f"Histórico insuficiente para consistência: {len(self.performance_history)}/{required_consistency}")
            return False

        # Verificar últimos 'n' episódios
        recent_results = self.performance_history[-required_consistency:]

        # Calcular taxa de sucesso nos episódios recentes
        successful_episodes = sum(1 for r in recent_results if r.get("phase_success", False))
        consistency_ratio = successful_episodes / len(recent_results)

        required_success_rate = current_phase.transition_conditions["min_success_rate"]

        is_consistent = consistency_ratio >= required_success_rate

        if not is_consistent:
            self.logger.debug(f"Consistência insuficiente: {consistency_ratio:.3f} < {required_success_rate}")

        return is_consistent
        
    def _calculate_balance_recovery_score(self, recent_results: List[Dict]) -> float:
        """Calcula capacidade de recuperar equilíbrio"""
        if len(recent_results) < 3:
            return 0.5
            
        # Analisar padrão de recuperação após instabilidade
        recovery_events = 0
        total_events = 0
        
        for i in range(1, len(recent_results)):
            prev_roll = abs(recent_results[i-1].get("roll", 0))
            curr_roll = abs(recent_results[i].get("roll", 0))
            
            if prev_roll > 0.3 and curr_roll < 0.2:
                recovery_events += 1
            total_events += 1
            
        score = recovery_events / max(total_events, 1)
    
        return score
        
    def _calculate_propulsion_efficiency(self) -> float:
        """Calcula eficiência propulsiva"""
        if not self.performance_history:
            return 0.0
            
        speeds = [r.get("speed", 0) for r in self.performance_history[-10:]]
        efforts = [r.get("energy_used", 1) for r in self.performance_history[-10:]]
        
        efficiencies = [s / max(e, 0.1) for s, e in zip(speeds, efforts)]
        return np.mean(efficiencies) / 2.0  # Normalizado
        
    def _calculate_consistency_score(self) -> float:
        """Calcula score de consistência geral"""
        if len(self.performance_history) < 5:
            return 0.0
            
        distances = [r.get("distance", 0) for r in self.performance_history[-10:]]
        speeds = [r.get("speed", 0) for r in self.performance_history[-10:]]
        
        # Baixa variância = alta consistência
        distance_consistency = 1.0 - min(np.std(distances) / 2.0, 1.0)
        speed_consistency = 1.0 - min(np.std(speeds) / 1.0, 1.0)
        
        return (distance_consistency + speed_consistency) / 2.0
        
    def _apply_phase_config(self):
        """Aplica a configuração da fase atual ao sistema de recompensa"""
        current_phase = self.phases[self.current_phase]
        
        # Atualizar componentes habilitados e pesos
        for component_name, component in self.reward_system.components.items():
            if component_name in current_phase.enabled_components:
                component.enabled = True
                target_weight = current_phase.component_weights.get(component_name, component.weight)
                component.weight = target_weight
            else:
                component.enabled = False
        
    def get_current_speed_target(self) -> float:
        """Retorna a velocidade alvo da fase atual"""
        return self.phases[self.current_phase].target_speed
        
    def get_detailed_status(self) -> Dict:
        """Retorna status detalhado com métricas de progresso"""
        current_phase = self.phases[self.current_phase]
        
        # Calcular métricas do histórico COMPLETO
        total_episodes = len(self.performance_history)
        success_rate = self._calculate_success_rate()
        
        # Calcular métricas por fase
        phase_0_episodes = len([r for r in self.performance_history if r.get('phase', 0) == 0])
        phase_1_episodes = len([r for r in self.performance_history if r.get('phase', 0) == 1])
        
        return {
            "current_phase": current_phase.name,
            "phase_index": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "target_speed": current_phase.target_speed,
            "performance_metrics": {
                "success_rate": success_rate,
                "avg_distance": self._calculate_average_distance(),
                "history_size": total_episodes,
                "phase_0_episodes": phase_0_episodes,
                "phase_1_episodes": phase_1_episodes,
                "total_successes": sum(1 for r in self.performance_history if r.get("phase_success", False))
            },
            "skill_assessment": self._assess_phase_skills(),
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "regression_count": self.regression_count
        }
        
    def _calculate_progress_metrics(self) -> Dict:
        """Calcula métricas detalhadas de progresso"""
        success_rate = self._calculate_success_rate() 
        return {
            "success_rate": success_rate,
            "avg_distance": self._calculate_average_distance(),
            "avg_speed": self._calculate_average_speed(),
            "avg_roll": self._calculate_average_roll(),
            "stability_score": 1.0 - min(self._calculate_average_roll() / 0.5, 1.0),
            "efficiency_score": np.mean([r.get("efficiency", 0) for r in self.performance_history[-5:]]),
            "consistency_score": self._calculate_consistency_score(),
            "history_size": len(self.performance_history)
        }
        
    def reset(self):
        """Reinicia o DPG para a primeira fase"""
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.performance_history = []
        self.skill_assessment_history = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.stagnation_counter = 0
        self.regression_count = 0
        self._apply_phase_config()
        
    # Métodos de cálculo mantidos para compatibilidade
    def _calculate_success_rate(self) -> float:
        if not self.performance_history:
            return 0.0
        successes = sum(1 for result in self.performance_history if result.get("phase_success", False))
        total = len(self.performance_history)

        return successes / total
        
    def _calculate_average_distance(self) -> float:
        if not self.performance_history:
            return 0.0
        return np.mean([result.get("distance", 0) for result in self.performance_history])
        
    def _calculate_average_speed(self) -> float:
        if not self.performance_history:
            return 0.0
        speeds = [result.get("speed", 0) for result in self.performance_history 
                 if result.get("speed", 0) > 0]
        return np.mean(speeds) if speeds else 0.0
        
    def _calculate_average_roll(self) -> float:
        if not self.performance_history:
            return 0.0
        return np.mean([abs(result.get("roll", 0)) for result in self.performance_history])

    # Método de compatibilidade
    def get_status(self) -> Dict:
        """Método legado para compatibilidade"""
        status = self.get_detailed_status()
        return {
            "current_phase": status["current_phase"],
            "phase_index": status["phase_index"],
            "episodes_in_phase": status["episodes_in_phase"],
            "target_speed": status["target_speed"],
            "total_phases": status["total_phases"],
            "phase_progress": status["phase_progress"]
        }
    
    def _get_current_reward_components(self) -> Dict:
        """Obtém contribuição atual dos componentes de recompensa"""
        components = {}
        if hasattr(self.reward_system, 'components'):
            for name, component in self.reward_system.components.items():
                if component.enabled:
                    components[name] = component.value
        return components

    def get_detailed_status(self) -> Dict:
        """Retorna status detalhado com métricas de progresso"""
        current_phase = self.phases[self.current_phase]

        # Calcular métricas atuais
        success_rate = self._calculate_success_rate()
        avg_distance = self._calculate_average_distance()
        avg_speed = self._calculate_average_speed()

        status = {
            "current_phase": current_phase.name,
            "phase_index": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,  # AGORA deve mostrar corretamente
            "target_speed": current_phase.target_speed,
            "total_phases": len(self.phases),
            "phase_progress": f"{self.current_phase + 1}/{len(self.phases)}",
            "performance_metrics": {
                "success_rate": success_rate,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "history_size": len(self.performance_history)
            },
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "regression_count": self.regression_count
        }

        # Adicionar habilidades se houver histórico
        if len(self.performance_history) >= 3:
            try:
                skill_scores = self._assess_phase_skills()
                status["skill_assessment"] = {k: round(v, 3) for k, v in skill_scores.items()}
            except Exception as e:
                status["skill_assessment"] = {"error": str(e)}

        return status