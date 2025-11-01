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
        self.progression_history = []  
        self.max_progression_history = 50
        self.skill_assessment_history = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.phase_validation_history = []
        self.stagnation_counter = 0
        self.last_avg_distance = 0.0
        self.regression_count = 0

        self._initialize_enhanced_gait_phases()

    def _initialize_enhanced_gait_phases(self):
        """Fases mais graduais e realistas"""

        # FASE 0: ESTABILIDADE BÁSICA 
        phase0 = GaitPhaseConfig(
            name="estabilidade_postural",
            target_speed=0.2, 
            enabled_components=[
                "stability_roll", "stability_pitch", "center_bonus", 
                "success_bonus", "distance_bonus", "warning_penalty"
            ],
            component_weights={
                "stability_roll": 0.7,      
                "stability_pitch": 0.15,    
                "center_bonus": 0.05,
                "success_bonus": 0.04,
                "distance_bonus": 0.06,     
                "warning_penalty": 0.0,     
            },
            phase_duration=15,  
            transition_conditions={
                "min_success_rate": 0.15,   
                "min_avg_distance": 0.2,    
                "max_avg_roll": 0.5,        
                "min_avg_steps": 5,         
                "consistency_count": 1,     
            },
            skill_requirements={
                "basic_balance": 0.3,       
                "postural_stability": 0.2,   
                "gait_initiation": 0.1,     
            },
            regression_thresholds={
                "max_failures": 40,         
                "min_success_rate": 0.05,
                "stagnation_episodes": 80,
            },
        )

        # FASE 1: MARCHA LENTA 
        phase1 = GaitPhaseConfig(
            name="marcha_lenta_alternada",
            target_speed=0.8,  
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "success_bonus", "center_bonus",
            ],
            component_weights={
                "progress": 0.3,        
                "distance_bonus": 0.25,
                "stability_roll": 0.2,
                "stability_pitch": 0.15,
                "alternating_foot_contact": 0.06,
                "success_bonus": 0.02,
                "center_bonus": 0.02,
            },
            phase_duration=25, 
            transition_conditions={
                "min_success_rate": 0.15,  
                "min_avg_distance": 1.5,   
                "max_avg_roll": 0.4,      
                "min_alternating_score": 0.3,
                "min_gait_coordination": 0.2,
            },
            skill_requirements={
                "basic_balance": 0.5,
                "postural_stability": 0.4,
                "step_consistency": 0.4,
            },
            regression_thresholds={
                "max_failures": 30, 
                "min_success_rate": 0.08,
                "stagnation_episodes": 50,
            },
        )

        # FASE 2: MARCHA RÁPIDA 
        phase2 = GaitPhaseConfig(
            name="marcha_rapida_propulsiva",
            target_speed=1.5,  
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "alternating_foot_contact", "gait_pattern_cross",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "effort_square_penalty", 
                "y_axis_deviation_square_penalty", "jerk_penalty",
            ],
            component_weights={
                "progress": 0.35,      
                "stability_roll": 0.18,
                "stability_pitch": 0.15,
                "alternating_foot_contact": 0.08,
                "gait_pattern_cross": 0.08,  
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.05,
                "success_bonus": 0.02,
                "distance_bonus": 0.03,
                "effort_square_penalty": 0.008,
                "y_axis_deviation_square_penalty": 0.01,
                "jerk_penalty": 0.005,
            },
            phase_duration=40,  
            transition_conditions={
                "min_success_rate": 0.5,    
                "min_avg_distance": 2.5,   
                "min_avg_speed": 0.8,      
                "max_avg_roll": 0.25,       
                "min_propulsion_efficiency": 0.4,
                "min_gait_coordination": 0.6,
                "consistency_count": 8,    
            },
            skill_requirements={
                "balance_recovery": 0.5,   
                "propulsive_phase": 0.4,     
                "dynamic_balance": 0.6,    
            },
            regression_thresholds={
                "max_failures": 15, 
                "min_success_rate": 0.3,
                "stagnation_episodes": 20,
            },
        )

        # FASE 3: MARCHA EFICIENTE 
        phase3 = GaitPhaseConfig(
            name="marcha_eficiente",
            target_speed=1.8,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "gait_pattern_cross", "effort_square_penalty",
                "jerk_penalty", "center_bonus", "warning_penalty",
            ],
            component_weights={
                "progress": 0.4,
                "stability_roll": 0.15,
                "stability_pitch": 0.12,
                "foot_clearance": 0.07,
                "pitch_forward_bonus": 0.06,
                "success_bonus": 0.03,
                "distance_bonus": 0.04,
                "gait_pattern_cross": 0.06,
                "effort_square_penalty": 0.01,
                "jerk_penalty": 0.008,
                "center_bonus": 0.02,
                "warning_penalty": 0.015,
            },
            phase_duration=35,
            transition_conditions={
                "min_success_rate": 0.7,
                "min_avg_distance": 4.0,
                "min_avg_speed": 1.2,
                "max_avg_roll": 0.2,
                "min_gait_consistency": 0.7,
                "consistency_count": 10,
            },
            skill_requirements={
                "energy_efficiency": 0.6,
                "dynamic_balance": 0.7,
                "step_consistency": 0.7,
            },
            regression_thresholds={
                "max_failures": 10,
                "min_success_rate": 0.5,
                "stagnation_episodes": 15,
            },
        )

        # FASE 4: CORRIDA AVANÇADA E EFICIENTE
        phase4 = GaitPhaseConfig(
            name="corrida_avancada_eficiente",
            target_speed=2.0,
            enabled_components=[
                "progress",
                "stability_roll",
                "stability_pitch",
                "foot_clearance",
                "pitch_forward_bonus",
                "success_bonus",
                "distance_bonus",
                "gait_pattern_cross",
                "center_bonus",
                "warning_penalty",
                "effort_square_penalty",
                "jerk_penalty",
                "direction_change_penalty",
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
                "direction_change_penalty": 0.005,
            },
            phase_duration=20,
            transition_conditions={
                "min_success_rate": 0.9,
                "min_avg_distance": 9.0,
                "min_avg_speed": 2.0,
                "max_avg_roll": 0.1,
                "min_energy_efficiency": 0.8,
                "min_gait_consistency": 0.85,
                "consistency_count": 18,
            },
            skill_requirements={"energy_efficiency": 0.8, "high_speed_maneuver": 0.75, "adaptive_gait": 0.7},
            regression_thresholds={"max_failures": 5, "min_success_rate": 0.8, "stagnation_episodes": 8},
        )

        self.phases = [phase0, phase1, phase2, phase3, phase4]

    def update_phase(self, episode_results: Dict) -> PhaseTransitionResult:
        """
        Atualiza a fase atual com validação robusta e fallback adaptativo
        """    
        episode_duration = episode_results.get('duration', 0)
        episode_distance = episode_results.get('distance', 0)
        episode_steps = episode_results.get('steps', 0)

        is_valid_episode = (
            episode_duration >= 0.1 and     
            episode_steps >= 5 and           
            episode_distance >= 0            
        )

        if not is_valid_episode:
            self.logger.debug(f"Episódio ignorado - dados insuficientes: duration={episode_duration:.2f}s, steps={episode_steps}, distance={episode_distance:.2f}m")
            return PhaseTransitionResult.FAILURE

        self.episodes_in_phase += 1

        if self.current_phase == 0 and self.episodes_in_phase % 50 == 0:
            self.logger.warning("🔄 FRESH START - Reset automático a cada 50 episódios")
            self.progression_history = self.progression_history[-5:] if len(self.progression_history) > 5 else self.progression_history
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.stagnation_counter = 0
        
        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS 

        enhanced_results = self._enhance_episode_results(episode_results)

        self.performance_history.append(enhanced_results) 
        self.progression_history.append(enhanced_results) 

        if len(self.progression_history) > self.max_progression_history:
            removed = self.progression_history.pop(0)

        self._update_success_failure_counters(enhanced_results)

        if len(self.progression_history) < 3:
            self.logger.debug(f"Aguardando histórico suficiente: {len(self.progression_history)}/3")
            return PhaseTransitionResult.FAILURE

        regression_result = self._check_regression_or_stagnation()
        if regression_result != PhaseTransitionResult.SUCCESS:
            return regression_result

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

        if (self.current_phase == 0 and 
            self.episodes_in_phase > current_phase_config.phase_duration * 3 and  
            self._calculate_success_rate() < 0.1):  
            self.logger.warning("🎯 RESET ESTRATÉGICO - Performance muito baixa por muito tempo")
            self.progression_history = []  
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            return False
    
        if not self._meets_minimum_requirements():
            return False
        if self._has_recent_instability():
            return False

        success_rate = self._calculate_success_rate()
        avg_distance = self._calculate_average_distance()
        avg_roll = self._calculate_average_roll()
        avg_steps = np.mean([r.get("steps", 0) for r in self.progression_history])  

        basic_conditions_met = (
            success_rate >= current_phase_config.transition_conditions["min_success_rate"]
            and avg_distance >= current_phase_config.transition_conditions["min_avg_distance"]
            and avg_roll <= current_phase_config.transition_conditions["max_avg_roll"]
            and avg_steps >= current_phase_config.transition_conditions.get("min_avg_steps", 5)
        )

        if not basic_conditions_met:
            return False

        additional_conditions_met = self._check_performance_consistency() and self._validate_phase_skills()

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

        keep_episodes = min(10, len(self.progression_history))
        self.progression_history = self.progression_history[-keep_episodes:]
        new_phase_config = self.phases[self.current_phase]

        self.logger.info(f"🎉 AVANÇO DE FASE: {self.phases[old_phase].name} → {new_phase_config.name}")

        self._apply_phase_config()
        return PhaseTransitionResult.SUCCESS

    def _enhance_episode_results(self, episode_results: Dict) -> Dict:
        """Adiciona métricas calculadas aos resultados do episódio"""
        enhanced = episode_results.copy()

        if "distance" in episode_results and "duration" in episode_results:
            enhanced["speed"] = episode_results["distance"] / max(episode_results["duration"], 0.1)

        episode_steps = episode_results.get("steps", 0)
        episode_distance = episode_results.get("distance", 0)

        if episode_distance > 0.3 and episode_steps > 10:
            enhanced["alternating_score"] = 0.3  
        elif episode_distance > 0.1:
            enhanced["alternating_score"] = 0.1
        else:
            enhanced["alternating_score"] = 0.0

        enhanced["gait_pattern_score"] = episode_results.get("gait_pattern_score", 0.3)

        current_phase = self.current_phase
        episode_distance = episode_results.get("distance", 0)
        episode_steps = episode_results.get("steps", 0)
        episode_roll = abs(episode_results.get("roll", 0))
        
        if current_phase == 0:
            phase_success = (
                episode_roll < 0.6 and          
                episode_results.get("imu_z", 0.8) > 0.6 and  
                episode_steps > 3 and           
                episode_distance > 0.05           
            )
        elif current_phase == 1:  
            phase_success = (
                episode_distance > 0.5 and          
                episode_steps > 8 and               
                episode_roll < 0.6)                     
        elif current_phase == 2:  
            phase_success = (episode_distance > 2.0)  
        elif current_phase == 3:  
            phase_success = (episode_distance > 5.0)  
        else: 
            phase_success = episode_results.get("success", False)

        enhanced["phase_success"] = phase_success
        if current_phase == 0:
            stability_score = 1.0 - min(episode_roll / 1.0, 1.0)
            progress_score = min(episode_distance / 0.5, 1.0)  
            enhanced["gait_initiation_score"] = (stability_score * 0.6 + progress_score * 0.4)

        return enhanced

    def _update_success_failure_counters(self, episode_results: Dict):
        """Atualiza contadores de sucesso e fracasso"""
        phase_success = episode_results.get("phase_success", False)
        if phase_success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= 5 and self.current_phase == 0:
                successful_episodes = [r for r in self.progression_history if r.get("phase_success", False)]
                if len(successful_episodes) > 8:
                    self.progression_history = successful_episodes[-8:]  
                    self.logger.info("🧹 Reset parcial - mantendo apenas episódios bem-sucedidos")
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        current_avg_distance = self._calculate_average_distance()
        if current_avg_distance > self.last_avg_distance + 0.05:  
            self.stagnation_counter = 0  
        elif abs(current_avg_distance - self.last_avg_distance) < 0.02:  
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1) 
        
        self.last_avg_distance = current_avg_distance

    def _check_regression_or_stagnation(self) -> PhaseTransitionResult:
        """Sistema de regressão mais inteligente e menos agressivo"""
        if self.current_phase == 0:
            if self.consecutive_failures >= 30:
                self.logger.warning("🚨 RESET RÁPIDO - Limpando histórico contaminado")
                keep_episodes = max(1, len(self.progression_history) // 3)
                self.progression_history = self.progression_history[-keep_episodes:]
                self.consecutive_failures = 0
                self.consecutive_successes = 0
                self.stagnation_counter = 0
            return PhaseTransitionResult.SUCCESS

        current_phase_config = self.phases[self.current_phase]
        regression_thresholds = current_phase_config.regression_thresholds

        # Verificação de regressão mais tolerante
        if self.consecutive_failures >= regression_thresholds["max_failures"] * 1.5:
            self.regression_count += 1
            # Só regride após múltiplas falhas e tempo suficiente na fase
            if self._should_regress_phase():
                return self._regress_to_previous_phase()
            return PhaseTransitionResult.FAILURE

        # Estagnação mais tolerante
        if self.stagnation_counter >= regression_thresholds["stagnation_episodes"] * 1.5:
            self.logger.warning("Estagnação detectada - considerando regressão")
            return PhaseTransitionResult.STAGNATION

        # Sucesso muito baixo por muito tempo
        success_rate = self._calculate_success_rate()
        if (success_rate < regression_thresholds["min_success_rate"] * 0.7 and 
            self.episodes_in_phase > current_phase_config.phase_duration * 1.5):
            self.regression_count += 1
            if self._should_regress_phase():
                return self._regress_to_previous_phase()

        return PhaseTransitionResult.SUCCESS

    def _should_regress_phase(self) -> bool:
        """Decisão de regressão mais conservadora"""
        if self.current_phase == 0:
            return False

        # Só regride se estiver claramente travado
        should_regress = (
            self.current_phase > 0 and 
            self.regression_count >= 4 and  # Mais regressões necessárias
            self.episodes_in_phase > self.phases[self.current_phase].phase_duration * 2.5  # Mais tempo
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

        keep_episodes = min(10, len(self.progression_history))
        self.progression_history = self.progression_history[-keep_episodes:]
        self.logger.warning(f"🔄 REGRESSÃO DE FASE: {self.phases[old_phase].name} → {self.phases[self.current_phase].name}")

        self._apply_phase_config()
        return PhaseTransitionResult.REGRESSION

    def _meets_minimum_requirements(self) -> bool:
        """Verifica requisitos mínimos para transição"""
        current_phase_config = self.phases[self.current_phase]
        has_minimum_duration = self.episodes_in_phase >= current_phase_config.phase_duration

        has_sufficient_history = len(self.progression_history) >= 5

        if not has_minimum_duration:
            self.logger.debug(f"Aguardando duração mínima: {self.episodes_in_phase}/{current_phase_config.phase_duration}")

        if not has_sufficient_history:
            self.logger.debug(f"Aguardando histórico suficiente: {len(self.progression_history)}/5")

        return has_minimum_duration and has_sufficient_history

    def _has_recent_instability(self) -> bool:
        """Detecta instabilidade recente baseada em roll alto"""
        if len(self.progression_history) < 5:
            return True  

        recent_rolls = [abs(r.get("roll", 0)) for r in self.progression_history[-5:]]
        avg_recent_roll = np.mean(recent_rolls)

        return avg_recent_roll > 0.6

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
        """Cálculo de habilidades CORRIGIDO - foco em estabilidade"""
        if len(self.progression_history) < 3:
            return {
                "basic_balance": 0.3,
                "postural_stability": 0.2, 
                "gait_initiation": 0.1
            }

        recent_results = self.progression_history[-5:]  

        avg_roll = np.mean([abs(r.get("roll", 0)) for r in recent_results])
        roll_stability = 1.0 - min(avg_roll / 1.0, 1.0)
        z_positions = [r.get("imu_z", 0.8) for r in recent_results]  
        avg_height = np.mean(z_positions)
        height_stability = min(avg_height / 0.8, 1.0)  
        balance_score = (roll_stability * 0.7 + height_stability * 0.3)

        gait_scores = [r.get("gait_initiation_score", 0) for r in recent_results]
        if gait_scores:
            gait_initiation = np.mean(gait_scores)
        else:
            successes = sum(1 for r in recent_results if r.get("phase_success", False))
            gait_initiation = successes / len(recent_results) if recent_results else 0.1

        all_skills = {
            "basic_balance": min(balance_score * 1.1, 1.0), 
            "postural_stability": min(roll_stability * 1.2, 1.0), 
            "gait_initiation": gait_initiation,
        }

        return all_skills

    def _calculate_energy_efficiency(self, recent_results: List[Dict]) -> float:
        """Calcula eficiência energética de forma mais realista"""
        if not recent_results:
            return 0.5

        distances = [r.get("distance", 0) for r in recent_results]
        energies = [r.get("energy_used", 1) for r in recent_results]

        efficiencies = []
        for dist, energy in zip(distances, energies):
            if energy > 0.1:  # Evitar divisão por zero
                efficiency = dist / energy
                efficiencies.append(min(efficiency / 2.0, 1.0))  # Normalizado

        return np.mean(efficiencies) if efficiencies else 0.5

    def _calculate_enhanced_balance_recovery(self, recent_results: List[Dict]) -> float:
        """Calcula capacidade de recuperar equilíbrio de forma mais robusta"""
        if len(recent_results) < 3:
            return 0.5

        recovery_events = 0
        total_critical_events = 0

        for i in range(1, len(recent_results)):
            prev_roll = abs(recent_results[i - 1].get("roll", 0))
            curr_roll = abs(recent_results[i].get("roll", 0))

            if prev_roll > 0.5:
                total_critical_events += 1
                if curr_roll < 0.3 and curr_roll < prev_roll * 0.7:
                    recovery_events += 1

        if total_critical_events == 0:
            return 0.8

        return recovery_events / total_critical_events

    def _calculate_balance_recovery_score(self, recent_results: List[Dict]) -> float:
        """Calcula capacidade de recuperar equilíbrio"""
        if len(recent_results) < 3:
            return 0.5

        recovery_events = 0
        total_critical_events = 0

        for i in range(1, len(recent_results)):
            prev_roll = abs(recent_results[i - 1].get("roll", 0))
            curr_roll = abs(recent_results[i].get("roll", 0))

            if prev_roll > 0.5:
                total_critical_events += 1
                if curr_roll < 0.3 and curr_roll < prev_roll * 0.7:
                    recovery_events += 1

        if total_critical_events == 0:
            return 0.8

        return recovery_events / total_critical_events

    def _calculate_propulsion_efficiency(self) -> float:
        """Calcula eficiência propulsiva baseada no progression_history"""
        if not self.progression_history:  
            return 0.0

        speeds = [r.get("speed", 0) for r in self.progression_history[-10:]]  
        efforts = [r.get("energy_used", 1) for r in self.progression_history[-10:]]  

        efficiencies = [s / max(e, 0.1) for s, e in zip(speeds, efforts)]
        return np.mean(efficiencies) / 2.0  

    def _calculate_consistency_score(self) -> float:
        """Calcula score de consistência geral baseado no progression_history"""
        if len(self.progression_history) < 5:  
            return 0.0

        distances = [r.get("distance", 0) for r in self.progression_history[-10:]]  
        speeds = [r.get("speed", 0) for r in self.progression_history[-10:]]  

        distance_consistency = 1.0 - min(np.std(distances) / 2.0, 1.0)
        speed_consistency = 1.0 - min(np.std(speeds) / 1.0, 1.0)

        return (distance_consistency + speed_consistency) / 2.0

    def _apply_phase_config(self):
        """Aplica a configuração da fase atual ao sistema de recompensa"""
        current_phase = self.phases[self.current_phase]

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
        """Retorna status detalhado com ambos históricos"""
        current_phase = self.phases[self.current_phase]
        success_rate = self._calculate_success_rate()
        avg_distance = self._calculate_average_distance()
        avg_speed = self._calculate_average_speed()
        avg_roll = self._calculate_average_roll()

        status = {
            "current_phase": current_phase.name,
            "phase_index": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "target_speed": current_phase.target_speed,
            # Métricas de performance
            "performance_metrics": {
                "success_rate": success_rate,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "avg_roll": avg_roll,
                "progression_history_size": len(self.progression_history),  
                "full_history_size": len(self.performance_history)         
            },
            "success_rate": success_rate,
            "avg_distance": avg_distance,
            "avg_speed": avg_speed,
            "avg_roll": avg_roll,
            # Históricos
            "performance_history_size": len(self.performance_history),
            "progression_history_size": len(self.progression_history),
            # Contadores
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "regression_count": self.regression_count,
            # Requisitos da fase atual
            "requirements": {
                "min_success_rate": current_phase.transition_conditions.get("min_success_rate", 0),
                "min_avg_distance": current_phase.transition_conditions.get("min_avg_distance", 0),
                "max_avg_roll": current_phase.transition_conditions.get("max_avg_roll", 0),
                "min_avg_steps": current_phase.transition_conditions.get("min_avg_steps", 0),
            },
        }

        return status

    def _calculate_success_rate(self) -> float:
        """Calcula taxa de sucesso baseada no histórico de progressão"""
        if not self.progression_history:
            return 0.0

        if self.current_phase == 0:
            recent_history = self.progression_history[-10:] 
        else:
            recent_history = self.progression_history
    
        total_episodes = len(recent_history)
        successes = sum(1 for result in recent_history if result.get("phase_success", False))

        success_rate = successes / total_episodes if total_episodes > 0 else 0.0

        return success_rate  

    def _calculate_average_distance(self) -> float:
        if not self.progression_history:  
            return 0.0
        
        if self.current_phase == 0:
            recent_history = self.progression_history[-10:]  
        else:
            recent_history = self.progression_history
            
        return np.mean([r.get("distance", 0) for r in recent_history])  

    def _calculate_average_speed(self) -> float:
        if not self.progression_history:  
            return 0.0
        
        if self.current_phase == 0:
            recent_history = self.progression_history[-10:]  
        else:
            recent_history = self.progression_history

        return np.mean([r.get("speed", 0) for r in recent_history])  

    def _calculate_average_roll(self) -> float:
        if not self.progression_history:
            return 0.0
        
        if self.current_phase == 0:
            recent_history = self.progression_history[-10:]  
        else:
            recent_history = self.progression_history
    
        return np.mean([abs(r.get("roll", 0)) for r in recent_history])

    def _check_performance_consistency(self) -> bool:
        if len(self.progression_history) < 5:
            return False
        recent_successes = sum(1 for r in self.progression_history[-5:] if r.get("phase_success", False))  
        return recent_successes >= 2  