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

        self.transition_episodes = 0
        self.transition_active = False
        self.old_weights = None
        self.old_enabled_components = None
        self.transition_total_episodes = 8
        
        self._initialize_enhanced_gait_phases()

    def _initialize_enhanced_gait_phases(self):
        """Progressão GRADUAL com fases intermediárias"""

        # FASE 0: Estabilidade Postural
        phase0 = GaitPhaseConfig(
            name="estabilidade_postural",
            target_speed=0.2,
            enabled_components=["stability_roll", "stability_pitch", "center_bonus", "success_bonus"],
            component_weights={
                "stability_roll": 0.6,
                "stability_pitch": 0.3,
                "center_bonus": 0.08,
                "success_bonus": 0.02,
            },
            phase_duration=15,
            transition_conditions={
                "min_success_rate": 0.15,
                "min_avg_distance": 0.1,
                "max_avg_roll": 0.8,
                "min_avg_steps": 3,
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

        # FASE 1: Marcha Básica (0.3-0.5m)
        phase1 = GaitPhaseConfig(
            name="marcha_basica", 
            target_speed=0.4,
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "success_bonus", "center_bonus",
            ],
            component_weights={
                "progress": 0.35,
                "distance_bonus": 0.25,
                "stability_roll": 0.2,
                "stability_pitch": 0.12,
                "alternating_foot_contact": 0.05,
                "success_bonus": 0.02,
                "center_bonus": 0.01,
            },
            phase_duration=20,
            transition_conditions={
                "min_success_rate": 0.25,
                "min_avg_distance": 0.3,    
                "max_avg_roll": 0.7,        
                "min_avg_steps": 6,
                "min_avg_speed": 0.1,
            },
            skill_requirements={
                "basic_balance": 0.4,
                "postural_stability": 0.3,
                "step_consistency": 0.1,    
            },
            regression_thresholds={
                "max_failures": 30,
                "min_success_rate": 0.1,
                "stagnation_episodes": 50,
            },
        )

        # FASE 2: Marcha Lenta Estável (0.5-1.0m)
        phase2 = GaitPhaseConfig(
            name="marcha_lenta_estavel",
            target_speed=0.6,
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "success_bonus", "center_bonus",
            ],
            component_weights={
                "progress": 0.4,
                "distance_bonus": 0.3,
                "stability_roll": 0.15,
                "stability_pitch": 0.08,
                "alternating_foot_contact": 0.04,
                "gait_pattern_cross": 0.02,
                "success_bonus": 0.01,
            },
            phase_duration=25,
            transition_conditions={
                "min_success_rate": 0.3,
                "min_avg_distance": 0.5,    
                "max_avg_roll": 0.6,        
                "min_avg_steps": 8,
                "min_avg_speed": 0.2,
                "min_alternating_score": 0.2,
            },
            skill_requirements={
                "basic_balance": 0.5,
                "postural_stability": 0.4,
                "step_consistency": 0.2,
            },
            regression_thresholds={
                "max_failures": 25,
                "min_success_rate": 0.15,
                "stagnation_episodes": 40,
            },
        )

        # FASE 3: Marcha Confiante (1.0-2.0m)
        phase3 = GaitPhaseConfig(
            name="marcha_confiante",
            target_speed=0.8,
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "foot_clearance", "success_bonus",
                "effort_square_penalty", "center_bonus",
            ],
            component_weights={
                "progress": 0.4,
                "distance_bonus": 0.25,
                "stability_roll": 0.12,
                "stability_pitch": 0.08,
                "alternating_foot_contact": 0.05,
                "gait_pattern_cross": 0.04,
                "foot_clearance": 0.03,
                "success_bonus": 0.02,
                "effort_square_penalty": 0.005,
                "center_bonus": 0.005,
            },
            phase_duration=30,
            transition_conditions={
                "min_success_rate": 0.4,
                "min_avg_distance": 1.0,    
                "max_avg_roll": 0.5,       
                "min_avg_steps": 12,
                "min_avg_speed": 0.4,
                "min_alternating_score": 0.4,
                "min_gait_coordination": 0.3,
            },
            skill_requirements={
                "basic_balance": 0.6,
                "postural_stability": 0.5,
                "step_consistency": 0.3,
                "dynamic_balance": 0.4,
            },
            regression_thresholds={
                "max_failures": 20,
                "min_success_rate": 0.2,
                "stagnation_episodes": 30,
            },
        )

        # FASE 4: Marcha Rápida (2.0-4.0m)
        phase4 = GaitPhaseConfig(
            name="marcha_rapida",
            target_speed=1.2,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "alternating_foot_contact", "gait_pattern_cross",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "effort_square_penalty", 
                "y_axis_deviation_square_penalty", "jerk_penalty",
            ],
            component_weights={
                "progress": 0.35,
                "stability_roll": 0.15,
                "stability_pitch": 0.1,
                "alternating_foot_contact": 0.08,
                "gait_pattern_cross": 0.08,
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.05,
                "success_bonus": 0.03,
                "distance_bonus": 0.04,
                "effort_square_penalty": 0.008,
                "y_axis_deviation_square_penalty": 0.01,
                "jerk_penalty": 0.005,
            },
            phase_duration=35,
            transition_conditions={
                "min_success_rate": 0.5,
                "min_avg_distance": 2.0,    
                "min_avg_speed": 0.6,
                "max_avg_roll": 0.4,
                "min_propulsion_efficiency": 0.3,
                "min_gait_coordination": 0.5,
                "consistency_count": 5,
            },
            skill_requirements={
                "balance_recovery": 0.4,
                "propulsive_phase": 0.3,
                "dynamic_balance": 0.5,
                "step_consistency": 0.4,
            },
            regression_thresholds={
                "max_failures": 15,
                "min_success_rate": 0.3,
                "stagnation_episodes": 25,
            },
        )

        # FASE 5: Marcha Propulsiva (4.0-6.0m)
        phase5 = GaitPhaseConfig(
            name="marcha_propulsiva",
            target_speed=1.5,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "alternating_foot_contact", "gait_pattern_cross",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "effort_square_penalty", 
                "y_axis_deviation_square_penalty", "jerk_penalty",
            ],
            component_weights={
                "progress": 0.4,
                "stability_roll": 0.12,
                "stability_pitch": 0.08,
                "alternating_foot_contact": 0.07,
                "gait_pattern_cross": 0.07,
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.06,
                "success_bonus": 0.04,
                "distance_bonus": 0.05,
                "effort_square_penalty": 0.01,
                "y_axis_deviation_square_penalty": 0.012,
                "jerk_penalty": 0.006,
            },
            phase_duration=40,
            transition_conditions={
                "min_success_rate": 0.6,
                "min_avg_distance": 4.0,    
                "min_avg_speed": 0.8,
                "max_avg_roll": 0.3,
                "min_propulsion_efficiency": 0.4,
                "min_gait_coordination": 0.6,
                "consistency_count": 8,
            },
            skill_requirements={
                "balance_recovery": 0.5,
                "propulsive_phase": 0.4,
                "dynamic_balance": 0.6,
                "energy_efficiency": 0.4,
            },
            regression_thresholds={
                "max_failures": 12,
                "min_success_rate": 0.4,
                "stagnation_episodes": 20,
            },
        )

        # FASE 6: Marcha Eficiente (6.0-9.0m - OBJETIVO FINAL)
        phase6 = GaitPhaseConfig(
            name="marcha_eficiente",
            target_speed=1.8,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "gait_pattern_cross", "effort_square_penalty",
                "jerk_penalty", "center_bonus", "warning_penalty",
            ],
            component_weights={
                "progress": 0.45,
                "stability_roll": 0.1,
                "stability_pitch": 0.07,
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.07,
                "success_bonus": 0.05,
                "distance_bonus": 0.06,
                "gait_pattern_cross": 0.05,
                "effort_square_penalty": 0.012,
                "jerk_penalty": 0.008,
                "center_bonus": 0.02,
                "warning_penalty": 0.015,
            },
            phase_duration=25,
            transition_conditions={
                "min_success_rate": 0.7,
                "min_avg_distance": 6.0,    
                "min_avg_speed": 1.0,
                "max_avg_roll": 0.25,
                "min_energy_efficiency": 0.6,
                "min_gait_consistency": 0.7,
                "consistency_count": 10,
            },
            skill_requirements={
                "energy_efficiency": 0.6,
                "dynamic_balance": 0.7,
                "step_consistency": 0.6,
                "gait_coordination": 0.7,
            },
            regression_thresholds={
                "max_failures": 8,
                "min_success_rate": 0.5,
                "stagnation_episodes": 15,
            },
        )

        self.phases = [phase0, phase1, phase2, phase3, phase4, phase5, phase6]

    def update_phase(self, episode_results: Dict) -> PhaseTransitionResult:
        """
        Atualiza a fase atual com validação robusta e transição gradual
        """    
        if self.transition_active:
            return self._update_transition_progress()

        episode_duration = episode_results.get('duration', 0)
        episode_distance = episode_results.get('distance', 0)
        episode_steps = episode_results.get('steps', 0)

        is_valid_episode = (
            episode_duration >= 0.1 and     
            episode_steps >= 5 and           
            episode_distance >= 0            
        )

        if not is_valid_episode:
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
            return PhaseTransitionResult.FAILURE

        regression_result = self._check_regression_or_stagnation()
        if regression_result != PhaseTransitionResult.SUCCESS:
            return regression_result

        can_advance = self._check_phase_advancement()
        if can_advance:
            self.logger.info("CONDIÇÕES ATENDIDAS - Iniciando transição para próxima fase!")
            return self._start_gradual_transition()

        return PhaseTransitionResult.FAILURE

    def _check_phase_advancement(self) -> bool:
        """Verifica se todas as condições para avançar de fase foram atendidas"""
        if self.current_phase >= len(self.phases) - 1:
            return False

        current_phase_config = self.phases[self.current_phase]
        conditions = current_phase_config.transition_conditions
    
        if not self._meets_minimum_requirements():
            return False
        
        all_conditions_met = True

        if "min_success_rate" in conditions:
            min_success = conditions["min_success_rate"]
            current_success = self._calculate_success_rate()
            if current_success < min_success:
                all_conditions_met = False
    
        if "min_avg_distance" in conditions:
            min_distance = conditions["min_avg_distance"]
            current_avg_distance = self._calculate_average_distance()
            if current_avg_distance < min_distance:
                all_conditions_met = False

        if "max_avg_roll" in conditions:
            max_roll = conditions["max_avg_roll"]
            current_avg_roll = self._calculate_average_roll()
            if current_avg_roll > max_roll:
                all_conditions_met = False

        if "min_avg_speed" in conditions:
            min_speed = conditions["min_avg_speed"]
            current_avg_speed = self._calculate_average_speed()
            if current_avg_speed < min_speed:
                all_conditions_met = False

        if self.current_phase == 2:
            if "min_propulsion_efficiency" in conditions:
                min_propulsion = conditions["min_propulsion_efficiency"]
                propulsion = self._calculate_propulsion_efficiency()
                if propulsion < min_propulsion:
                    all_conditions_met = False

            if "min_gait_coordination" in conditions:
                min_coordination = conditions["min_gait_coordination"]
                coordination = self._calculate_gait_coordination()
                if coordination < min_coordination:
                    all_conditions_met = False

        return all_conditions_met

    def _enhance_episode_results(self, episode_results: Dict) -> Dict:
        """Adiciona métricas calculadas aos resultados do episódio"""
        enhanced = episode_results.copy()

        current_phase = self.current_phase
        if current_phase >= len(self.phases):
            enhanced["phase_success"] = False
            return enhanced
    
        current_config = self.phases[current_phase]
        conditions = current_config.transition_conditions

        episode_success = True

        episode_distance = episode_results.get("distance", 0)
        episode_roll = abs(episode_results.get("roll", 0))
        episode_steps = episode_results.get("steps", 0)
        episode_z = episode_results.get("imu_z", 0.8)
        episode_speed = episode_results.get("speed", 0)

        if "min_avg_distance" in conditions:
            min_distance = conditions["min_avg_distance"]
            if episode_distance < min_distance:
                episode_success = False

        if "max_avg_roll" in conditions:
            max_roll = conditions["max_avg_roll"]
            if episode_roll > max_roll:
                episode_success = False

        if "min_avg_steps" in conditions:
            min_steps = conditions["min_avg_steps"]
            if episode_steps < min_steps:
                episode_success = False

        if "min_avg_speed" in conditions:
            min_speed = conditions["min_avg_speed"]
            if episode_speed < min_speed:
                episode_success = False

        if current_phase == 0:
            if "min_height" in conditions:
                min_height = conditions["min_height"]
                if episode_z < min_height:
                    episode_success = False

            if "min_progress" in conditions:
                min_progress = conditions["min_progress"]
                if episode_distance < min_progress:
                    episode_success = False

        elif current_phase == 1:
            if "min_foot_alternation" in conditions:
                left_contact = episode_results.get("left_contact", False)
                right_contact = episode_results.get("right_contact", False)
                if not (left_contact or right_contact):  
                    episode_success = False

            if "min_contact_time" in conditions:
                if episode_steps < 5:  
                    episode_success = False

        else:
            if "min_foot_clearance" in conditions:
                min_clearance = conditions["min_foot_clearance"]
                left_clearance = episode_results.get("left_foot_height", 0)
                right_clearance = episode_results.get("right_foot_height", 0)
                if left_clearance < min_clearance and right_clearance < min_clearance:
                    episode_success = False

            if "max_energy_usage" in conditions:
                max_energy = conditions["max_energy_usage"]
                energy_used = episode_results.get("energy_used", 0)
                if energy_used > max_energy:
                    episode_success = False

        enhanced["phase_success"] = episode_success

        if current_phase == 0:
            enhanced["gait_initiation_score"] = 1.0 if episode_success else 0.0

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
            return PhaseTransitionResult.SUCCESS

        current_phase_config = self.phases[self.current_phase]
        regression_thresholds = current_phase_config.regression_thresholds

        success_rate = self._calculate_success_rate()

        if self.current_phase <= 2:
            if (self.consecutive_failures >= regression_thresholds["max_failures"] * 2 and
                success_rate < regression_thresholds["min_success_rate"] * 0.5):
                return self._regress_to_previous_phase()
        else:
            if (self.consecutive_failures >= regression_thresholds["max_failures"] and
                success_rate < regression_thresholds["min_success_rate"] * 0.7):
                return self._regress_to_previous_phase()

        if self.current_phase >= 3 and self.stagnation_counter >= regression_thresholds["stagnation_episodes"]:
            return PhaseTransitionResult.STAGNATION

        return PhaseTransitionResult.SUCCESS

    def _start_gradual_transition(self) -> PhaseTransitionResult:
        """Inicia transição gradual para próxima fase"""
        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS

        self.old_weights = {}
        self.old_enabled_components = {}
        
        current_config = self.phases[self.current_phase]
        for component_name in current_config.enabled_components:
            if component_name in self.reward_system.components:
                self.old_weights[component_name] = self.reward_system.components[component_name].weight
                self.old_enabled_components[component_name] = self.reward_system.components[component_name].enabled

        self.transition_active = True
        self.transition_episodes = 0
        self.transition_total_episodes = 10  

        self._smart_buffer_transition()
        
        return PhaseTransitionResult.SUCCESS
    
    def _update_transition_progress(self) -> PhaseTransitionResult:
        """Atualiza progresso da transição gradual"""
        self.transition_episodes += 1
        transition_progress = self.transition_episodes / self.transition_total_episodes
        self._apply_gradual_phase_config(transition_progress)
        
        if self.transition_episodes >= self.transition_total_episodes:
            return self._complete_phase_transition()
        
        return PhaseTransitionResult.SUCCESS
    
    def _apply_gradual_phase_config(self, progress: float):
        """Aplica configuração gradualmente durante transição"""
        current_phase_config = self.phases[self.current_phase]
        next_phase_config = self.phases[self.current_phase + 1]
        
        for component_name, component in self.reward_system.components.items():
            old_weight = self.old_weights.get(component_name, 0.0)
            new_weight = next_phase_config.component_weights.get(component_name, 0.0)
            interpolated_weight = old_weight * (1 - progress) + new_weight * progress
            component.weight = interpolated_weight
            old_enabled = self.old_enabled_components.get(component_name, False)
            new_enabled = component_name in next_phase_config.enabled_components
            component.enabled = old_enabled or new_enabled

    def _complete_phase_transition(self) -> PhaseTransitionResult:
        """Completa a transição para próxima fase"""
        old_phase = self.current_phase
        self.current_phase += 1
        self.episodes_in_phase = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.stagnation_counter = 0
        self.regression_count = 0
        
        self.transition_active = False
        self.transition_episodes = 0
        self.old_weights = None
        self.old_enabled_components = None
        
        self._apply_phase_config()
        
        keep_episodes = min(8, len(self.progression_history))
        self.progression_history = self.progression_history[-keep_episodes:]
        
        new_phase_config = self.phases[self.current_phase]
        self.logger.info(f"🎉 TRANSIÇÃO CONCLUÍDA: {self.phases[old_phase].name} → {new_phase_config.name}")
        
        return PhaseTransitionResult.SUCCESS
    
    def _smart_buffer_transition(self):
        """Transição inteligente do buffer preservando melhores experiências"""
        try:
            if hasattr(self.reward_system, 'agent') and self.reward_system.agent:
                if hasattr(self.reward_system.agent.model, 'replay_buffer'):
                    buffer = self.reward_system.agent.model.replay_buffer
                    buffer_size_before = len(buffer)
                    
                    if self.current_phase <= 1:
                        buffer.clear()
                    else:
                        preserved_count = self._preserve_high_value_experiences(buffer, preserve_ratio=0.3)
                    
                    self._refill_replay_buffer()
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na transição do buffer: {e}")
            self._clear_agent_replay_buffer()

    def _preserve_high_value_experiences(self, buffer, preserve_ratio=0.3) -> int:
        """Preserva as melhores experiências do buffer baseado em qualidade"""
        try:
            if not hasattr(buffer, 'buffer') or len(buffer) == 0:
                return 0
            
            experiences = []
            for i in range(len(buffer)):
                experience = buffer.buffer[i]
                quality_score = self._calculate_experience_quality(experience)
                experiences.append((quality_score, experience))
            
            experiences.sort(key=lambda x: x[0], reverse=True)
            
            preserve_count = int(len(experiences) * preserve_ratio)
            preserved_experiences = experiences[:preserve_count]
            
            buffer.clear()
            for quality_score, experience in preserved_experiences:
                buffer.add(*experience)
            
            return preserve_count
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao preservar experiências: {e}")
            return 0
        
    def _calculate_experience_quality(self, experience) -> float:
        """Calcula qualidade de uma experiência baseado em estabilidade e progresso"""
        try:
            obs, next_obs, action, reward, done, info = experience
            
            quality_score = 0.0
            
            quality_score += min(abs(reward) * 0.1, 1.0)
            
            if not done:
                quality_score += 0.5
            
            if hasattr(action, '__len__'):
                action_smoothness = 1.0 - min(np.std(action) * 2.0, 1.0)
                quality_score += action_smoothness * 0.3
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            return 0.5  
        
    def _regress_to_previous_phase(self) -> PhaseTransitionResult:
        """Regride para a fase anterior com transição gradual"""
        if self.current_phase == 0:
            return PhaseTransitionResult.FAILURE

        self.old_weights = {}
        self.old_enabled_components = {}
        
        current_config = self.phases[self.current_phase]
        for component_name in current_config.enabled_components:
            if component_name in self.reward_system.components:
                self.old_weights[component_name] = self.reward_system.components[component_name].weight
                self.old_enabled_components[component_name] = self.reward_system.components[component_name].enabled

        self.transition_active = True
        self.transition_episodes = 0
        self.transition_total_episodes = 8  
        self._smart_buffer_transition()

        old_phase_name = self.phases[self.current_phase].name
        new_phase_name = self.phases[self.current_phase - 1].name
        self.logger.warning(f"🔄 REGRESSÃO INICIADA: {old_phase_name} → {new_phase_name}")

        return PhaseTransitionResult.REGRESSION

    def _clear_agent_replay_buffer(self):
        """Limpar buffer de replay do agente quando as recompensas mudam"""
        try:
            if hasattr(self.reward_system, 'agent') and self.reward_system.agent:
                if hasattr(self.reward_system.agent.model, 'replay_buffer'):
                    buffer_size_before = len(self.reward_system.agent.model.replay_buffer)
                    self.reward_system.agent.model.replay_buffer.clear()
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível limpar buffer: {e}")
        
    def _refill_replay_buffer(self):
        """Preencher rapidamente o buffer após limpeza"""
        try:
            if hasattr(self.reward_system, 'agent') and self.reward_system.agent:
                agent = self.reward_system.agent
                target_size = agent.prefill_steps // 2  

                if hasattr(agent.model, 'replay_buffer'):
                    current_size = len(agent.model.replay_buffer)

                    if current_size < target_size:
                        obs, _ = self.reward_system.agent.env.reset()
                        steps_collected = 0

                        while steps_collected < target_size and not self.reward_system.agent.env.exit_value.value:
                            action = self.reward_system.agent.env.robot.get_example_action(
                                self.reward_system.agent.env.episode_steps * self.reward_system.agent.env.time_step_s
                            )

                            next_obs, reward, terminated, truncated, info = self.reward_system.agent.env.step(action)
                            done = terminated or truncated

                            agent.model.replay_buffer.add(
                                obs.flatten(), 
                                next_obs.flatten(), 
                                action, 
                                reward, 
                                done, 
                                [info]
                            )

                            obs = next_obs
                            steps_collected += 1

                            if done:
                                obs, _ = self.reward_system.agent.env.reset()

        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao preencher buffer: {e}")
        
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
        """Cálculo de habilidades"""

        if len(self.progression_history) < 2:
            return {
                "basic_balance": 0.3,
                "postural_stability": 0.2, 
                "gait_initiation": 0.1,
                "step_consistency": 0.1,
                "dynamic_balance": 0.1,
                "balance_recovery": 0.1,
                "propulsive_phase": 0.1,
                "energy_efficiency": 0.1,
                "gait_coordination": 0.1,
            }

        recent_results = self.progression_history[-8:]

        # MÉTRICAS BÁSICAS
        success_rate = self._calculate_success_rate()
        avg_roll = self._calculate_average_roll()
        avg_distance = self._calculate_average_distance()
        avg_speed = self._calculate_average_speed()

        # 1. BASIC BALANCE - estabilidade geral
        z_positions = [r.get("imu_z", 0.8) for r in recent_results]
        avg_height = np.mean(z_positions)
        height_stability = min(avg_height / 0.8, 1.0)  
        roll_stability = 1.0 - min(avg_roll / 1.0, 1.0)
        basic_balance = (roll_stability * 0.7 + height_stability * 0.3)

        # 2. POSTURAL STABILITY - controle postural
        roll_values = [abs(r.get("roll", 0)) for r in recent_results]
        if len(roll_values) > 1:
            roll_consistency = 1.0 - min(np.std(roll_values) / 0.5, 1.0)
        else:
            roll_consistency = 0.5
        postural_stability = (roll_stability * 0.6 + roll_consistency * 0.4)

        # 3. GAIT INITIATION - início da marcha
        gait_initiation = success_rate

        # 4. STEP CONSISTENCY - consistência de passos
        distances = [r.get("distance", 0) for r in recent_results]
        if len(distances) >= 3 and np.mean(distances) > 0.1:
            std_dev = np.std(distances)
            avg_dist = np.mean(distances)
            cv = std_dev / avg_dist if avg_dist > 0 else 1.0
            step_consistency = max(0.0, 1.0 - min(cv, 2.0) / 2.0)
        else:
            step_consistency = 0.1

        # 5. DYNAMIC BALANCE - equilíbrio dinâmico
        dynamic_balance = 1.0 - min(avg_roll / 0.6, 1.0)  # Mais exigente que basic_balance

        # 6. BALANCE RECOVERY - recuperação de equilíbrio
        balance_recovery = self._calculate_balance_recovery_score(recent_results)

        # 7. PROPULSIVE PHASE - fase propulsiva
        propulsion_efficiency = self._calculate_propulsion_efficiency()

        # 8. ENERGY EFFICIENCY - eficiência energética
        energy_efficiency = self._calculate_energy_efficiency(recent_results)

        # 9. GAIT COORDINATION - coordenação da marcha
        gait_coordination = self._calculate_gait_coordination(recent_results)

        all_skills = {
            "basic_balance": min(basic_balance, 1.0),
            "postural_stability": min(postural_stability, 1.0),
            "gait_initiation": gait_initiation,
            "step_consistency": step_consistency,
            "dynamic_balance": dynamic_balance,
            "balance_recovery": balance_recovery,
            "propulsive_phase": propulsion_efficiency,
            "energy_efficiency": energy_efficiency,
            "gait_coordination": gait_coordination,
        }

        return all_skills

    def _calculate_energy_efficiency(self, recent_results: List[Dict]) -> float:
        """Calcula eficiência energética de forma mais realista"""
        if not recent_results:
            return 0.5

        efficiencies = []
        for result in recent_results:
            distance = result.get("distance", 0)
            energy = result.get("energy_used", 1.0)
            if energy > 0.1:  # Evitar divisão por zero
                efficiency = distance / energy
                efficiencies.append(min(efficiency / 2.0, 1.0))
        
        return np.mean(efficiencies) if efficiencies else 0.5

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

        return min(recovery_events / total_critical_events, 1.0)

    def _calculate_gait_coordination(self, recent_results: List[Dict]) -> float:
        """Calcular coordenação de marcha """
        if not recent_results:
            return 0.3

        coordination_scores = []
        for result in recent_results:
            left_contact = result.get("left_contact", False)
            right_contact = result.get("right_contact", False)
            alternation = 1.0 if left_contact != right_contact else 0.0

            cross_pattern = result.get("gait_pattern_score", 0.0)

            coordination = (alternation * 0.6 + cross_pattern * 0.4)
            coordination_scores.append(coordination)

        return np.mean(coordination_scores) if coordination_scores else 0.3

    def _calculate_propulsion_efficiency(self) -> float:
        """Calcular eficiência propulsiva para Fase 2"""
        if not self.progression_history:
            return 0.3

        recent_results = self.progression_history[-10:]
        speeds = [r.get("speed", 0) for r in recent_results]
        efforts = [r.get("energy_used", 1) for r in recent_results]

        efficiencies = []
        for speed, effort in zip(speeds, efforts):
            if effort > 0.1:  
                efficiency = speed / effort
                efficiencies.append(min(efficiency / 2.0, 1.0))

        return np.mean(efficiencies) if efficiencies else 0.3 

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
            # Transição
            "transition_active": self.transition_active,
            "transition_episodes": self.transition_episodes,
            "transition_total_episodes": self.transition_total_episodes,
            "transition_progress": self.transition_episodes / self.transition_total_episodes if self.transition_active else 1.0,
        }

        return status

    def _calculate_success_rate(self) -> float:
        """Calcula taxa de sucesso baseada no histórico de progressão"""
        if not self.progression_history:
            return 0.0
        successes = sum(1 for r in self.progression_history if r.get("phase_success", False))

        return successes / len(self.progression_history)

    def _calculate_average_distance(self) -> float:
        if not self.progression_history:  
            return 0.0
            
        return np.mean([r.get("distance", 0) for r in self.progression_history])

    def _calculate_average_speed(self) -> float:
        if not self.progression_history:  
            return 0.0
        
        return np.mean([r.get("speed", 0) for r in self.progression_history])

    def _calculate_average_roll(self) -> float:
        if not self.progression_history:
            return 0.0
        
        return np.mean([abs(r.get("roll", 0)) for r in self.progression_history])

    def _check_performance_consistency(self) -> bool:
        if len(self.progression_history) < 5:
            return False
        recent_successes = sum(1 for r in self.progression_history[-5:] if r.get("phase_success", False))  
        return recent_successes >= 2  