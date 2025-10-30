# dpg_gait_phases.py
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
        self.max_history_size = 50
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
            phase_duration=15,  
            transition_conditions={
                "min_success_rate": 0.3,       
                "min_avg_distance": 0.5,       
                "max_avg_roll": 0.5,          
                "min_avg_steps": 30,           
                "min_alternating_score": 0.3,  
                "consistency_count": 3        
            },
            skill_requirements={
                "balance_recovery": 0.7,       
                "postural_stability": 0.8,     
                "gait_initiation": 0.6        
            },
            regression_thresholds={
                "max_failures": 999,           
                "min_success_rate": 0.05,      
                "stagnation_episodes": 25     
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
                "rhythmic_gait": 0.7,          
                "lateral_stability": 0.75,     
                "foot_clearance_control": 0.6  
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
                "propulsive_phase": 0.7,      
                "dynamic_balance": 0.8,        
                "rhythm_consistency": 0.75     
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
        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS  # Fase final alcançada
            
        self.episodes_in_phase += 1
        
        # Adicionar resultado ao histórico
        enhanced_results = self._enhance_episode_results(episode_results)
        self.performance_history.append(enhanced_results)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
            
        # Atualizar contadores de sucesso/fracasso
        self._update_success_failure_counters(enhanced_results)
        
        # Verificar regressão ou estagnação
        regression_result = self._check_regression_or_stagnation()
        if regression_result != PhaseTransitionResult.SUCCESS:
            return regression_result
            
        # Verificar condições mínimas para transição
        if not self._meets_minimum_requirements():
            return PhaseTransitionResult.FAILURE
            
        # Validar habilidades específicas da fase
        if not self._validate_phase_skills():
            return PhaseTransitionResult.FAILURE
            
        # Verificar consistência de desempenho
        if not self._check_performance_consistency():
            return PhaseTransitionResult.FAILURE
            
        # TODAS as condições atendidas - transicionar
        old_phase = self.current_phase
        self.current_phase += 1
        self.episodes_in_phase = 0
        self.consecutive_successes = 0
        self.performance_history = []
        self.skill_assessment_history = []
        
        self.logger.info(f"TRANSIÇÃO DE FASE: {self.phases[old_phase].name} → {self.phases[self.current_phase].name}")
        self.logger.info(f"NOVO ALVO: {self.phases[self.current_phase].target_speed} m/s")
        
        self._apply_phase_config()
        return PhaseTransitionResult.SUCCESS
        
    def _enhance_episode_results(self, episode_results: Dict) -> Dict:
        """Adiciona métricas calculadas aos resultados do episódio"""
        enhanced = episode_results.copy()
        
        # Calcular métricas adicionais
        if "distance" in episode_results and "duration" in episode_results:
            enhanced["speed"] = episode_results["distance"] / max(episode_results["duration"], 0.1)
            enhanced["efficiency"] = episode_results["distance"] / max(episode_results.get("energy_used", 1), 1)
        
        # Score de padrão alternado (simplificado)
        left_contact = episode_results.get("left_contact", 0)
        right_contact = episode_results.get("right_contact", 0)
        enhanced["alternating_score"] = 1.0 if left_contact != right_contact else 0.0
        
        # Score de coordenação
        enhanced["coordination_score"] = episode_results.get("gait_pattern_score", 0.5)
        
        return enhanced
        
    def _update_success_failure_counters(self, episode_results: Dict):
        """Atualiza contadores de sucesso e fracasso"""
        if episode_results.get("success", False):
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

        # PERMITIR MAIS TENTATIVAS NA FASE 0
        if self.current_phase == 0:
            # Na fase inicial, ser mais tolerante com falhas
            if self.consecutive_failures >= 15:  # Aumentar limite
                return PhaseTransitionResult.FAILURE
            return PhaseTransitionResult.SUCCESS

        # Relaxar outros thresholds
        if self.consecutive_failures >= regression_thresholds["max_failures"] * 1.5:  # Aumentar em 50%
            self.regression_count += 1
            if self._should_regress_phase():
                return self._regress_to_previous_phase()
            return PhaseTransitionResult.FAILURE

        # Relaxar threshold de estagnação
        if self.stagnation_counter >= regression_thresholds["stagnation_episodes"] * 1.5:
            return PhaseTransitionResult.STAGNATION

        # Relaxar taxa de sucesso mínima
        success_rate = self._calculate_success_rate()
        if success_rate < regression_thresholds["min_success_rate"] * 0.8:  # Reduzir para 80%
            self.regression_count += 1
            if self._should_regress_phase():
                return self._regress_to_previous_phase()
            return PhaseTransitionResult.SUCCESS

        return PhaseTransitionResult.SUCCESS
        
    def _should_regress_phase(self) -> bool:
        """Decide se deve regredir para fase anterior"""
        # Só regredir se não for a fase inicial e tiver múltiplas regressões
        if self.current_phase == 0:
            return False
    
        return (self.current_phase > 0 and 
                self.regression_count >= 2 and 
                self.episodes_in_phase > self.phases[self.current_phase].phase_duration * 1.5)
        
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
        
        self.logger.warning(f"REGRESSÃO DE FASE: {self.phases[old_phase].name} → {self.phases[self.current_phase].name}")
        
        self._apply_phase_config()
        return PhaseTransitionResult.REGRESSION
        
    def _meets_minimum_requirements(self) -> bool:
        """Verifica requisitos mínimos para transição"""
        if self.episodes_in_phase < self.phases[self.current_phase].phase_duration:
            return False
            
        if len(self.performance_history) < 10:  # Mínimo de dados para análise
            return False
            
        return True
        
    def _validate_phase_skills(self) -> bool:
        """Valida se habilidades específicas da fase foram adquiridas"""
        current_phase = self.phases[self.current_phase]
        skill_requirements = current_phase.skill_requirements
        
        # Avaliar cada habilidade requerida
        skill_scores = self._assess_phase_skills()
        
        for skill, required_score in skill_requirements.items():
            if skill_scores.get(skill, 0) < required_score:
                self.logger.debug(f"Habilidade '{skill}' insuficiente: {skill_scores.get(skill, 0):.2f} < {required_score}")
                return False
                
        return True
        
    def _assess_phase_skills(self) -> Dict[str, float]:
        """Avalia habilidades específicas baseadas no histórico"""
        if len(self.performance_history) < 5:
            return {}
            
        recent_results = self.performance_history[-10:]
        
        skill_scores = {
            # Habilidades gerais
            "balance_recovery": self._calculate_balance_recovery_score(recent_results),
            "postural_stability": 1.0 - min(self._calculate_average_roll() / 0.5, 1.0),
            "gait_initiation": self._calculate_success_rate(),
            
            # Habilidades de marcha
            "rhythmic_gait": np.mean([r.get("alternating_score", 0) for r in recent_results]),
            "lateral_stability": 1.0 - min(np.mean([abs(r.get("imu_y", 0)) for r in recent_results]) / 0.3, 1.0),
            "foot_clearance_control": np.mean([r.get("clearance_score", 0.5) for r in recent_results]),
            
            # Habilidades avançadas
            "propulsive_phase": self._calculate_propulsion_efficiency(),
            "dynamic_balance": 1.0 - min(self._calculate_average_roll() / 0.3, 1.0),
            "rhythm_consistency": 1.0 - np.std([r.get("speed", 0) for r in recent_results]) / 2.0,
            
            # Habilidades de corrida
            "flight_phase_control": np.mean([r.get("flight_quality", 0.5) for r in recent_results]),
            "impact_absorption": 1.0 - min(np.mean([r.get("impact_force", 0) for r in recent_results]) / 1000, 1.0),
            "high_speed_stability": 1.0 - min(self._calculate_average_roll() / 0.2, 1.0),
            
            # Habilidades de eficiência
            "energy_efficiency": np.mean([r.get("efficiency", 0) for r in recent_results]),
            "high_speed_maneuver": min(self._calculate_average_speed() / 2.0, 1.0),
            "adaptive_gait": self._calculate_consistency_score()
        }
        
        return skill_scores
        
    def _check_performance_consistency(self) -> bool:
        """Verifica consistência do desempenho"""
        current_phase = self.phases[self.current_phase]
        required_consistency = current_phase.transition_conditions.get("consistency_count", 5)
        
        if len(self.performance_history) < required_consistency:
            return False
            
        # Verificar últimos 'n' episódios
        recent_results = self.performance_history[-required_consistency:]
        
        # Todos devem ter sucesso OU mostrar progresso consistente
        successful_episodes = sum(1 for r in recent_results if r.get("success", False))
        consistency_ratio = successful_episodes / len(recent_results)
        
        required_success_rate = current_phase.transition_conditions["min_success_rate"]
        return consistency_ratio >= required_success_rate
        
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
            
            if prev_roll > 0.3 and curr_roll < 0.2:  # Recuperou de instabilidade
                recovery_events += 1
            total_events += 1
            
        return recovery_events / max(total_events, 1)
        
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
                
        self.logger.info(f"Fase '{current_phase.name}' aplicada - Alvo: {current_phase.target_speed} m/s")
        
        # Log detalhado
        active_components = [(name, self.reward_system.components[name].weight) 
                           for name in current_phase.enabled_components]
        self.logger.info(f"Componentes ativos: {active_components}")
        
    def get_current_speed_target(self) -> float:
        """Retorna a velocidade alvo da fase atual"""
        return self.phases[self.current_phase].target_speed
        
    def get_detailed_status(self) -> Dict:
        """Retorna status detalhado com métricas de progresso"""
        current_phase = self.phases[self.current_phase]
        
        if len(self.performance_history) >= 5:
            skill_scores = self._assess_phase_skills()
            progress_metrics = self._calculate_progress_metrics()
        else:
            skill_scores = {}
            progress_metrics = {}
            
        return {
            "current_phase": current_phase.name,
            "phase_index": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "target_speed": current_phase.target_speed,
            "total_phases": len(self.phases),
            "phase_progress": f"{self.current_phase + 1}/{len(self.phases)}",
            "performance_metrics": progress_metrics,
            "skill_assessment": skill_scores,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "regression_count": self.regression_count
        }
        
    def _calculate_progress_metrics(self) -> Dict:
        """Calcula métricas detalhadas de progresso"""
        return {
            "success_rate": self._calculate_success_rate(),
            "avg_distance": self._calculate_average_distance(),
            "avg_speed": self._calculate_average_speed(),
            "avg_roll": self._calculate_average_roll(),
            "stability_score": 1.0 - min(self._calculate_average_roll() / 0.5, 1.0),
            "efficiency_score": np.mean([r.get("efficiency", 0) for r in self.performance_history[-5:]]),
            "consistency_score": self._calculate_consistency_score()
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
        successes = sum(1 for result in self.performance_history if result.get("success", False))
        return successes / len(self.performance_history)
        
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