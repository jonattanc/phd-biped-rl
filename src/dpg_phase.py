# dpg_phase.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PhaseTransitionResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure" 
    REGRESSION = "regression"
    STAGNATION = "stagnation"


@dataclass
class PhaseConfig:
    """Configuração de uma fase individual"""
    name: str
    target_speed: float
    enabled_components: List[str]
    component_weights: Dict[str, float]
    min_episodes: int
    transition_conditions: Dict[str, float]
    focus_skills: List[str]


class PhaseManager:
    """
    Gerenciador especializado em fases e transições
    """
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.phases = self._initialize_phases()
        
        # Histórico e métricas
        self.performance_history = []
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.stagnation_counter = 0
    
    def _initialize_phases(self):
        """Inicializa as fases do DPG - versão simplificada"""
        return [
            # Fase 0: Estabilidade Inicial
            PhaseConfig(
                name="estabilidade_inicial",
                target_speed=0.3,
                enabled_components=["stability", "basic_progress", "posture"],
                component_weights={"stability": 0.6, "basic_progress": 0.3, "posture": 0.1},
                min_episodes=10,
                transition_conditions={
                    "min_success_rate": 0.4,
                    "min_avg_distance": 0.2,
                    "max_avg_roll": 1.0,
                    "min_avg_steps": 3
                },
                focus_skills=["basic_balance", "postural_stability"]
            ),
            
            # Fase 1: Marcha Básica
            PhaseConfig(
                name="marcha_basica", 
                target_speed=0.8,
                enabled_components=["velocity", "stability", "phase_angles", "propulsion"],
                component_weights={"velocity": 0.3, "stability": 0.25, "phase_angles": 0.25, "propulsion": 0.2},
                min_episodes=15,
                transition_conditions={
                    "min_success_rate": 0.5,
                    "min_avg_distance": 0.8, 
                    "max_avg_roll": 0.7,
                    "min_avg_speed": 0.2,
                    "min_alternating_score": 0.3
                },
                focus_skills=["step_consistency", "dynamic_balance"]
            ),
            
            # Fase 2: Marcha Rápida
            PhaseConfig(
                name="marcha_rapida",
                target_speed=1.5,
                enabled_components=["velocity", "stability", "propulsion", "clearance", "coordination"],
                component_weights={"velocity": 0.25, "stability": 0.2, "propulsion": 0.25, "clearance": 0.15, "coordination": 0.15},
                min_episodes=20,
                transition_conditions={
                    "min_success_rate": 0.6,
                    "min_avg_distance": 2.0,
                    "max_avg_roll": 0.5,
                    "min_avg_speed": 0.6,
                    "min_gait_coordination": 0.4
                },
                focus_skills=["energy_efficiency", "gait_coordination"]
            )
        ]
    
    def update_phase(self, episode_results):
        """Atualiza fase baseada nos resultados do episódio"""
        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS
        
        self.episodes_in_phase += 1
        self.performance_history.append(episode_results)
        
        # Manter histórico limitado
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
        
        # Atualizar contadores de sucesso/fracasso
        self._update_success_counters(episode_results)
        
        # Verificar se pode avançar
        if self._should_advance_phase():
            return self._advance_to_next_phase()
        
        # Verificar se precisa regredir
        if self._should_regress_phase():
            return self._regress_to_previous_phase()
        
        return PhaseTransitionResult.SUCCESS
    
    def _should_advance_phase(self):
        """Verifica se condições para avançar de fase foram atendidas"""
        if self.episodes_in_phase < self.current_phase_config.min_episodes:
            return False
        
        conditions = self.current_phase_config.transition_conditions
        
        # Verificar condições básicas
        success_rate = self._calculate_success_rate()
        if success_rate < conditions["min_success_rate"]:
            return False
        
        avg_distance = self._calculate_avg_distance()
        if avg_distance < conditions["min_avg_distance"]:
            return False
        
        avg_roll = self._calculate_avg_roll()
        if avg_roll > conditions["max_avg_roll"]:
            return False
        
        # Condições específicas por fase
        if "min_avg_speed" in conditions:
            avg_speed = self._calculate_avg_speed()
            if avg_speed < conditions["min_avg_speed"]:
                return False
        
        if "min_alternating_score" in conditions and self.current_phase >= 1:
            alternation = self._calculate_alternating_score()
            if alternation < conditions["min_alternating_score"]:
                return False
        
        return True
    
    def _should_regress_phase(self):
        """Verifica se precisa regredir para fase anterior"""
        if self.current_phase == 0:
            return False
        
        # Regressão por falhas consecutivas
        if self.consecutive_failures > 25:
            return True
        
        # Regressão por estagnação prolongada
        if self.stagnation_counter > 40:
            return True
        
        # Regressão por performance muito baixa
        success_rate = self._calculate_success_rate()
        if success_rate < 0.1 and self.episodes_in_phase > 30:
            return True
        
        return False
    
    def _advance_to_next_phase(self):
        """Avança para próxima fase"""
        old_phase = self.current_phase
        self.current_phase += 1
        self.episodes_in_phase = 0
        self.consecutive_failures = 0
        self.stagnation_counter = 0
        
        self.logger.info(f"🎯 TRANSIÇÃO DE FASE: {self.phases[old_phase].name} → {self.current_phase_config.name}")
        return PhaseTransitionResult.SUCCESS
    
    def _regress_to_previous_phase(self):
        """Regride para fase anterior"""
        old_phase = self.current_phase
        self.current_phase = max(0, self.current_phase - 1)
        self.episodes_in_phase = 0
        self.consecutive_failures = 0
        self.stagnation_counter = 0
        
        self.logger.info(f"🔄 REGRESSÃO DE FASE: {self.phases[old_phase].name} → {self.current_phase_config.name}")
        return PhaseTransitionResult.REGRESSION
    
    def _update_success_counters(self, episode_results):
        """Atualiza contadores de sucesso e estagnação"""
        success = episode_results.get("success", False)
        distance = episode_results.get("distance", 0)
        
        if success and distance > 0.1:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        # Detectar estagnação
        if len(self.performance_history) >= 5:
            recent_distances = [r.get("distance", 0) for r in self.performance_history[-5:]]
            if np.std(recent_distances) < 0.05 and np.mean(recent_distances) < 0.3:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
    
    def get_current_phase_info(self):
        """Retorna informações da fase atual"""
        config = self.current_phase_config
        return {
            'phase': self.current_phase,
            'name': config.name,
            'target_speed': config.target_speed,
            'enabled_components': config.enabled_components,
            'component_weights': config.component_weights,
            'focus_skills': config.focus_skills,
            'episodes_in_phase': self.episodes_in_phase
        }
    
    def get_status(self):
        """Retorna status do gerenciador de fases"""
        return {
            "current_phase": self.current_phase,
            "phase_name": self.current_phase_config.name,
            "episodes_in_phase": self.episodes_in_phase,
            "total_phases": len(self.phases),
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "performance_history_size": len(self.performance_history)
        }
    
    def get_phase_name(self):
        """Retorna nome da fase atual"""
        return self.current_phase_config.name
    
    @property
    def current_phase_config(self):
        """Configuração da fase atual"""
        return self.phases[self.current_phase]
    
    # Métricas de performance
    def _calculate_success_rate(self):
        if not self.performance_history:
            return 0.0
        successes = sum(1 for r in self.performance_history if r.get("success", False))
        return successes / len(self.performance_history)
    
    def _calculate_avg_distance(self):
        if not self.performance_history:
            return 0.0
        return np.mean([r.get("distance", 0) for r in self.performance_history])
    
    def _calculate_avg_roll(self):
        if not self.performance_history:
            return 0.0
        return np.mean([abs(r.get("roll", 0)) for r in self.performance_history])
    
    def _calculate_avg_speed(self):
        if not self.performance_history:
            return 0.0
        return np.mean([r.get("speed", 0) for r in self.performance_history])
    
    def _calculate_alternating_score(self):
        if not self.performance_history:
            return 0.0
        alternations = sum(1 for r in self.performance_history if r.get("alternating", False))
        return alternations / len(self.performance_history)