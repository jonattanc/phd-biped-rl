# dpg_gait_phases.py
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import utils

@dataclass
class GaitPhaseConfig:
    """Configuração de uma fase da marcha para DPG"""
    name: str
    target_speed: float  # m/s
    enabled_components: List[str]
    component_weights: Dict[str, float]
    phase_duration: int  # episódios mínimos na fase
    transition_conditions: Dict[str, float]  # condições para transição

class GaitPhaseDPG:
    """
    Sistema DPG baseado nas fases da marcha descritas no documento
    """
    
    def __init__(self, logger, reward_system):
        self.logger = logger
        self.reward_system = reward_system
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.phases = []
        self.performance_history = []
        self.max_history_size = 50
        
        # Definir fases baseadas no documento "Fases da Marcha"
        self._initialize_gait_phases()
        
    def _initialize_gait_phases(self):
        """Inicializa as fases da marcha baseadas no documento"""
        
        # Fase 1: Marcha lenta (alvo 1.6 m/s)
        phase1 = GaitPhaseConfig(
            name="marcha_lenta",
            target_speed=1.6,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "foot_clearance", "gait_rhythm"
            ],
            component_weights={
                "progress": 0.4,
                "stability_roll": 0.15,
                "stability_pitch": 0.15,
                "alternating_foot_contact": 0.1,
                "gait_pattern_cross": 0.1,
                "foot_clearance": 0.05,
                "gait_rhythm": 0.05
            },
            phase_duration=50,
            transition_conditions={
                "min_success_rate": 0.6,
                "min_avg_distance": 3.0,
                "max_avg_roll": 0.3
            }
        )
        
        # Fase 2: Marcha rápida (alvo 2.2 m/s)
        phase2 = GaitPhaseConfig(
            name="marcha_rapida",
            target_speed=2.2,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "foot_clearance", "gait_rhythm", "pitch_forward_bonus"
            ],
            component_weights={
                "progress": 0.5,
                "stability_roll": 0.12,
                "stability_pitch": 0.12,
                "alternating_foot_contact": 0.08,
                "gait_pattern_cross": 0.08,
                "foot_clearance": 0.05,
                "gait_rhythm": 0.03,
                "pitch_forward_bonus": 0.02
            },
            phase_duration=40,
            transition_conditions={
                "min_success_rate": 0.7,
                "min_avg_distance": 5.0,
                "min_avg_speed": 1.8
            }
        )
        
        # Fase 3: Corrida (alvo 2.6-2.8 m/s)
        phase3 = GaitPhaseConfig(
            name="corrida",
            target_speed=2.6,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch", "foot_clearance",
                "pitch_forward_bonus", "success_bonus", "distance_bonus"
            ],
            component_weights={
                "progress": 0.6,
                "stability_roll": 0.1,
                "stability_pitch": 0.1,
                "foot_clearance": 0.08,
                "pitch_forward_bonus": 0.06,
                "success_bonus": 0.04,
                "distance_bonus": 0.02
            },
            phase_duration=30,
            transition_conditions={
                "min_success_rate": 0.8,
                "min_avg_distance": 7.0,
                "min_avg_speed": 2.2
            }
        )
        
        self.phases = [phase1, phase2, phase3]
        
    def update_phase(self, episode_results: Dict) -> bool:
        """
        Atualiza a fase atual baseada no desempenho
        Retorna True se houve transição de fase
        """
        if self.current_phase >= len(self.phases) - 1:
            return False  # Já está na última fase
            
        self.episodes_in_phase += 1
        
        # Adicionar resultado ao histórico
        self.performance_history.append(episode_results)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
            
        # Verificar condições para transição
        current_phase = self.phases[self.current_phase]
        
        # Mínimo de episódios na fase atual
        if self.episodes_in_phase < current_phase.phase_duration:
            return False
            
        # Calcular métricas do histórico
        if len(self.performance_history) < 10:  # Mínimo de dados
            return False
            
        success_rate = self._calculate_success_rate()
        avg_distance = self._calculate_average_distance()
        avg_speed = self._calculate_average_speed()
        avg_roll = self._calculate_average_roll()
        
        # Verificar condições de transição
        conditions = current_phase.transition_conditions
        can_transition = True
        
        if "min_success_rate" in conditions and success_rate < conditions["min_success_rate"]:
            can_transition = False
        if "min_avg_distance" in conditions and avg_distance < conditions["min_avg_distance"]:
            can_transition = False
        if "min_avg_speed" in conditions and avg_speed < conditions["min_avg_speed"]:
            can_transition = False
        if "max_avg_roll" in conditions and avg_roll > conditions["max_avg_roll"]:
            can_transition = False
            
        if can_transition:
            old_phase = self.current_phase
            self.current_phase += 1
            self.episodes_in_phase = 0
            self.performance_history = []  # Resetar histórico para nova fase
            
            self.logger.info(f"Transição de fase: {self.phases[old_phase].name} -> {self.phases[self.current_phase].name}")
            self._apply_phase_config()
            return True
            
        return False
        
    def _calculate_success_rate(self) -> float:
        """Calcula taxa de sucesso dos últimos episódios"""
        if not self.performance_history:
            return 0.0
        successes = sum(1 for result in self.performance_history if result.get("success", False))
        return successes / len(self.performance_history)
        
    def _calculate_average_distance(self) -> float:
        """Calcula distância média dos últimos episódios"""
        if not self.performance_history:
            return 0.0
        return np.mean([result.get("distance", 0) for result in self.performance_history])
        
    def _calculate_average_speed(self) -> float:
        """Calcula velocidade média dos últimos episódios"""
        if not self.performance_history:
            return 0.0
        distances = [result.get("distance", 0) for result in self.performance_history]
        times = [result.get("time", 1) for result in self.performance_history]  # Evitar divisão por zero
        speeds = [d/t if t > 0 else 0 for d, t in zip(distances, times)]
        return np.mean(speeds)
        
    def _calculate_average_roll(self) -> float:
        """Calcula roll médio dos últimos episódios"""
        if not self.performance_history:
            return 0.0
        return np.mean([abs(result.get("roll", 0)) for result in self.performance_history])
        
    def _apply_phase_config(self):
        """Aplica a configuração da fase atual ao sistema de recompensa"""
        current_phase = self.phases[self.current_phase]
        
        # Atualizar componentes habilitados e pesos
        for component_name, component in self.reward_system.components.items():
            if component_name in current_phase.enabled_components:
                component.enabled = True
                component.weight = current_phase.component_weights.get(component_name, component.weight)
            else:
                component.enabled = False
                
        self.logger.info(f"Fase {current_phase.name} aplicada - Velocidade alvo: {current_phase.target_speed} m/s")
        
    def get_current_speed_target(self) -> float:
        """Retorna a velocidade alvo da fase atual"""
        return self.phases[self.current_phase].target_speed
        
    def get_status(self) -> Dict:
        """Retorna status atual do DPG de fases da marcha"""
        current_phase = self.phases[self.current_phase]
        return {
            "current_phase": current_phase.name,
            "phase_index": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "target_speed": current_phase.target_speed,
            "total_phases": len(self.phases)
        }
        
    def reset(self):
        """Reinicia o DPG para a primeira fase"""
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.performance_history = []
        self._apply_phase_config()