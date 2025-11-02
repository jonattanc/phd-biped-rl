# dpg_reward.py
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class RewardComponent:
    name: str
    weight: float
    calculator: callable


class RewardCalculator:
    """
    Calculador especializado em todas as recompensas DPG
    """
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.components = self._initialize_components()
    
    def _initialize_components(self):
        """Inicializa todos os componentes de recompensa"""
        return {
            "velocity": RewardComponent("velocity", 2.0, self._calculate_velocity_reward),
            "stability": RewardComponent("stability", 3.0, self._calculate_stability_reward),
            "phase_angles": RewardComponent("phase_angles", 1.0, self._calculate_phase_angle_reward),
            "propulsion": RewardComponent("propulsion", 0.5, self._calculate_propulsion_reward),
            "clearance": RewardComponent("clearance", 0.2, self._calculate_clearance_reward),
            "coordination": RewardComponent("coordination", 0.4, self._calculate_coordination_reward),
            "basic_progress": RewardComponent("basic_progress", 1.0, self._calculate_basic_progress),
            "posture": RewardComponent("posture", 0.5, self._calculate_posture_reward),
        }
    
    def calculate(self, sim, action, phase_info):
        """Calcula recompensa total baseada nos componentes habilitados"""
        total_reward = 0.0
        enabled_components = phase_info['enabled_components']
        
        # Bônus massivo para fase inicial
        if phase_info['phase'] == 0:
            phase_0_bonus = self._calculate_phase_0_bonus(sim)
            total_reward += phase_0_bonus
        
        # Calcular cada componente habilitado
        for component_name in enabled_components:
            if component_name in self.components:
                component = self.components[component_name]
                component_reward = component.calculator(sim, phase_info)
                
                # Ajustar peso baseado na fase
                adjusted_weight = self._get_adjusted_weight(component, phase_info)
                total_reward += adjusted_weight * component_reward
        
        # Aplicar penalidades globais
        penalties = self._calculate_global_penalties(sim, action)
        total_reward -= penalties
        
        return total_reward
    
    def _calculate_phase_0_bonus(self, sim):
        """Bônus massivo para fase inicial - foco em progresso básico"""
        bonus = 0.0
        
        # Bônus por qualquer progresso positivo
        distance = getattr(sim, "episode_distance", 0)
        if distance > 0:
            progress_bonus = min(distance * 5.0, 3.0)
            bonus += progress_bonus
        
        # Bônus por velocidade positiva
        velocity = getattr(sim, "robot_x_velocity", 0)
        if velocity > 0.05:
            velocity_bonus = min(velocity * 0.5, 1.0)
            bonus += velocity_bonus
        
        # Bônus por estabilidade básica
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))
        if roll < 0.5 and pitch < 0.4:
            stability_bonus = 2.0
            bonus += stability_bonus
        
        return bonus
    
    def _calculate_velocity_reward(self, sim, phase_info):
        """Recompensa de velocidade adaptativa"""
        vx = getattr(sim, "robot_x_velocity", 0)
        target_speed = phase_info['target_speed']
        
        # Penalizar movimento significativo para trás
        if vx < -0.05:
            return -0.5 * min(abs(vx), 1.0)
        
        # Recompensa normalizada para velocidade positiva
        v_min = target_speed * 0.1
        v_max = target_speed * 1.5
        
        if v_max - v_min > 0:
            normalized_vel = (vx - v_min) / (v_max - v_min)
            clipped_vel = np.clip(normalized_vel, 0.0, 1.0)
            return clipped_vel ** 1.2
        else:
            return 0.0
    
    def _calculate_stability_reward(self, sim, phase_info):
        """Recompensa de estabilidade com tolerância para inclinação frontal"""
        pitch = getattr(sim, "robot_pitch", 0)
        roll = getattr(sim, "robot_roll", 0)
        
        # Penalidade adaptativa para pitch (mais tolerante com inclinação frontal)
        if pitch < -0.1:  # Inclinação frontal (propulsão)
            pitch_penalty = max(0, abs(pitch) - 0.1) * 0.5
        else:  # Inclinação traseira (instabilidade)
            pitch_penalty = abs(pitch) * 1.0
        
        # Penalidade para roll (sempre instabilidade)
        roll_penalty = abs(roll) * 0.8
        
        stability_penalty = pitch_penalty + roll_penalty
        return np.exp(-stability_penalty / 0.5)
    
    def _calculate_phase_angle_reward(self, sim, phase_info):
        """Recompensa por seguir ângulos articulares de referência"""
        # Implementação simplificada - usar targets do config
        try:
            left_hip = abs(getattr(sim, "robot_left_hip_frontal_angle", 0))
            right_hip = abs(getattr(sim, "robot_right_hip_frontal_angle", 0))
            left_knee = abs(getattr(sim, "robot_left_knee_angle", 0))
            right_knee = abs(getattr(sim, "robot_right_knee_angle", 0))
            
            # Recompensa baseada na proximidade dos ângulos ideais
            ideal_hip = 0.2
            ideal_knee = 0.3
            
            hip_reward = np.exp(-0.5 * (left_hip - ideal_hip)**2 / 0.2**2)
            hip_reward += np.exp(-0.5 * (right_hip - ideal_hip)**2 / 0.2**2)
            knee_reward = np.exp(-0.5 * (left_knee - ideal_knee)**2 / 0.2**2)
            knee_reward += np.exp(-0.5 * (right_knee - ideal_knee)**2 / 0.2**2)
            
            return (hip_reward + knee_reward) / 4.0
            
        except Exception:
            return 0.5
    
    def _calculate_propulsion_reward(self, sim, phase_info):
        """Recompensa por geração de propulsão"""
        velocity = getattr(sim, "robot_x_velocity", 0)
        pitch = getattr(sim, "robot_pitch", 0)
        
        # Bônus por inclinação frontal combinada com velocidade
        if pitch < -0.1 and velocity > 0.1:
            propulsion = min(abs(pitch) * velocity * 2.0, 1.0)
            return propulsion
        elif pitch < -0.1:  # Inclinação frontal sem velocidade
            return 0.2
        else:
            return 0.0
    
    def _calculate_clearance_reward(self, sim, phase_info):
        """Recompensa por clearance adequado na oscilação"""
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            
            clearance_reward = 0.0
            
            if not left_contact:
                left_height = getattr(sim, "robot_left_foot_height", 0)
                clearance_reward += 1.0 / (1.0 + np.exp(-100 * (left_height - 0.025)))
            
            if not right_contact:
                right_height = getattr(sim, "robot_right_foot_height", 0)
                clearance_reward += 1.0 / (1.0 + np.exp(-100 * (right_height - 0.025)))
            
            return clearance_reward / 2.0
            
        except Exception:
            return 0.0
    
    def _calculate_coordination_reward(self, sim, phase_info):
        """Recompensa por coordenação braço-perna"""
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            left_arm = getattr(sim, "robot_left_shoulder_front_angle", 0)
            right_arm = getattr(sim, "robot_right_shoulder_front_angle", 0)
            
            coordination = 0.0
            
            # Padrão cruzado: braço esquerdo avança quando pé direito no ar
            if not right_contact and left_arm > 0.1:
                coordination += 0.5
            if not left_contact and right_arm > 0.1:
                coordination += 0.5
            
            return coordination
            
        except Exception:
            return 0.0
    
    def _calculate_basic_progress(self, sim, phase_info):
        """Recompensa básica de progresso para fases iniciais"""
        distance = getattr(sim, "episode_distance", 0)
        return min(distance / 1.0, 1.0)
    
    def _calculate_posture_reward(self, sim, phase_info):
        """Recompensa por postura adequada"""
        try:
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            height = getattr(sim, "robot_z_position", 0.8)
            
            # Recompensa por postura ereta
            posture_score = 0.0
            if roll < 0.3:
                posture_score += 0.4
            if abs(pitch) < 0.2:
                posture_score += 0.4
            if height > 0.7:
                posture_score += 0.2
            
            return posture_score
            
        except Exception:
            return 0.5
    
    def _calculate_global_penalties(self, sim, action):
        """Calcula todas as penalidades globais"""
        penalties = 0.0
        w = self.config.phase_weights
        
        # Penalidade de esforço
        if hasattr(sim, "joint_velocities"):
            joint_velocities = getattr(sim, "joint_velocities", [0])
            torque_cost = w["effort_torque"] * sum(v**2 for v in joint_velocities)
            power_cost = w["effort_power"] * sum(max(0, v)**2 for v in joint_velocities)
            penalties += torque_cost + power_cost
        
        # Penalidade de suavidade da ação
        if hasattr(sim, "episode_last_action"):
            last_action = getattr(sim, "episode_last_action", np.zeros_like(action))
            action_smoothness = w["action_smoothness"] * np.sum((action - last_action)**2)
            penalties += action_smoothness
        
        # Penalidade de deriva lateral
        vy = abs(getattr(sim, "robot_y_velocity", 0))
        penalties += w["lateral_penalty"] * vy
        
        return penalties
    
    def _get_adjusted_weight(self, component, phase_info):
        """Ajusta peso do componente baseado na fase e habilidades focadas"""
        base_weight = component.weight
        focus_skills = phase_info['focus_skills']
        
        # Aumentar peso de componentes relacionados às habilidades focadas
        skill_boost_map = {
            "stability": ["basic_balance", "postural_stability"],
            "velocity": ["dynamic_balance", "energy_efficiency"], 
            "propulsion": ["propulsive_phase"],
            "coordination": ["gait_coordination", "step_consistency"]
        }
        
        for skill, related_components in skill_boost_map.items():
            if component.name in related_components and skill in focus_skills:
                return base_weight * 1.5
        
        return base_weight