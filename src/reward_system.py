# reward_system.py
import math
import os
import time
import numpy as np
import pybullet as p
import json
from dataclasses import dataclass
import utils


@dataclass
class RewardComponent:
    name: str
    weight: float
    value: float = 0.0
    enabled: bool = True
    min_value: float = -float("inf")
    max_value: float = float("inf")


class RewardSystem:
    def __init__(self, logger):
        logger.info("Inicializando RewardSystem...")

        self.logger = logger
        self.components = {}

        self.safe_zone = 0.2  # m
        self.warning_zone = 0.4  # m

        self.load_configuration_file("default.json", is_default_file=True)
        self.default_components = self.get_configuration_as_dict()

    def set_is_fast_td3(self, is_fast_td3):
        self.is_fast_td3 = is_fast_td3

    def is_component_enabled(self, name):
        if name not in self.components:
            return False
        return self.components[name].enabled

    def calculate_reward(self, sim, action, evaluation=False):
        """Calcula recompensa padrão"""

        # Resetar valores dos componentes
        for component in self.components.values():
            component.value = 0.0

        total_reward = 0.0
        weight_adjustments = {}  # Multiplicadores específicos por componente

        # VERIFICAÇÃO FastTD3 - Obter multiplicadores específicos

        if not evaluation and self.is_fast_td3:
            weight_adjustments = sim.agent.model.get_phase_weight_adjustments()
        else:
            weight_adjustments = {
                "gait_state_change": 1.0,
                "progress": 1.0,
                "xcom_stability": 1.0,
                "simple_stability": 1.0,
                "pitch_forward_bonus": 1.0,
                "knee_flexion": 1.0,
                "hip_extension": 1.0,
                "efficiency_bonus": 1.0,
                "foot_clearance": 1.0,
                "distance_bonus": 1.0,
                "gait_pattern_cross": 1.0,
                "gait_rhythm": 1.0,
                "alternating_foot_contact": 1.0,
                "success_bonus": 1.0,
                "adaptability_bonus": 1.0,
                "propulsion_efficiency": 1.0,
                "fall_penalty": 1.0,
                "yaw_penalty": 1.0,
                "y_axis_deviation_square_penalty": 1.0,
                "stability_pitch": 1.0,
                "stability_roll": 1.0,
                "stability_yaw": 1.0,
                "foot_back_penalty": 1.0,
                "foot_inclination_penalty": 1.0,
                "effort_square_penalty": 1.0,
                "jerk_penalty": 1.0,
            }

        # COMPONENTES PARA MARCHA
        distance_y_from_center = abs(sim.robot_y_position)

        # Bonus ativos

        # DPG - 1. Transições de estado
        if self.is_component_enabled("gait_state_change"):
            self.components["gait_state_change"].value = sim.has_gait_state_changed
            weight_multiplier = weight_adjustments.get("gait_state_change", 1.0)
            adjusted_weight = self.components["gait_state_change"].weight * weight_multiplier

            total_reward += self.components["gait_state_change"].value * adjusted_weight

        # DPG - 2. PROGRESSO E VELOCIDADE
        if self.is_component_enabled("progress"):
            self.components["progress"].value = sim.target_x_velocity - abs(sim.target_x_velocity - sim.robot_x_velocity)
            weight_multiplier = weight_adjustments.get("progress", 1.0)
            adjusted_weight = self.components["progress"].weight * weight_multiplier

            total_reward += self.components["progress"].value * adjusted_weight

        # DPG - 3. XCOM
        if self.is_component_enabled("xcom_stability"):
            if hasattr(sim, "mos_min") and sim.mos_min > -0.5:  
                stability_reward = math.tanh(sim.mos_min * 3.0)
                if sim.mos_min > 0.05:
                    stability_reward += 0.3
                elif sim.mos_min > 0.02: 
                    stability_reward += 0.1
            else:
                stability_reward = -0.5  
            self.components["xcom_stability"].value = stability_reward
            weight_multiplier = weight_adjustments.get("xcom_stability", 1.0)
            adjusted_weight = self.components["xcom_stability"].weight * weight_multiplier

            total_reward += self.components["xcom_stability"].value * adjusted_weight

        # DPG - 4. Evitar escorregar
        if self.is_component_enabled("simple_stability"):
            stability_bonus = 0.0
            if sim.robot_left_foot_contact != sim.robot_right_foot_contact:
                stability_bonus += 0.1
            if sim.robot_left_foot_contact and abs(sim.robot_left_foot_x_velocity) < 0.05:
                stability_bonus += 0.05
            if sim.robot_right_foot_contact and abs(sim.robot_right_foot_x_velocity) < 0.05:
                stability_bonus += 0.05
            if not sim.robot_left_foot_contact and sim.robot_left_foot_height > 0.02:
                stability_bonus += 0.02
            if not sim.robot_right_foot_contact and sim.robot_right_foot_height > 0.02:
                stability_bonus += 0.02
            self.components["simple_stability"].value = stability_bonus
            weight_multiplier = weight_adjustments.get("simple_stability", 1.0)
            adjusted_weight = self.components["simple_stability"].weight * weight_multiplier

            total_reward += self.components["simple_stability"].value * adjusted_weight

        # DPG - 5. Pitch
        if self.is_component_enabled("pitch_forward_bonus"):
            current_pitch = sim.robot_pitch
            if current_pitch > 0:
                if current_pitch <= 0.349:
                    normalized = current_pitch / 0.349
                    pitch_bonus = 4.0 * normalized * (1.0 - normalized)
                else:
                    pitch_bonus = 0
            else:
                pitch_bonus = 0
            self.components["pitch_forward_bonus"].value = pitch_bonus
            weight_multiplier = weight_adjustments.get("pitch_forward_bonus", 1.0)
            adjusted_weight = self.components["pitch_forward_bonus"].weight * weight_multiplier

            total_reward += self.components["pitch_forward_bonus"].value * adjusted_weight

        # DPG - 6. Flexão dos joelhos
        if self.is_component_enabled("knee_flexion"):
            self.components["knee_flexion"].value = abs(sim.robot_right_knee_angle) + abs(sim.robot_left_knee_angle)
            weight_multiplier = weight_adjustments.get("knee_flexion", 1.0)
            adjusted_weight = self.components["knee_flexion"].weight * weight_multiplier

            total_reward += self.components["knee_flexion"].value * adjusted_weight

        # DPG - 7. Quadril
        if self.is_component_enabled("hip_extension"):
            self.components["hip_extension"].value = abs(sim.robot_right_hip_frontal_angle) + abs(sim.robot_left_hip_frontal_angle)
            weight_multiplier = weight_adjustments.get("hip_extension", 1.0)
            adjusted_weight = self.components["hip_extension"].weight * weight_multiplier

            total_reward += self.components["hip_extension"].value * adjusted_weight

        # DPG - 8. EFICIÊNCIA
        if self.is_component_enabled("efficiency_bonus"):
            steps = max(sim.episode_steps, 1)
            reward_per_step = sim.episode_reward / steps
            distance_per_step = sim.episode_distance / steps
            efficiency_score = reward_per_step * 0.6 + distance_per_step * 50 * 0.4
            self.components["efficiency_bonus"].value = max(0, efficiency_score * 2.0)
            weight_multiplier = weight_adjustments.get("efficiency_bonus", 1.0)
            adjusted_weight = self.components["efficiency_bonus"].weight * weight_multiplier

            total_reward += self.components["efficiency_bonus"].value * adjusted_weight

        # DPG - 9. Clearance
        if self.is_component_enabled("foot_clearance"):
            self.components["foot_clearance"].value = self._calculate_foot_clearance_optimized(sim)
            weight_multiplier = weight_adjustments.get("foot_clearance", 1.0)
            adjusted_weight = self.components["foot_clearance"].weight * weight_multiplier

            total_reward += self.components["foot_clearance"].value * adjusted_weight

        # DPG - 10. Bonus de distância
        if self.is_component_enabled("distance_bonus"):
            self.components["distance_bonus"].value = sim.episode_distance
            weight_multiplier = weight_adjustments.get("distance_bonus", 1.0)
            adjusted_weight = self.components["distance_bonus"].weight * weight_multiplier

            total_reward += self.components["distance_bonus"].value * adjusted_weight

        # DPG - 11. PADRÃO DE MARCHA CRUZADA (Coordenação braço-perna)
        if self.is_component_enabled("gait_pattern_cross"):
            self.components["gait_pattern_cross"].value = self._calculate_cross_gait_pattern(sim)
            weight_multiplier = weight_adjustments.get("gait_pattern_cross", 1.0)
            adjusted_weight = self.components["gait_pattern_cross"].weight * weight_multiplier

            total_reward += self.components["gait_pattern_cross"].value * adjusted_weight

        # DPG - 12. PADRÃO RÍTMICO (Regularidade da marcha)
        if self.is_component_enabled("gait_rhythm"):
            self.components["gait_rhythm"].value = self._calculate_gait_rhythm(sim)
            weight_multiplier = weight_adjustments.get("gait_rhythm", 1.0)
            adjusted_weight = self.components["gait_rhythm"].weight * weight_multiplier

            total_reward += self.components["gait_rhythm"].value * adjusted_weight

        # DPG - 13. ALTERNÂNCIA DE PASSOS (Critério fundamental da marcha)
        if self.is_component_enabled("alternating_foot_contact"):
            self.components["alternating_foot_contact"].value = sim.robot_left_foot_contact != sim.robot_right_foot_contact
            weight_multiplier = weight_adjustments.get("alternating_foot_contact", 1.0)
            adjusted_weight = self.components["alternating_foot_contact"].weight * weight_multiplier

            total_reward += self.components["alternating_foot_contact"].value * adjusted_weight

        # DPG - 14. Bonus de sucesso
        if self.is_component_enabled("success_bonus"):
            if sim.episode_termination == "success":
                self.components["success_bonus"].value = 1
            else:
                self.components["success_bonus"].value = 0
            weight_multiplier = weight_adjustments.get("success_bonus", 1.0)
            adjusted_weight = self.components["success_bonus"].weight * weight_multiplier

            total_reward += self.components["success_bonus"].value * adjusted_weight

        # Penalidades ativas

        # DPG - 1. PENALIDADES POR QUEDA
        if self.is_component_enabled("fall_penalty"):
            if sim.episode_termination == "fell":
                self.components["fall_penalty"].value = 1
            else:
                self.components["fall_penalty"].value = 0
            weight_multiplier = weight_adjustments.get("fall_penalty", 1.0)
            adjusted_weight = self.components["fall_penalty"].weight * weight_multiplier

            total_reward += self.components["fall_penalty"].value * adjusted_weight

        # DPG - 2. CONTROLE DE TRAJETÓRIA (Manter direção)
        if self.is_component_enabled("yaw_penalty"):
            if sim.episode_termination == "yaw_deviated":
                self.components["yaw_penalty"].value = 1
            else:
                self.components["yaw_penalty"].value = 0
            weight_multiplier = weight_adjustments.get("yaw_penalty", 1.0)
            adjusted_weight = self.components["yaw_penalty"].weight * weight_multiplier

            total_reward += self.components["yaw_penalty"].value * adjusted_weight

        # DPG - 3. Ir pra frente
        if self.is_component_enabled("foot_back_penalty"):
            backwards_velocity = 0
            if sim.robot_left_foot_x_velocity < 0:
                backwards_velocity += abs(sim.robot_left_foot_x_velocity)
            if sim.robot_right_foot_x_velocity < 0:
                backwards_velocity += abs(sim.robot_right_foot_x_velocity)

            self.components["foot_back_penalty"].value = backwards_velocity
            weight_multiplier = weight_adjustments.get("foot_back_penalty", 1.0)
            adjusted_weight = self.components["foot_back_penalty"].weight * weight_multiplier

            total_reward += self.components["foot_back_penalty"].value * adjusted_weight

        # DPG - 4. Inclinar os pés
        if self.is_component_enabled("foot_inclination_penalty"):
            right_foot_inclination = abs(sim.robot_right_foot_roll) + abs(sim.robot_right_foot_pitch)
            left_foot_inclination = abs(sim.robot_left_foot_roll) + abs(sim.robot_left_foot_pitch)
            self.components["foot_inclination_penalty"].value = right_foot_inclination + left_foot_inclination
            weight_multiplier = weight_adjustments.get("foot_inclination_penalty", 1.0)
            adjusted_weight = self.components["foot_inclination_penalty"].weight * weight_multiplier

            total_reward += self.components["foot_inclination_penalty"].value * adjusted_weight

        # DPG - 5. ESTABILIDADE DA MARCHA (Controle postural)
        if self.is_component_enabled("stability_pitch"):
            self.components["stability_pitch"].value = (sim.robot_pitch - sim.target_pitch_rad) ** 2
            weight_multiplier = weight_adjustments.get("stability_pitch", 1.0)
            adjusted_weight = self.components["stability_pitch"].weight * weight_multiplier

            total_reward += self.components["stability_pitch"].value * adjusted_weight

        if self.is_component_enabled("stability_original_pitch"):
            self.components["stability_original_pitch"].value = (sim.robot_pitch - sim.target_pitch_rad) ** 2
            weight_multiplier = weight_adjustments.get("stability_original_pitch", 1.0)
            adjusted_weight = self.components["stability_original_pitch"].weight * weight_multiplier

            total_reward += self.components["stability_original_pitch"].value * adjusted_weight

        # DPG - 6. ESTABILIDADE DA MARCHA (Controle postural)
        if self.is_component_enabled("stability_yaw"):
            self.components["stability_yaw"].value = sim.robot_yaw**2
            weight_multiplier = weight_adjustments.get("stability_yaw", 1.0)
            adjusted_weight = self.components["stability_yaw"].weight * weight_multiplier
            total_reward += self.components["stability_yaw"].value * adjusted_weight

        # DPG - 7. ESTABILIDADE DA MARCHA (Controle postural)
        if self.is_component_enabled("stability_roll"):
            roll_error = sim.robot_roll**2
            self.components["stability_roll"].value = roll_error
            weight_multiplier = weight_adjustments.get("stability_roll", 1.0)
            adjusted_weight = self.components["stability_roll"].weight * weight_multiplier

            total_reward += self.components["stability_roll"].value * adjusted_weight

        # DPG - 8. EFICIÊNCIA ENERGÉTICA (Movimentos suaves)
        if self.is_component_enabled("effort_square_penalty"):
            effort = sum(v**2 for v in sim.joint_velocities)
            self.components["effort_square_penalty"].value = effort
            weight_multiplier = weight_adjustments.get("effort_square_penalty", 1.0)
            adjusted_weight = self.components["effort_square_penalty"].weight * weight_multiplier
            total_reward += self.components["effort_square_penalty"].value * adjusted_weight

        # DPG - 9. Se manter na pista
        if self.is_component_enabled("y_axis_deviation_square_penalty"):
            self.components["y_axis_deviation_square_penalty"].value = distance_y_from_center**2
            weight_multiplier = weight_adjustments.get("y_axis_deviation_square_penalty", 1.0)
            adjusted_weight = self.components["y_axis_deviation_square_penalty"].weight * weight_multiplier

            total_reward += self.components["y_axis_deviation_square_penalty"].value * adjusted_weight

        # DPG - 10. Jerk
        if self.is_component_enabled("jerk_penalty"):
            jerk = sum(abs(v1 - v2) for v1, v2 in zip(sim.joint_velocities, sim.last_joint_velocities))
            self.components["jerk_penalty"].value = jerk
            weight_multiplier = weight_adjustments.get("jerk_penalty", 1.0)
            adjusted_weight = self.components["jerk_penalty"].weight * weight_multiplier
            total_reward += self.components["jerk_penalty"].value * adjusted_weight
        
        # DPG - Novas Ideias
        
        # DPG - Eficiencia na propulsão
        if self.is_component_enabled("propulsion_efficiency"):
            speed = abs(sim.robot_x_velocity)
            energy = sum(abs(v) for v in sim.joint_velocities) 
            if energy > 0:
                propulsion_efficiency = speed / (energy + 0.1)  
            else:
                propulsion_efficiency = speed
            
            self.components["propulsion_efficiency"].value = propulsion_efficiency
            weight_multiplier = weight_adjustments.get("propulsion_efficiency", 1.0)
            adjusted_weight = self.components["propulsion_efficiency"].weight * weight_multiplier
            total_reward += self.components["propulsion_efficiency"].value * adjusted_weight

        # DPG - Adaptação suave
        if self.is_component_enabled("adaptability_bonus"):
            adaptability_score = 0.0
            if hasattr(sim, 'last_joint_positions'):
                joint_variation = sum(abs(np.array(sim.joint_positions) - np.array(sim.last_joint_positions)))
                smoothness = max(0, 0.5 - joint_variation * 2.0)
                adaptability_score += smoothness
            right_coordination = abs(sim.robot_right_knee_angle + sim.robot_right_hip_frontal_angle * 0.7)
            left_coordination = abs(sim.robot_left_knee_angle + sim.robot_left_hip_frontal_angle * 0.7)
            coordination_bonus = max(0, 1.0 - (right_coordination + left_coordination) * 0.3)
            adaptability_score += coordination_bonus * 0.5
            self.components["adaptability_bonus"].value = adaptability_score
            weight_multiplier = weight_adjustments.get("adaptability_bonus", 1.0)
            adjusted_weight = self.components["adaptability_bonus"].weight * weight_multiplier

            total_reward += self.components["adaptability_bonus"].value * adaptability_score



        # RECOMPENSAS DINÂMICAS PARA FASE 3
        if not evaluation and self.is_fast_td3:
            phase_info = sim.agent.model.get_phase_info()
            if phase_info["phase"] == 3:
                # 1. Bônus por velocidade consistente (> 0.8 m/s)
                if sim.robot_x_velocity > 0.8:
                    velocity_bonus = (sim.robot_x_velocity - 0.8) * 15
                    total_reward += velocity_bonus

                # 2. Bônus por completar último terço do percurso
                progress_ratio = sim.episode_distance / sim.success_distance
                if progress_ratio > 0.66:
                    final_stretch_bonus = 10.0 * progress_ratio
                    total_reward += final_stretch_bonus

                # 3. Bônus por "momentum" - velocidade positiva consistente
                momentum_bonus = 0.0
                if sim.robot_x_velocity > 0.5:
                    momentum_bonus = sim.robot_x_velocity * 3.0
                elif abs(sim.robot_x_velocity) < 0.1:
                    momentum_bonus = -8.0

                total_reward += momentum_bonus

                # 4. Bônus EXTRA por sucesso na Fase 3
                if sim.episode_success:
                    success_bonus_extra = 200.0
                    total_reward += success_bonus_extra

        # Componentes não ativos
        if self.is_component_enabled("foot_clearance_original"):
            foot_height = 0
            if not sim.robot_left_foot_contact:
                foot_height += sim.robot_left_foot_height
            if not sim.robot_right_foot_contact:
                foot_height += sim.robot_right_foot_height

            self.components["foot_clearance_original"].value = foot_height
            total_reward += self.components["foot_clearance_original"].value * self.components["foot_clearance_original"].weight

        if self.is_component_enabled("height_deviation_square_penalty"):
            height_error = (sim.robot_z_position - 0.8) ** 2
            self.components["height_deviation_square_penalty"].value = height_error
            total_reward += height_error * self.components["height_deviation_square_penalty"].weight

        if self.is_component_enabled("height_deviation_penalty"):
            self.components["height_deviation_penalty"].value = abs(sim.robot_y_position - sim.episode_robot_y_initial_position)
            total_reward += self.components["height_deviation_penalty"].value * self.components["height_deviation_penalty"].weight

        if self.is_component_enabled("effort_penalty"):
            effort = sum(abs(v) for v in sim.joint_velocities)
            self.components["effort_penalty"].value = effort
            total_reward += effort * self.components["effort_penalty"].weight

        if self.is_component_enabled("direction_change_penalty"):
            action_products = action * sim.episode_last_action
            direction_changes = np.sum(action_products < 0)
            self.components["direction_change_penalty"].value = direction_changes
            total_reward += direction_changes * self.components["direction_change_penalty"].weight

        if self.is_component_enabled("hip_openning"):
            self.components["hip_openning"].value = abs(sim.robot_right_hip_lateral_angle) + abs(sim.robot_left_hip_lateral_angle)
            total_reward += self.components["hip_openning"].value * self.components["hip_openning"].weight

        if self.is_component_enabled("hip_openning_square"):
            self.components["hip_openning_square"].value = sim.robot_right_hip_lateral_angle**2 + sim.robot_left_hip_lateral_angle**2
            total_reward += self.components["hip_openning_square"].value * self.components["hip_openning_square"].weight

        if self.is_component_enabled("y_axis_deviation_penalty"):
            penalty = distance_y_from_center
            self.components["y_axis_deviation_penalty"].value = penalty
            total_reward += penalty * self.components["y_axis_deviation_penalty"].weight

        if self.is_component_enabled("y_axis_deviation_cube_penalty"):
            penalty = distance_y_from_center**3
            self.components["y_axis_deviation_cube_penalty"].value = penalty
            total_reward += penalty * self.components["y_axis_deviation_cube_penalty"].weight

        if self.is_component_enabled("center_bonus"):
            if distance_y_from_center <= self.safe_zone:
                safe_factor = 1.0 - (distance_y_from_center / self.safe_zone)
                self.components["center_bonus"].value = safe_factor
                total_reward += safe_factor * self.components["center_bonus"].weight

        if self.is_component_enabled("warning_penalty"):
            if distance_y_from_center > self.safe_zone and distance_y_from_center <= self.warning_zone:
                warning_factor = (distance_y_from_center - self.safe_zone) / (self.warning_zone - self.safe_zone)
                self.components["warning_penalty"].value = warning_factor
                total_reward += warning_factor * self.components["warning_penalty"].weight

        return total_reward

    def _calculate_cross_gait_pattern(self, sim):
        """Recompensa por alternância dinâmica braço-perna"""

        try:
            right_arm_angle = getattr(sim, "robot_right_shoulder_front_angle", 0)
            left_arm_angle = getattr(sim, "robot_left_shoulder_front_angle", 0)
        except:
            return 0.0

        if not hasattr(self, '_arm_history'):
            self._arm_history = []
        arm_moving = abs(right_arm_angle) > 0.05 or abs(left_arm_angle) > 0.05
        if arm_moving:
            current_state = f"R{right_arm_angle:.2f}_L{left_arm_angle:.2f}"
            self._arm_history.append(current_state)
            if len(self._arm_history) > 10:
                self._arm_history.pop(0)
        score = 0.0
        right_arm_back = right_arm_angle > 0.1
        left_arm_back = left_arm_angle > 0.1
        if right_arm_back != left_arm_back:
            score += 0.5  
            if right_arm_back and not sim.robot_left_foot_contact:
                score += 1.0  
            elif left_arm_back and not sim.robot_right_foot_contact:
                score += 1.0  
        if right_arm_back and left_arm_back:
            score -= 0.3
        if len(set(self._arm_history)) > 5:
            score += 0.4

        return max(0.0, score)

    def _calculate_foot_clearance_optimized(self, sim):
        """Calcula recompensa por altura adequada dos pés durante o balanço"""
        optimal_clearance = 0.05  # 5cm ideal durante balanço
        clearance_score = 0.0

        # Pé direito no balanço
        if not sim.robot_right_foot_contact:
            current_clearance = sim.robot_right_foot_height
            # Recompensa por estar próximo da altura ideal
            clearance_error = abs(current_clearance - optimal_clearance)
            clearance_score += max(0, 0.1 - clearance_error)

        # Pé esquerdo no balanço
        if not sim.robot_left_foot_contact:
            current_clearance = sim.robot_left_foot_height
            clearance_error = abs(current_clearance - optimal_clearance)
            clearance_score += max(0, 0.1 - clearance_error)

        return clearance_score

    def _calculate_gait_rhythm(self, sim):
        """Calcula regularidade rítmica da marcha"""
        # Baseado na periodicidade das forças articulares
        if not hasattr(self, "last_step_time"):
            self.last_step_time = time.time()
            self.step_intervals = []
            return 0.5  # Valor neutro inicial

        current_time = time.time()
        step_interval = current_time - self.last_step_time

        # Detectar transição de passo (mudança no contato dos pés)
        foot_state_changed = sim.robot_left_foot_contact != getattr(self, "last_left_contact", False) or sim.robot_right_foot_contact != getattr(self, "last_right_contact", False)

        if foot_state_changed and step_interval > 0.1:  # Evitar detecções muito rápidas
            self.step_intervals.append(step_interval)
            self.last_step_time = current_time

            # Manter apenas últimos 10 intervalos
            if len(self.step_intervals) > 10:
                self.step_intervals.pop(0)

        # Atualizar estados anteriores
        self.last_left_contact = sim.robot_left_foot_contact
        self.last_right_contact = sim.robot_right_foot_contact

        # Calcular regularidade (baixa variância = boa regularidade)
        if len(self.step_intervals) >= 3:
            rhythm_std = np.std(self.step_intervals)
            # Recompensa por baixa variabilidade temporal
            rhythm_score = max(0, 1.0 - rhythm_std * 10)  # Normalizar
            return rhythm_score

        return 0.5  # Valor neutro até ter dados suficientes

    def get_configuration_as_dict(self):
        """Retorna configuração atual em formato dicionário"""
        config = {}

        for name, component in self.components.items():
            config[name] = {"weight": component.weight, "enabled": component.enabled, "min_value": component.min_value, "max_value": component.max_value}

        return config

    def load_configuration(self, config_dict, is_default_file=False):
        """Carrega configuração de um dicionário"""
        self.logger.info(" RewardSystem.load_configuration called")

        try:
            # Carregar componentes
            if "components" not in config_dict:
                raise ValueError("Configuração inválida: 'components' ausente")

            components_config = config_dict["components"]

            if not is_default_file:
                warning_msg = ""
                missing_components = set(self.default_components.keys()) - set(components_config.keys())
                extra_components = set(components_config.keys()) - set(self.default_components.keys())

                if extra_components:
                    warning_msg += "Componentes extras na configuração carregada, ignorando:\n"

                    for comp in extra_components:
                        warning_msg += f" - {comp}\n"
                        del components_config[comp]

                if missing_components:
                    warning_msg += "Componentes faltando na configuração carregada, usando valores default:\n"

                    for comp in missing_components:
                        warning_msg += f" - {comp}\n"
                        components_config[comp] = self.default_components[comp]
                        components_config[comp]["enabled"] = False  # Desabilitar componentes faltantes por segurança

                if warning_msg:
                    self.logger.warning(warning_msg)

            self.components = {}

            for name, content in components_config.items():
                weight = content.get("weight", 0.0)
                enabled = content.get("enabled", False)
                min_value = content.get("min_value", -float("inf"))
                max_value = content.get("max_value", float("inf"))

                self.components[name] = RewardComponent(name=name, weight=weight, enabled=enabled, min_value=min_value, max_value=max_value)

            self.logger.info("Configuração carregada com sucesso")
            return True

        except Exception as e:
            self.logger.exception("Erro ao carregar configuração")
            return False

    def load_configuration_file(self, filepath, is_default_file=False):
        """Carrega configuração de arquivo JSON"""
        self.logger.info(f" RewardSystem.load_configuration_file called with {filepath}")

        if not filepath.endswith(".json"):
            filepath += ".json"

        filepath = os.path.join(utils.REWARD_CONFIGS_PATH, filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config = json.load(f)
            return self.load_configuration(config, is_default_file)
        except Exception as e:
            self.logger.exception(f"Erro ao carregar arquivo de configuração: {filepath}")
            return False

    def update_component(self, name, weight=None, enabled=None):
        """Atualiza um componente"""
        self.logger.info(f" RewardSystem.update_component called for {name}")

        if name not in self.components:
            self.logger.error(f"  Componente {name} não encontrado")
            return False

        if weight is not None:
            self.components[name].weight = weight
            self.logger.info(f"  weight atualizado para {weight}")

        if enabled is not None:
            self.components[name].enabled = enabled
            self.logger.info(f"  enabled atualizado para {enabled}")

        return True