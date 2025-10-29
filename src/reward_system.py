# reward_system.py
import os
import numpy as np
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
        
        # DPG
        self.dpg_manager = None
        self.phase_detector = None
        self.gait_phase_dpg = None
        self.dpg_enabled = False

        self.safe_zone = 0.2  # m
        self.warning_zone = 0.4  # m

        self.load_configuration_file("default.json", is_default_file=True)
        self.default_components = self.get_configuration_as_dict()

    def set_phase_detector(self, phase_detector):
        """Configura o detector de fases (para compatibilidade com DPG)"""
        self.phase_detector = phase_detector
        
    def set_dpg_manager(self, dpg_manager):
        """Configura o gerenciador DPG (opcional)"""
        self.dpg_manager = dpg_manager

    def enable_dpg_progression(self, enabled=True):
        """Ativa/desativa progressão por fases para DPG"""
        self.dpg_enabled = enabled
        if enabled:
            if not hasattr(self, 'gait_phase_dpg'):
                from dpg_gait_phases import GaitPhaseDPG
                self.gait_phase_dpg = GaitPhaseDPG(self.logger, self)
                
            self.gait_phase_dpg._apply_phase_config()  # Aplicar configuração inicial
            self.logger.info("DPG com Fases da Marcha ativado")
            
            if hasattr(self, 'phase_detector') and self.phase_detector:
                self.logger.info("Detector de fases da marcha inicializado")
        else:
            self.logger.info("Sistema de recompensa padrão")
            
    def is_component_enabled(self, name):
        if name not in self.components:
            return False
        return self.components[name].enabled

    def calculate_reward(self, sim, action):
        """Método principal - escolhe entre DPG progressivo ou padrão"""
        # Modifique esta verificação:
        if hasattr(self, 'dpg_manager') and self.dpg_manager and self.dpg_manager.config.enabled:
            return self.calculate_dpg_reward(sim, action)
        else:
            return self.calculate_standard_reward(sim, action)

    def calculate_standard_reward(self, sim, action):
        """Calcula recompensa padrão (sem DPG)"""

        # Resetar valores dos componentes
        for component in self.components.values():
            component.value = 0.0

        total_reward = 0.0

        distance_y_from_center = abs(sim.robot_y_position)

        # Componentes de recompensa
        if self.is_component_enabled("gait_state_change"):
            self.components["gait_state_change"].value = sim.has_gait_state_changed
            total_reward += self.components["gait_state_change"].value * self.components["gait_state_change"].weight

        if self.is_component_enabled("progress"):
            progress = sim.target_x_velocity - abs(sim.target_x_velocity - sim.robot_x_velocity)
            self.components["progress"].value = progress
            total_reward += progress * self.components["progress"].weight

        if self.is_component_enabled("distance_bonus"):
            self.components["distance_bonus"].value = sim.episode_distance
            total_reward += sim.episode_distance * self.components["distance_bonus"].weight

        if self.is_component_enabled("gait_pattern_cross"):
            cross_gait_score = self._calculate_cross_gait_pattern(sim)
            self.components["gait_pattern_cross"].value = cross_gait_score
            total_reward += cross_gait_score * self.components["gait_pattern_cross"].weight

        if self.is_component_enabled("stability_roll"):
            self.components["stability_roll"].value = sim.robot_roll**2
            total_reward += sim.robot_roll**2 * self.components["stability_roll"].weight

        if self.is_component_enabled("stability_pitch"):
            self.components["stability_pitch"].value = (sim.robot_pitch - sim.target_pitch_rad) ** 2
            total_reward += self.components["stability_pitch"].value * self.components["stability_pitch"].weight

        if self.is_component_enabled("pitch_forward_bonus"):
            target_forward_pitch = 0.05
            pitch_error = abs(sim.robot_pitch - target_forward_pitch)
            pitch_bonus = max(0, 0.1 - pitch_error)
            self.components["pitch_forward_bonus"].value = pitch_bonus
            total_reward += pitch_bonus * self.components["pitch_forward_bonus"].weight

        if self.is_component_enabled("stability_yaw"):
            self.components["stability_yaw"].value = sim.robot_yaw**2
            total_reward += sim.robot_yaw**2 * self.components["stability_yaw"].weight

        if self.is_component_enabled("yaw_penalty"):
            if sim.episode_termination == "yaw_deviated":
                self.components["yaw_penalty"].value = 1
                total_reward += self.components["yaw_penalty"].weight

        if self.is_component_enabled("fall_penalty"):
            if sim.episode_termination == "fell":
                self.components["fall_penalty"].value = 1
                total_reward += self.components["fall_penalty"].weight

        if self.is_component_enabled("height_deviation_penalty"):
            self.components["height_deviation_penalty"].value = abs(sim.robot_y_position - sim.episode_robot_y_initial_position)
            total_reward += self.components["height_deviation_penalty"].value * self.components["height_deviation_penalty"].weight

        if self.is_component_enabled("height_deviation_square_penalty"):
            self.components["height_deviation_square_penalty"].value = (sim.robot_y_position - sim.episode_robot_y_initial_position) ** 2
            total_reward += self.components["height_deviation_square_penalty"].value * self.components["height_deviation_square_penalty"].weight

        if self.is_component_enabled("success_bonus"):
            if sim.episode_termination == "success":
                self.components["success_bonus"].value = 1
                total_reward += self.components["success_bonus"].weight

        if self.is_component_enabled("effort_penalty"):
            effort = sum(abs(v) for v in sim.joint_velocities)
            self.components["effort_penalty"].value = effort
            total_reward += effort * self.components["effort_penalty"].weight

        if self.is_component_enabled("effort_square_penalty"):
            effort = sum(v**2 for v in sim.joint_velocities)
            self.components["effort_square_penalty"].value = effort
            total_reward += effort * self.components["effort_square_penalty"].weight

        if self.is_component_enabled("direction_change_penalty"):
            action_products = action * sim.episode_last_action  # Números positivos indicam que a direção é a mesma
            direction_changes = np.sum(action_products < 0)  # Conta mudanças de direção
            self.components["direction_change_penalty"].value = direction_changes
            total_reward += direction_changes * self.components["direction_change_penalty"].weight

        if self.is_component_enabled("foot_clearance"):
            foot_height = 0

            if not sim.robot_left_foot_contact:
                foot_height += sim.robot_left_foot_height

            if not sim.robot_right_foot_contact:
                foot_height += sim.robot_right_foot_height

            self.components["foot_clearance"].value = foot_height
            total_reward += self.components["foot_clearance"].value * self.components["foot_clearance"].weight

        if self.is_component_enabled("alternating_foot_contact"):
            self.components["alternating_foot_contact"].value = sim.robot_left_foot_contact != sim.robot_right_foot_contact
            total_reward += self.components["alternating_foot_contact"].value * self.components["alternating_foot_contact"].weight

        if self.is_component_enabled("knee_flexion"):
            self.components["knee_flexion"].value = abs(sim.robot_right_knee_angle) + abs(sim.robot_left_knee_angle)
            total_reward += self.components["knee_flexion"].value * self.components["knee_flexion"].weight

        if self.is_component_enabled("hip_extension"):
            self.components["hip_extension"].value = abs(sim.robot_right_hip_frontal_angle) + abs(sim.robot_left_hip_frontal_angle)
            total_reward += self.components["hip_extension"].value * self.components["hip_extension"].weight

        if self.is_component_enabled("hip_openning"):
            self.components["hip_openning"].value = abs(sim.robot_right_hip_lateral_angle) + abs(sim.robot_left_hip_lateral_angle)
            total_reward += self.components["hip_openning"].value * self.components["hip_openning"].weight

        if self.is_component_enabled("hip_openning_square"):
            self.components["hip_openning_square"].value = sim.robot_right_hip_lateral_angle**2 + sim.robot_left_hip_lateral_angle**2
            total_reward += self.components["hip_openning_square"].value * self.components["hip_openning_square"].weight

        if self.is_component_enabled("jerk_penalty"):
            jerk = sum(abs(v1 - v2) for v1, v2 in zip(sim.joint_velocities, sim.last_joint_velocities))
            self.components["jerk_penalty"].value = jerk
            total_reward += jerk * self.components["jerk_penalty"].weight

        if self.is_component_enabled("y_axis_deviation_penalty"):
            penalty = distance_y_from_center
            self.components["y_axis_deviation_penalty"].value = penalty
            total_reward += penalty * self.components["y_axis_deviation_penalty"].weight

        if self.is_component_enabled("y_axis_deviation_square_penalty"):
            penalty = distance_y_from_center**2
            self.components["y_axis_deviation_square_penalty"].value = penalty
            total_reward += penalty * self.components["y_axis_deviation_square_penalty"].weight

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

    def calculate_reward(self, sim, action):
        """Método principal - escolhe entre DPG progressivo ou padrão"""
        if self.dpg_enabled:
            return self.calculate_dpg_reward(sim, action)
        else:
            return self.calculate_standard_reward(sim, action)

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
            with open(filepath, "r") as f:
                config = json.load(f)
            return self.load_configuration(config, is_default_file)
        except Exception as e:
            self.logger.exception(f"Erro ao carregar arquivo de configuração: {filepath}")
            return False

    def update_component(self, name, weight=None, enabled=None):
        """Atualiza um componente - MANTIDO (já existe)"""
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
    
    def _calculate_cross_gait_pattern(self, sim):
        """Calcula recompensa por padrão de marcha cruzada"""
        # Implementação básica
        left_foot_contact = sim.robot_left_foot_contact
        right_foot_contact = sim.robot_right_foot_contact

        try:
            right_arm_angle = getattr(sim, "robot_right_shoulder_front_angle", 0)
            left_arm_angle = getattr(sim, "robot_left_shoulder_front_angle", 0)
        except:
            return 0.0

        cross_gait_score = 0.0

        if not right_foot_contact and left_arm_angle > 0:
            cross_gait_score += 0.5

        if not left_foot_contact and right_arm_angle > 0:
            cross_gait_score += 0.5

        return max(0.0, cross_gait_score)