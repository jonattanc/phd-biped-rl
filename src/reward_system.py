# reward_system.py
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
            if not hasattr(self, "gait_phase_dpg"):
                from dpg_gait_phases import GaitPhaseDPG

                self.gait_phase_dpg = GaitPhaseDPG(self.logger, self)

            self.gait_phase_dpg._apply_phase_config()  # Aplicar configuração inicial
            self.logger.info("DPG com Fases da Marcha ativado")

            if hasattr(self, "phase_detector") and self.phase_detector:
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
        if hasattr(self, "dpg_manager") and self.dpg_manager and self.dpg_manager.config.enabled:
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

        # COMPONENTES PARA MARCHA

        # 1. PROGRESSO E VELOCIDADE
        if self.is_component_enabled("progress"):
            progress = sim.target_x_velocity - abs(sim.target_x_velocity - sim.robot_x_velocity)
            self.components["progress"].value = progress
            total_reward += progress * self.components["progress"].weight

        # 2. ESTABILIDADE DA MARCHA (Controle postural)
        if self.is_component_enabled("stability_pitch"):
            pitch_error = (sim.robot_pitch - sim.target_pitch_rad) ** 2
            self.components["stability_pitch"].value = pitch_error
            total_reward += pitch_error * self.components["stability_pitch"].weight

        if self.is_component_enabled("stability_roll"):
            roll_error = sim.robot_roll**2
            self.components["stability_roll"].value = roll_error
            total_reward += roll_error * self.components["stability_roll"].weight

        # 3. PADRÃO DE MARCHA CRUZADA (Coordenação braço-perna)
        if self.is_component_enabled("gait_pattern_cross"):
            cross_gait_score = self._calculate_cross_gait_pattern(sim)
            self.components["gait_pattern_cross"].value = cross_gait_score
            total_reward += cross_gait_score * self.components["gait_pattern_cross"].weight

        # 4. ALTERNÂNCIA DE PASSOS (Critério fundamental da marcha)
        if self.is_component_enabled("alternating_foot_contact"):
            alternation = sim.robot_left_foot_contact != sim.robot_right_foot_contact
            self.components["alternating_foot_contact"].value = alternation
            total_reward += alternation * self.components["alternating_foot_contact"].weight

        # 5. ALTURA ADEQUADA DOS PÉS (Clearance durante o balanço)
        if self.is_component_enabled("foot_clearance_original"):
            foot_height = 0

            if not sim.robot_left_foot_contact:
                foot_height += sim.robot_left_foot_height

            if not sim.robot_right_foot_contact:
                foot_height += sim.robot_right_foot_height

            self.components["foot_clearance_original"].value = foot_height
            total_reward += self.components["foot_clearance_original"].value * self.components["foot_clearance_original"].weight

        if self.is_component_enabled("foot_clearance"):
            clearance_score = self._calculate_foot_clearance_optimized(sim)
            self.components["foot_clearance"].value = clearance_score
            total_reward += clearance_score * self.components["foot_clearance"].weight

        # 6. PADRÃO RÍTMICO (Regularidade da marcha)
        if self.is_component_enabled("gait_rhythm"):
            rhythm_score = self._calculate_gait_rhythm(sim)
            self.components["gait_rhythm"].value = rhythm_score
            total_reward += rhythm_score * self.components["gait_rhythm"].weight

        # 7. EFICIÊNCIA ENERGÉTICA (Movimentos suaves)
        if self.is_component_enabled("effort_square_penalty"):
            effort = sum(v**2 for v in sim.joint_velocities)
            self.components["effort_square_penalty"].value = effort
            total_reward += effort * self.components["effort_square_penalty"].weight

        # 8. CONTROLE DE TRAJETÓRIA (Manter direção)
        if self.is_component_enabled("yaw_penalty"):
            if sim.episode_termination == "yaw_deviated":
                self.components["yaw_penalty"].value = 1
                total_reward += self.components["yaw_penalty"].weight

        # 9. ESTABILIDADE VERTICAL (Controle de altura do corpo)
        if self.is_component_enabled("height_deviation_square_penalty"):
            height_error = (sim.robot_z_position - 0.8) ** 2  # Altura ideal ~0.8m
            self.components["height_deviation_square_penalty"].value = height_error
            total_reward += height_error * self.components["height_deviation_square_penalty"].weight

        # 10. PENALIDADES POR QUEDA (Segurança)
        if self.is_component_enabled("fall_penalty"):
            if sim.episode_termination == "fell":
                self.components["fall_penalty"].value = 1
                total_reward += self.components["fall_penalty"].weight

        # DEMAIS COMPONENTES

        # Transições de estado
        if self.is_component_enabled("gait_state_change"):
            self.components["gait_state_change"].value = sim.has_gait_state_changed
            total_reward += self.components["gait_state_change"].value * self.components["gait_state_change"].weight

        # Bonus de distância
        if self.is_component_enabled("distance_bonus"):
            self.components["distance_bonus"].value = sim.episode_distance
            total_reward += sim.episode_distance * self.components["distance_bonus"].weight

        # Inclinação frontal
        if self.is_component_enabled("pitch_forward_bonus"):
            target_forward_pitch = 0.05
            pitch_error = abs(sim.robot_pitch - target_forward_pitch)
            pitch_bonus = max(0, 0.1 - pitch_error)
            self.components["pitch_forward_bonus"].value = pitch_bonus
            total_reward += pitch_bonus * self.components["pitch_forward_bonus"].weight

        if self.is_component_enabled("stability_yaw"):
            self.components["stability_yaw"].value = sim.robot_yaw**2
            total_reward += sim.robot_yaw**2 * self.components["stability_yaw"].weight

        if self.is_component_enabled("height_deviation_penalty"):
            self.components["height_deviation_penalty"].value = abs(sim.robot_y_position - sim.episode_robot_y_initial_position)
            total_reward += self.components["height_deviation_penalty"].value * self.components["height_deviation_penalty"].weight

        if self.is_component_enabled("success_bonus"):
            if sim.episode_termination == "success":
                self.components["success_bonus"].value = 1
                total_reward += self.components["success_bonus"].weight

        if self.is_component_enabled("effort_penalty"):
            effort = sum(abs(v) for v in sim.joint_velocities)
            self.components["effort_penalty"].value = effort
            total_reward += effort * self.components["effort_penalty"].weight

        # Evita mudanças de direção
        if self.is_component_enabled("direction_change_penalty"):
            action_products = action * sim.episode_last_action
            direction_changes = np.sum(action_products < 0)
            self.components["direction_change_penalty"].value = direction_changes
            total_reward += direction_changes * self.components["direction_change_penalty"].weight

        if self.is_component_enabled("foot_back_penalty"):
            backwards_velocity = 0

            if sim.robot_left_foot_x_velocity < 0:
                backwards_velocity += abs(sim.robot_left_foot_x_velocity)

            if sim.robot_right_foot_x_velocity < 0:
                backwards_velocity += abs(sim.robot_right_foot_x_velocity)

            self.components["foot_back_penalty"].value = backwards_velocity
            total_reward += self.components["foot_back_penalty"].value * self.components["foot_back_penalty"].weight

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

    def _calculate_cross_gait_pattern(self, sim):
        """Calcula recompensa por padrão de marcha cruzada (contralateral)"""

        left_foot_contact = sim.robot_left_foot_contact
        right_foot_contact = sim.robot_right_foot_contact

        try:
            # Para braço direito: ângulo positivo = para trás, negativo = para frente
            right_arm_angle = getattr(sim, "robot_right_shoulder_front_angle", 0)
            left_arm_angle = getattr(sim, "robot_left_shoulder_front_angle", 0)
        except:
            return 0.0

        cross_gait_score = 0.0

        # Perna direita no ar + braço esquerdo para trás
        if not right_foot_contact and left_arm_angle > 0:
            cross_gait_score += 0.5

        # Perna esquerda no ar + braço direito para trás
        if not left_foot_contact and right_arm_angle > 0:
            cross_gait_score += 0.5

        # Perna direita no ar + braço direito para trás (errado)
        if not right_foot_contact and right_arm_angle > 0:
            cross_gait_score -= 0.3

        # Perna esquerda no ar + braço esquerdo para trás (errado)
        if not left_foot_contact and left_arm_angle > 0:
            cross_gait_score -= 0.3

        return max(0.0, cross_gait_score)

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
            with open(filepath, "r") as f:
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