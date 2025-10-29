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
        self.dpg_enabled = False
        self.phase = 1
        self.dpg_weights = np.ones(4)

        self.safe_zone = 0.2  # m
        self.warning_zone = 0.4  # m

        self.load_configuration_file("default.json", is_default_file=True)
        self.default_components = self.get_configuration_as_dict()

    def is_component_enabled(self, name):
        if name not in self.components:
            return False

        return self.components[name].enabled

    def enable_dpg_progression(self, enabled=True):
        """Ativa/desativa progressão por fases para DPG - VERSÃO SIMPLIFICADA"""
        self.dpg_enabled = enabled
        if enabled:
            self.logger.info("DPG Simplificado ativado - 2 fases apenas")
            self.phase = 1
            # Pesos iniciais: foco em estabilidade
            self.dpg_weights = np.array([0.3, 0.4, 0.2, 0.1])
        else:
            self.logger.info("Sistema de recompensa padrão")

    def update_progression(self, episode_results):
        """Atualiza progressão baseada em resultados - VERSÃO SIMPLIFICADA"""
        if not self.dpg_enabled:
            return

        # APENAS 2 FASES SIMPLES
        if episode_results["distance"] > 4.0 and self.phase == 1:
            self.phase = 2
            # Fase 2: mais foco em velocidade
            self.dpg_weights = np.array([0.5, 0.3, 0.1, 0.1])
            self.logger.info("Fase 2 ativada: Priorizando velocidade")

    def create_hybrid_reward_vector(self, sim, action, weights=None):
        """Cria vetor de recompensas com normalização"""
        components = np.zeros(4)

        # Componente 0: Progresso (normalizado)
        max_expected_velocity = 2.0  # m/s
        components[0] = np.clip(getattr(sim, "robot_x_velocity", 0) / max_expected_velocity, -1, 1)

        # Componente 1: Estabilidade (normalizada)
        robot_roll = getattr(sim, "robot_roll", 0)
        robot_pitch = getattr(sim, "robot_pitch", 0)
        robot_yaw = getattr(sim, "robot_yaw", 0)
        target_pitch = getattr(sim, "target_pitch_rad", 0)

        # Normalizar penalidades de estabilidade
        max_angle_error = 0.5  # ~28 graus
        stability_penalty = robot_roll**2 + (robot_pitch - target_pitch) ** 2 + robot_yaw**2
        components[1] = -np.clip(stability_penalty / max_angle_error, 0, 1)

        # Componente 2: Eficiência (normalizada)
        joint_velocities = getattr(sim, "joint_velocities", [0])
        max_expected_effort = 10.0  # Valor baseado em sua simulação
        effort = sum(abs(v) for v in joint_velocities) / len(joint_velocities) if joint_velocities else 0
        components[2] = -np.clip(effort / max_expected_effort, 0, 1)

        # Componente 3: Postura (normalizada)
        robot_z = getattr(sim, "robot_z_position", 0.8)
        max_height_error = 0.2  # 20cm
        height_penalty = abs(robot_z - 0.8)
        components[3] = -np.clip(height_penalty / max_height_error, 0, 1)

        # Aplicar pesos com verificação
        if weights is not None and len(weights) == len(components):
            weighted_components = components * weights
        else:
            weighted_components = components

        return weighted_components, components

    def calculate_dpg_reward(self, sim, action):
        """Calcula recompensa com DPG simplificado"""
        # Atualizar progressão
        episode_results = {
            "distance": sim.episode_distance, 
            "success": sim.episode_success, 
            "duration": sim.episode_steps * sim.time_step_s
        }
        self.update_progression(episode_results)
    
        # Calcular recompensa com pesos do DPG
        weighted_reward, components = self.create_hybrid_reward_vector(sim, action, self.dpg_weights)
        total_reward = np.sum(weighted_reward)
    
        # Bônus simples de fase
        if self.phase == 2 and sim.episode_distance > 4.0:
            total_reward += 0.3
    
        return total_reward

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

        if self.is_component_enabled("gait_regularity"):
            raise NotImplementedError("gait_regularity component is not implemented.")
            regularity = self._calculate_gait_regularity(sim.joint_velocities)
            self.components["gait_regularity"].value = regularity
            total_reward += regularity * self.components["gait_regularity"].weight

        if self.is_component_enabled("symmetry_bonus"):
            raise NotImplementedError("symmetry_bonus component is not implemented.")
            symmetry = self._calculate_symmetry(sim.joint_velocities)
            self.components["symmetry_bonus"].value = symmetry
            total_reward += symmetry * self.components["symmetry_bonus"].weight

        if self.is_component_enabled("clearance_bonus"):
            raise NotImplementedError("clearance_bonus component is not implemented.")
            clearance_ok = self._estimate_foot_clearance(sim.joint_positions)
            self.components["clearance_bonus"].value = clearance_ok
            total_reward += clearance_ok * self.components["clearance_bonus"].weight

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
        """Calcula recompensa por padrão de marcha cruzada (contralateral)"""

        # Estados dos pés (True = no chão, False = no ar)
        left_foot_contact = sim.robot_left_foot_contact
        right_foot_contact = sim.robot_right_foot_contact

        # Ângulos dos braços (assumindo que shoulder_front controla o balanço frontal)
        try:
            # Para braço direito: ângulo positivo = para trás, negativo = para frente
            right_arm_angle = getattr(sim, "robot_right_shoulder_front_angle", 0)
            left_arm_angle = getattr(sim, "robot_left_shoulder_front_angle", 0)
        except:
            # Fallback se os ângulos não estiverem disponíveis
            return 0.0

        # Padrão de marcha cruzada ideal:
        # - Quando perna DIREITA está no ar → braço ESQUERDO deve estar para trás (ângulo positivo)
        # - Quando perna ESQUERDA está no ar → braço DIREITO deve estar para trás (ângulo positivo)

        cross_gait_score = 0.0

        # Perna direita no ar + braço esquerdo para trás
        if not right_foot_contact and left_arm_angle > 0:
            cross_gait_score += 0.5

        # Perna esquerda no ar + braço direito para trás
        if not left_foot_contact and right_arm_angle > 0:
            cross_gait_score += 0.5

        # Penalizar padrão incorreto (marcha homolateral)
        # Perna direita no ar + braço direito para trás (errado)
        if not right_foot_contact and right_arm_angle > 0:
            cross_gait_score -= 0.3

        # Perna esquerda no ar + braço esquerdo para trás (errado)
        if not left_foot_contact and left_arm_angle > 0:
            cross_gait_score -= 0.3

        return max(0.0, cross_gait_score)

    def _calculate_symmetry(self, joint_velocities):
        """Calcula simetria entre lados do robô"""
        if not joint_velocities or len(joint_velocities) < 4:  # Precisa de pelo menos 2 juntas por lado
            return 0.5

        # Assumindo que as primeiras juntas são de um lado e as seguintes do outro
        mid = len(joint_velocities) // 2
        left_effort = np.sqrt(sum(v**2 for v in joint_velocities[:mid]))
        right_effort = np.sqrt(sum(v**2 for v in joint_velocities[mid : mid * 2]))

        if left_effort + right_effort == 0:
            return 0.5

        symmetry = 1.0 - abs(left_effort - right_effort) / (left_effort + right_effort)
        return symmetry

    def _estimate_foot_clearance(self, joint_positions):
        """Estima se o clearance do pé está adequado (simplificado)"""

        # Verificar se alguma junta está em posição extrema (proxy para pé no chão)
        extreme_positions = sum(1 for pos in joint_positions if abs(pos) > 0.8)

        if extreme_positions < len(joint_positions) * 0.5:
            return 1

        else:
            return 0
