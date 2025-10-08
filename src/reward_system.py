# reward_system.py
import os
import numpy as np
import json
import time
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
        self.history = []
        self.current_episode = 0
        self.episode_data = []
        self.components = {}

        # Configurações padrão
        self.fall_threshold = 0.5  # m
        self.success_distance = 9.0  # m
        self.platform_width = 1.0  # m
        self.safe_zone = 0.2  # m
        self.warning_zone = 0.4  # m
        self.yaw_threshold = 0.5  # rad

        self.load_configuration_file("default.json", is_default_file=True)
        self.default_components = self.get_configuration_as_dict()

    def is_component_enabled(self, name):
        if name not in self.components:
            return False

        return self.components[name].enabled

    def calculate_reward(self, simulation, action, robot_state, env_conditions=None):
        """Calcula a recompensa total baseada nos componentes ativos"""
        if env_conditions is None:
            env_conditions = {"foot_slip": 0.0, "ramp_speed": 0.0, "com_drop": 0.0, "joint_failure": False}

        # Resetar valores dos componentes
        for component in self.components.values():
            component.value = 0.0

        total_reward = 0.0
        reward_breakdown = {}

        try:
            # Dados básicos sempre disponíveis - COM VERIFICAÇÃO DE SEGURANÇA
            robot_position, robot_orientation = simulation.robot.get_imu_position_and_orientation()
            joint_positions, joint_velocities = simulation.robot.get_joint_states()

            # VERIFICAÇÃO CRÍTICA: garantir que os dados não são None
            if robot_position is None or robot_orientation is None:
                self.logger.warning("Dados do robô não disponíveis, retornando recompensa zero")
                return 0.0

            if joint_positions is None:
                joint_positions = []
            if joint_velocities is None:
                joint_velocities = []

            # ===== COMPONENTES CRÍTICOS =====

            # 1. Progresso
            if self.is_component_enabled("progress"):
                progress = simulation.episode_distance - simulation.episode_last_distance
                self.components["progress"].value = progress
                total_reward += progress * self.components["progress"].weight

            # 2. Bônus de distância acumulada
            if self.is_component_enabled("distance_bonus"):
                self.components["distance_bonus"].value = simulation.episode_distance
                total_reward += simulation.episode_distance * self.components["distance_bonus"].weight

            # 3. Estabilidade - Orientação
            roll, pitch, yaw = robot_orientation

            if self.is_component_enabled("stability_roll"):
                self.components["stability_roll"].value = roll**2
                total_reward += roll**2 * self.components["stability_roll"].weight

            if self.is_component_enabled("stability_pitch"):
                self.components["stability_pitch"].value = pitch**2
                total_reward += pitch**2 * self.components["stability_pitch"].weight

            # 4. Penalidade por desvio de yaw
            if self.is_component_enabled("yaw_penalty") and abs(yaw) > 0.35:  # ~20 graus
                self.components["yaw_penalty"].value = 1
                total_reward += self.components["yaw_penalty"].weight

            # 5. Queda
            if robot_position[2] < self.fall_threshold:
                if self.is_component_enabled("fall_penalty"):
                    self.components["fall_penalty"].value = 1
                    total_reward += self.components["fall_penalty"].weight
                simulation.episode_terminated = True

            # 6. Sucesso
            if simulation.episode_terminated and simulation.episode_success:
                if self.is_component_enabled("success_bonus"):
                    self.components["success_bonus"].value = 1
                    total_reward += self.components["success_bonus"].weight

            # ===== EFICIÊNCIA =====

            # 7. Penalidade por esforço
            if self.is_component_enabled("effort_penalty"):
                effort = sum(abs(v) for v in joint_velocities) if joint_velocities else 0.0
                self.components["effort_penalty"].value = effort
                total_reward += effort * self.components["effort_penalty"].weight

            # 8. Penalidade por jerk (estimado)
            if self.is_component_enabled("jerk_penalty"):
                # Estimativa simples de jerk - diferença de velocidades
                if hasattr(simulation, "last_joint_velocities") and simulation.last_joint_velocities is not None and joint_velocities:
                    jerk = sum(abs(v1 - v2) for v1, v2 in zip(joint_velocities, simulation.last_joint_velocities))
                    self.components["jerk_penalty"].value = jerk
                    total_reward += jerk * self.components["jerk_penalty"].weight
                simulation.last_joint_velocities = joint_velocities.copy() if joint_velocities else []

            # ===== NAVEGAÇÃO =====

            # 9. Manutenção no centro
            pos_y = robot_position[1]
            distance_from_center = abs(pos_y)

            if distance_from_center <= self.safe_zone:
                if self.is_component_enabled("center_bonus"):
                    safe_factor = 1.0 - (distance_from_center / self.safe_zone)
                    self.components["center_bonus"].value = safe_factor
                    total_reward += safe_factor * self.components["center_bonus"].weight

            elif distance_from_center <= self.warning_zone:
                if self.is_component_enabled("warning_penalty"):
                    warning_factor = (distance_from_center - self.safe_zone) / (self.warning_zone - self.safe_zone)
                    self.components["warning_penalty"].value = warning_factor
                    total_reward += warning_factor * self.components["warning_penalty"].weight

            # ===== COMPONENTES AVANÇADOS =====

            # 10. Regularidade da marcha
            if self.is_component_enabled("gait_regularity"):
                regularity = self._calculate_gait_regularity(joint_velocities)
                self.components["gait_regularity"].value = regularity
                total_reward += regularity * self.components["gait_regularity"].weight

            # 11. Simetria
            if self.is_component_enabled("symmetry_bonus"):
                symmetry = self._calculate_symmetry(joint_velocities)
                self.components["symmetry_bonus"].value = symmetry
                total_reward += symmetry * self.components["symmetry_bonus"].weight

            # 12. Clearance (estimado)
            if self.is_component_enabled("clearance_bonus"):
                clearance_ok = self._estimate_foot_clearance(joint_positions)
                self.components["clearance_bonus"].value = 1 if clearance_ok else 0
                total_reward += self.components["clearance_bonus"].weight if clearance_ok else 0

            # Coletar breakdown para análise
            for name, component in self.components.items():
                if component.enabled:
                    reward_breakdown[name] = {"value": component.value, "weighted_contribution": component.value * component.weight, "weight": component.weight}

            # Registrar para análise
            self._record_step_data(reward_breakdown, total_reward, simulation.episode_steps)

        except Exception as e:
            self.logger.exception("Erro no cálculo de recompensa")
            # Fallback para recompensa básica
            total_reward = (simulation.episode_distance - simulation.episode_last_distance) * 10.0

        return total_reward

    def get_configuration_as_dict(self):
        """Retorna configuração atual em formato dicionário"""
        self.logger.info(" RewardSystem.get_configuration_as_dict called")

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
            self.logger.error(f"Componente {name} não encontrado")
            return False

        if weight is not None:
            self.components[name].weight = weight
            self.logger.info(f"weight atualizado para {weight}")

        if enabled is not None:
            self.components[name].enabled = enabled
            self.logger.info(f"enabled atualizado para {enabled}")

        return True

    def _calculate_gait_regularity(self, joint_velocities):
        """Calcula regularidade da marcha baseado na variância das velocidades"""
        if not joint_velocities:  # Verifica se está vazio ou None
            return 0.5

        try:
            variance = np.var(joint_velocities)
            # Normalizar para 0-1 (menor variância = mais regular)
            regularity = 1.0 / (1.0 + variance)
            return regularity
        except Exception as e:
            self.logger.exception("Erro. Retornando valor padrão para regularidade da marcha")
            return 0.5

    def _calculate_symmetry(self, joint_velocities):
        """Calcula simetria entre lados do robô"""
        if not joint_velocities or len(joint_velocities) < 4:  # Precisa de pelo menos 2 juntas por lado
            return 0.5

        try:
            # Assumindo que as primeiras juntas são de um lado e as seguintes do outro
            mid = len(joint_velocities) // 2
            left_effort = np.sqrt(sum(v**2 for v in joint_velocities[:mid]))
            right_effort = np.sqrt(sum(v**2 for v in joint_velocities[mid : mid * 2]))

            if left_effort + right_effort == 0:
                return 0.5

            symmetry = 1.0 - abs(left_effort - right_effort) / (left_effort + right_effort)
            return symmetry
        except Exception as e:
            self.logger.exception("Erro. Retornando valor padrão para simetria")
            return 0.5

    def _estimate_foot_clearance(self, joint_positions):
        """Estima se o clearance do pé está adequado (simplificado)"""
        if not joint_positions:
            return True

        try:
            # Verificar se alguma junta está em posição extrema (proxy para pé no chão)
            extreme_positions = sum(1 for pos in joint_positions if abs(pos) > 0.8)
            return extreme_positions < len(joint_positions) * 0.5
        except Exception as e:
            self.logger.exception("Erro. Retornando valor padrão para clearance do pé")
            return True

    def _record_step_data(self, reward_breakdown, total_reward, step):
        """Registra dados do step para análise"""
        step_data = {"episode": self.current_episode, "step": step, "total_reward": total_reward, "breakdown": reward_breakdown.copy(), "timestamp": time.time()}
        self.episode_data.append(step_data)

    def start_episode(self, episode_number):
        """Inicia um novo episódio"""
        self.current_episode = episode_number
        self.episode_data = []

    def end_episode(self):
        """Finaliza o episódio e salva dados"""
        if self.episode_data:
            episode_summary = {
                "episode": self.current_episode,
                "total_steps": len(self.episode_data),
                "total_reward": sum(step["total_reward"] for step in self.episode_data),
                "component_contributions": self._calculate_episode_contributions(),
                "data": self.episode_data.copy(),
            }
            self.history.append(episode_summary)

    def _calculate_episode_contributions(self):
        """Calcula contribuição total de cada componente no episódio"""
        contributions = {}
        for step_data in self.episode_data:
            for component_name, data in step_data["breakdown"].items():
                if component_name not in contributions:
                    contributions[component_name] = 0.0
                contributions[component_name] += data["weighted_contribution"]
        return contributions
