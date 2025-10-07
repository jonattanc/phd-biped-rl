# reward_system.py
import datetime
import os
import numpy as np
import json
import time
from dataclasses import dataclass


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

        # Configurações padrão
        self.fall_threshold = 0.5
        self.success_distance = 10.0
        self.platform_width = 1.0
        self.safe_zone = 0.2
        self.warning_zone = 0.4

        # Inicializar componentes padrão
        self._initialize_default_components()

    def _initialize_default_components(self):
        """Inicializa os componentes de recompensa padrão"""
        default_components = [
            # Componentes Críticos (Sobrevivência)
            RewardComponent("progress", 15.0, enabled=True),
            RewardComponent("distance_bonus", 2.0, enabled=True),
            RewardComponent("stability_roll", -0.1, enabled=True),
            RewardComponent("stability_pitch", -0.4, enabled=True),
            RewardComponent("yaw_penalty", -2.0, enabled=True),
            RewardComponent("fall_penalty", -100.0, enabled=True),
            RewardComponent("success_bonus", 200.0, enabled=True),
            # Eficiência
            RewardComponent("effort_penalty", -0.001, enabled=True),
            RewardComponent("jerk_penalty", -0.05, enabled=True),
            # Navegação
            RewardComponent("center_bonus", 5.0, enabled=True),
            RewardComponent("warning_penalty", -3.0, enabled=True),
            # Componentes Avançados (inicialmente desabilitados)
            RewardComponent("gait_regularity", 2.0, enabled=False),
            RewardComponent("symmetry_bonus", 1.0, enabled=False),
            RewardComponent("clearance_bonus", 1.5, enabled=False),
        ]

        for component in default_components:
            self.components[component.name] = component

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
            if self.components["progress"].enabled:
                progress = simulation.episode_distance - simulation.episode_last_distance
                self.components["progress"].value = progress
                total_reward += progress * self.components["progress"].weight

            # 2. Bônus de distância acumulada
            if self.components["distance_bonus"].enabled:
                self.components["distance_bonus"].value = simulation.episode_distance
                total_reward += simulation.episode_distance * self.components["distance_bonus"].weight

            # 3. Estabilidade - Orientação
            roll, pitch, yaw = robot_orientation

            if self.components["stability_roll"].enabled:
                self.components["stability_roll"].value = roll**2
                total_reward += roll**2 * self.components["stability_roll"].weight

            if self.components["stability_pitch"].enabled:
                self.components["stability_pitch"].value = pitch**2
                total_reward += pitch**2 * self.components["stability_pitch"].weight

            # 4. Penalidade por desvio de yaw
            if self.components["yaw_penalty"].enabled and abs(yaw) > 0.35:  # ~20 graus
                self.components["yaw_penalty"].value = 1
                total_reward += self.components["yaw_penalty"].weight

            # 5. Queda
            if robot_position[2] < self.fall_threshold:
                if self.components["fall_penalty"].enabled:
                    self.components["fall_penalty"].value = 1
                    total_reward += self.components["fall_penalty"].weight
                simulation.episode_terminated = True

            # 6. Sucesso
            if simulation.episode_terminated and simulation.episode_success:
                if self.components["success_bonus"].enabled:
                    self.components["success_bonus"].value = 1
                    total_reward += self.components["success_bonus"].weight

            # ===== EFICIÊNCIA =====

            # 7. Penalidade por esforço
            if self.components["effort_penalty"].enabled:
                effort = sum(abs(v) for v in joint_velocities) if joint_velocities else 0.0
                self.components["effort_penalty"].value = effort
                total_reward += effort * self.components["effort_penalty"].weight

            # 8. Penalidade por jerk (estimado)
            if self.components["jerk_penalty"].enabled:
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
                if self.components["center_bonus"].enabled:
                    safe_factor = 1.0 - (distance_from_center / self.safe_zone)
                    self.components["center_bonus"].value = safe_factor
                    total_reward += safe_factor * self.components["center_bonus"].weight

            elif distance_from_center <= self.warning_zone:
                if self.components["warning_penalty"].enabled:
                    warning_factor = (distance_from_center - self.safe_zone) / (self.warning_zone - self.safe_zone)
                    self.components["warning_penalty"].value = warning_factor
                    total_reward += warning_factor * self.components["warning_penalty"].weight

            # ===== COMPONENTES AVANÇADOS =====

            # 10. Regularidade da marcha
            if self.components["gait_regularity"].enabled:
                regularity = self._calculate_gait_regularity(joint_velocities)
                self.components["gait_regularity"].value = regularity
                total_reward += regularity * self.components["gait_regularity"].weight

            # 11. Simetria
            if self.components["symmetry_bonus"].enabled:
                symmetry = self._calculate_symmetry(joint_velocities)
                self.components["symmetry_bonus"].value = symmetry
                total_reward += symmetry * self.components["symmetry_bonus"].weight

            # 12. Clearance (estimado)
            if self.components["clearance_bonus"].enabled:
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

    def get_configuration(self):
        """Retorna configuração atual em formato dicionário"""
        self.logger.info(" RewardSystem.get_configuration called")

        config = {}
        for name, component in self.components.items():
            config[name] = {"weight": component.weight, "enabled": component.enabled, "min_value": component.min_value, "max_value": component.max_value}
        return config

    def load_configuration(self, config_dict):
        """Carrega configuração de um dicionário"""
        self.logger.info(" RewardSystem.load_configuration called")

        try:
            # Carregar componentes
            if "components" in config_dict:
                components_config = config_dict["components"]
                for name, settings in components_config.items():
                    if name in self.components:
                        self.update_component(name, settings.get("weight"), settings.get("enabled"))

            # Carregar configurações globais
            if "global_settings" in config_dict:
                globals_config = config_dict["global_settings"]
                self.fall_threshold = globals_config.get("fall_threshold", self.fall_threshold)
                self.success_distance = globals_config.get("success_distance", self.success_distance)
                self.platform_width = globals_config.get("platform_width", self.platform_width)
                self.safe_zone = globals_config.get("safe_zone", self.safe_zone)
                self.warning_zone = globals_config.get("warning_zone", self.warning_zone)

            self.logger.info("Configuração carregada com sucesso")
            return True

        except Exception as e:
            self.logger.exception("Erro ao carregar configuração")
            return False

    def load_configuration_file(self, filepath):
        """Carrega configuração de arquivo JSON"""
        self.logger.info(f" RewardSystem.load_configuration_file called with {filepath}")

        try:
            with open(filepath, "r") as f:
                config = json.load(f)
            return self.load_configuration(config)
        except Exception as e:
            self.logger.exception("Erro ao carregar arquivo de configuração")
            return False

    def save_configuration(self, filepath, config_data=None):
        """Salva configuração em arquivo JSON"""
        self.logger.info(f" RewardSystem.save_configuration called with {filepath}")

        try:
            if config_data is None:
                config_data = {
                    "metadata": {"name": os.path.basename(filepath).replace(".json", ""), "version": "1.0", "created": datetime.now().isoformat(), "description": "Configuração salva automaticamente"},
                    "global_settings": {
                        "fall_threshold": self.fall_threshold,
                        "success_distance": self.success_distance,
                        "platform_width": self.platform_width,
                        "safe_zone": self.safe_zone,
                        "warning_zone": self.warning_zone,
                    },
                    "components": self.get_configuration(),
                }

            with open(filepath, "w") as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Configuração salva em: {filepath}")
            return True

        except Exception as e:
            self.logger.exception("Erro ao salvar configuração")
            return False

    def update_component(self, name, weight=None, enabled=None):
        """Atualiza um componente - MANTIDO (já existe)"""
        self.logger.info(f" RewardSystem.update_component called for {name}")

        if name in self.components:
            if weight is not None:
                self.components[name].weight = weight
            if enabled is not None:
                self.components[name].enabled = enabled
            return True
        return False

    def get_available_configurations(self):
        """Lista configurações disponíveis"""
        self.logger.info(" RewardSystem.get_available_configurations called")

        configs = []
        base_dir = "reward_configs"

        if not os.path.exists(base_dir):
            return configs

        # Configurações na raiz
        for file in os.listdir(base_dir):
            if file.endswith(".json") and file != "active.json":
                configs.append(file.replace(".json", ""))

        # Configurações em subdiretórios
        for category in ["training", "experiments"]:
            category_dir = os.path.join(base_dir, category)
            if os.path.exists(category_dir):
                for file in os.listdir(category_dir):
                    if file.endswith(".json"):
                        configs.append(f"{category}/{file.replace('.json', '')}")

        return configs

    def activate_configuration(self, config_name):
        """Ativa uma configuração específica"""
        self.logger.info(f" RewardSystem.activate_configuration called with {config_name}")

        try:
            # CORREÇÃO: Remover .json se já estiver presente
            if config_name.endswith(".json"):
                config_name = config_name[:-5]

            # Determinar caminho completo
            if "/" in config_name:
                category, name = config_name.split("/")
                config_path = f"{category}/{name}.json"
            else:
                config_path = f"{config_name}.json"

            full_path = os.path.join("reward_configs", config_path)

            if os.path.exists(full_path):
                # Criar arquivo de configuração ativa
                active_info = {"active_config": config_path, "activated_at": datetime.now().isoformat(), "name": config_name}

                active_config_path = os.path.join("reward_configs", "active.json")
                with open(active_config_path, "w") as f:
                    json.dump(active_info, f, indent=2)

                # Carregar a configuração
                success = self.load_configuration_file(full_path)
                if success:
                    self.logger.info(f"Configuração ativada: {config_name}")
                    return True
            else:
                self.logger.error(f"Arquivo de configuração não encontrado: {full_path}")
                return False

        except Exception as e:
            self.logger.exception("Falha ao ativar configuração")
            return False

    def load_active_configuration(self):
        """Carrega automaticamente a configuração ativa com verificações robustas"""
        self.logger.info(" RewardSystem.load_active_configuration called")

        try:
            # Primeiro verificar se o diretório existe
            configs_dir = "reward_configs"
            if not os.path.exists(configs_dir):
                self.logger.warning(f"Diretório de configurações não encontrado: {configs_dir}")
                self.logger.info("Criando diretório de configurações...")
                os.makedirs(configs_dir, exist_ok=True)
                return False

            self.logger.info(f"Diretório de configurações encontrado: {configs_dir}")
            self.logger.info(f"Conteúdo: {os.listdir(configs_dir)}")

            active_config_path = os.path.join(configs_dir, "active.json")

            # Se active.json não existe, usar padrão interno
            if not os.path.exists(active_config_path):
                self.logger.info("Nenhuma configuração ativa encontrada (active.json não existe)")
                return False

            self.logger.info("Arquivo active.json encontrado")

            # Carregar informação da configuração ativa
            with open(active_config_path, "r") as f:
                active_info = json.load(f)

            config_path = active_info.get("active_config", "default.json")

            # CORREÇÃO: Remover .json se já estiver presente para evitar duplicação
            if config_path.endswith(".json"):
                config_name = config_path[:-5]  # Remove .json
            else:
                config_name = config_path

            # CORREÇÃO: Construir o caminho corretamente
            full_path = os.path.join(configs_dir, f"{config_name}.json")

            self.logger.info(f"Tentando carregar: {full_path}")

            # Verificar se o arquivo de configuração existe
            if os.path.exists(full_path):
                success = self.load_configuration_file(full_path)
                if success:
                    self.logger.info(f"Configuração ativa carregada com sucesso: {config_name}")
                    return True
                else:
                    self.logger.error(f"Falha ao carregar configuração: {config_name}")
                    return False
            else:
                self.logger.warning(f"Arquivo de configuração não encontrado: {full_path}")
                self.logger.info(f"Arquivos em {configs_dir}: {os.listdir(configs_dir)}")
                return False

        except Exception as e:
            self.logger.exception("Erro ao carregar configuração ativa")
            # Em caso de erro, usar configuração padrão interna
            self.logger.info("Usando configuração padrão interna devido a erro")
            return False

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
