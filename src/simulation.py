# simulation.py
import pybullet as p
import gymnasium as gym
import time
import numpy as np
import random
import math
import queue
import os
import torch
from dpg_phase import PhaseTransitionResult
import best_model_tracker
import utils


class Simulation(gym.Env):
    def __init__(
        self,
        logger,
        robot,
        environment,
        reward_system,
        ipc_queue,
        ipc_queue_main_to_process,
        pause_value,
        exit_value,
        enable_visualization_value,
        enable_real_time_value,
        camera_selecion_value,
        config_changed_value,
        seed,
        num_episodes=1,
        initial_episode=0,
    ):
        super(Simulation, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.robot = robot
        self.environment = environment
        self.ipc_queue = ipc_queue
        self.ipc_queue_main_to_process = ipc_queue_main_to_process
        self.pause_value = pause_value
        self.exit_value = exit_value
        self.enable_visualization_value = enable_visualization_value
        self.is_visualization_enabled = enable_visualization_value.value
        self.enable_real_time_value = enable_real_time_value
        self.is_real_time_enabled = enable_real_time_value.value
        self.camera_selection_value = camera_selecion_value
        self.last_selected_camera = self.camera_selection_value.value
        self.config_changed_value = config_changed_value
        self.num_episodes = num_episodes
        self.episode_count = initial_episode
        self.total_steps = 0

        self.logger = logger
        self.agent = None
        self.physics_client = None
        self.com_marker = None
        self.reward_system = reward_system
        self.should_save_model = False

        self.tracker = best_model_tracker.BestModelTracker(self)

        # Configurações de simulação
        self.target_pitch_rad = math.radians(1)  # rad
        self.target_x_velocity = 2.0  # m/s
        self.fall_threshold = 0.5  # m
        self.success_distance = 9.0  # m
        self.yaw_threshold = math.radians(60)  # rad
        self.episode_training_timeout_s = 20  # s
        self.episode_pre_fill_timeout_s = 10  # s
        self.episode_timeout_s = self.episode_training_timeout_s
        self.physics_step_s = 1 / 240.0  # 240 Hz, ~4.16 ms
        self.physics_step_multiplier = 8
        self.time_step_s = self.physics_step_s * self.physics_step_multiplier  # 240/5 = 48 Hz, ~20.83 ms # 240/8 = 30 Hz, ~33.33 ms # 240/10 = 24 Hz, ~41.66 ms
        self.max_motor_velocity = 2.0  # rad/s
        self.max_motor_torque = 130.0  # Nm
        self.max_training_steps = int(self.episode_training_timeout_s / self.time_step_s)
        self.max_pre_fill_steps = int(self.episode_pre_fill_timeout_s / self.time_step_s)
        self.max_steps = self.max_training_steps

        self.lateral_friction = 2.0
        self.spinning_friction = 1.0
        self.rolling_friction = 0.001
        self.restitution = 0.0

        # Configurar ambiente de simulação PRIMEIRO
        self.setup_sim_env()

        # AGORA podemos obter as informações do robô carregado
        self.action_dim = self.robot.get_num_revolute_joints()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.observation_dim = len(self.robot.get_observation())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

        self.logger.info(f"Simulação configurada: {self.robot.name} no {self.environment.name}")
        self.logger.info(f"Robô: {self.robot.name}")
        self.logger.info(f"DOF: {self.action_dim}")
        self.logger.info(f"Ambiente: {self.environment.name}")
        self.logger.info(f"Visualização: {self.enable_visualization_value.value}")
        self.logger.info(f"Tempo Real: {self.enable_real_time_value.value}")
        self.logger.info(f"Action space: {self.action_dim}, Observation space: {self.observation_dim}")

        # Variáveis para coleta de dados
        self.reset_episode_vars()

    def create_com_marker(self):
        com_pos = self.robot.get_center_of_mass()

        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
        self.com_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=com_pos)

        self.com_line_id = p.addUserDebugLine(lineFromXYZ=com_pos, lineToXYZ=[com_pos[0], com_pos[1], 0], lineColorRGB=[1, 0, 0], lineWidth=2.0)

    def update_com_marker(self):
        com_pos = self.robot.get_center_of_mass()

        p.resetBasePositionAndOrientation(self.com_marker, com_pos, [0, 0, 0, 1])

        self.com_line_id = p.addUserDebugLine(lineFromXYZ=com_pos, lineToXYZ=[com_pos[0], com_pos[1], 0], lineColorRGB=[1, 0, 0], lineWidth=2.0, replaceItemUniqueId=self.com_line_id)

    def setup_sim_env(self):
        """Conecta ao PyBullet e carrega ambiente e robô"""
        if self.physics_client is not None:
            p.disconnect()

        # Usar visualização apenas se estiver habilitada
        if self.is_visualization_enabled:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.last_selected_camera = self.camera_selection_value.value

        if self.last_selected_camera == 1:  # Ambiente geral
            p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=35, cameraPitch=-45, cameraTargetPosition=[6.0, 0.0, 0.6])

        elif self.last_selected_camera == 2:  # Robô - Diagonal direita
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=40, cameraPitch=-25, cameraTargetPosition=[0.8, 0.0, 0.0])

        elif self.last_selected_camera == 3:  # Robô - Diagonal esquerda
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=140, cameraPitch=-25, cameraTargetPosition=[0.8, 0.0, 0.0])

        elif self.last_selected_camera == 4:  # Robô - Lateral direita
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-5, cameraTargetPosition=[0.5, 0.0, 0.0])

        elif self.last_selected_camera == 5:  # Robô - Lateral esquerda
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=180, cameraPitch=-5, cameraTargetPosition=[0.5, 0.0, 0.0])

        elif self.last_selected_camera == 6:  # Robô - Frontal
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, cameraPitch=-5, cameraTargetPosition=[0.0, 0.0, 0.5])

        elif self.last_selected_camera == 7:  # Robô - Traseira
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-90, cameraPitch=-5, cameraTargetPosition=[0.0, 0.0, 0.5])

        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.physics_step_s)

        p.setPhysicsEngineParameter(enableConeFriction=1)

        # Carregar ambiente primeiro
        self.environment.load_in_simulation(use_fixed_base=True)

        # Carregar robô após o ambiente
        self.robot.load_in_simulation()

        self.create_com_marker()

        if self.is_visualization_enabled:
            for i in range(5):  # Realiza diversas iterações para focar a janela assim que possível
                time.sleep(0.1)  # Aguarda a inicialização da janela do PyBullet
                self.ipc_queue.put({"type": "pybullet_window_ready"})

    def set_agent(self, agent):
        self.agent = agent

    def pre_fill_buffer(self):
        dpg_enabled = False
        if hasattr(self.reward_system, "dpg_manager") and self.reward_system.dpg_manager:
            dpg_enabled = self.reward_system.dpg_manager.config.enabled
        if dpg_enabled:
            return

        obs = self.reset()

        self.episode_timeout_s = self.episode_pre_fill_timeout_s
        self.max_steps = self.max_pre_fill_steps

        while self.total_steps < self.agent.prefill_steps and not self.exit_value.value:
            t = self.episode_steps * self.time_step_s
            action = self.robot.get_example_action(t)

            next_obs, reward, episode_terminated, episode_truncated, info = self.step(action)
            done = episode_terminated or episode_truncated

            if isinstance(obs, (list, tuple)):
                obs = np.concatenate([np.ravel(o) for o in obs if not isinstance(o, dict)])
            else:
                obs = np.array(obs).flatten()

            if isinstance(next_obs, (list, tuple)):
                next_obs = np.concatenate([np.ravel(o) for o in next_obs if not isinstance(o, dict)])
            else:
                next_obs = np.array(next_obs).flatten()

            action = np.array(action).flatten()
            self.agent.model.replay_buffer.add(obs, next_obs, action, reward, done, infos=[info])
            obs = next_obs

            if done:
                obs = self.reset()

        if hasattr(self.reward_system, "dpg_manager") and self.reward_system.dpg_manager:
            self.reward_system.dpg_manager.config.enabled = dpg_enabled

        self.episode_timeout_s = self.episode_training_timeout_s
        self.max_steps = self.max_training_steps

        self.logger.info("Pré-preenchimento do buffer de replay concluído.")

    def soft_env_reset(self):
        # Remover corpos antigos se existirem
        if hasattr(self, "robot") and self.robot.id is not None:
            p.removeBody(self.robot.id)

        if hasattr(self.environment, "id") and self.environment.id is not None:
            p.removeBody(self.environment.id)

        # Recarregar ambiente
        self.environment.load_in_simulation(use_fixed_base=True)

        # Recarregar robô após o ambiente
        self.robot.load_in_simulation()

        if self.is_visualization_enabled:
            self.update_com_marker()

    def reset_episode_vars(self):
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.episode_robot_x_initial_position = 0.0
        self.episode_robot_y_initial_position = 0.0
        self.episode_distance = 0.0
        self.joint_velocities = self.action_dim * [0.0]
        self.episode_success = False
        self.episode_terminated = False
        self.episode_truncated = False
        self.episode_episode_done = False
        self.episode_termination = "none"
        self.episode_last_action = np.zeros(self.action_dim, dtype=float)
        self.episode_steps = 0
        self.episode_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        Reinicia o ambiente e retorna o estado inicial.
        """
        # Reiniciar variáveis do episódio
        self.reset_episode_vars()

        # Resetar ambiente de simulação
        if self.config_changed_value.value:
            self.config_changed_value.value = 0
            self.is_visualization_enabled = self.enable_visualization_value.value
            self.is_real_time_enabled = self.enable_real_time_value.value
            self.setup_sim_env()

        else:
            self.soft_env_reset()

        # Obter posição inicial
        robot_position, robot_velocity, robot_orientation, robot_orientation_velocity = self.robot.get_imu_position_velocity_orientation()
        self.episode_robot_x_initial_position = robot_position[0]
        self.episode_robot_y_initial_position = robot_position[1]

        # Configurar parâmetros físicos para estabilidade
        self._configure_robot_stability()

        # Retornar observação inicial
        obs = self.robot.get_observation()
        return obs, {}

    def _configure_robot_stability(self):
        """Configura parâmetros para melhorar a estabilidade inicial do robô"""

        for link_index in range(-1, self.robot.get_num_joints()):
            p.changeDynamics(
                self.robot.id, link_index, lateralFriction=self.lateral_friction, spinningFriction=self.spinning_friction, rollingFriction=self.rolling_friction, restitution=self.restitution
            )

        for link_index in range(-1, self.environment.get_num_joints()):
            p.changeDynamics(
                self.environment.id, link_index, lateralFriction=self.lateral_friction, spinningFriction=self.spinning_friction, rollingFriction=self.rolling_friction, restitution=self.restitution
            )

        # Reduzir damping para menos oscilação
        p.changeDynamics(self.robot.id, -1, linearDamping=0.04, angularDamping=0.04)

    def transmit_episode_info(self):
        """Transmite informações do episódio via IPC"""

        if hasattr(self.agent, "model") and hasattr(self.agent.model, "ep_info_buffer"):
            if self.agent.model.ep_info_buffer is not None and len(self.agent.model.ep_info_buffer) > 0 and len(self.agent.model.ep_info_buffer[0]) > 0:
                self.agent.model.ep_info_buffer = []

        self.episode_count += 1

        self.ipc_queue.put({"type": "tracker_status", "tracker_status": self.tracker.get_status()})

        # Criar dados do episódio
        episode_data = {
            "type": "episode_data",
            "episodes": self.episode_count,
            "rewards": self.episode_reward,
            "times": self.episode_steps * self.time_step_s,
            "steps": self.episode_steps,
            "total_steps": self.total_steps,
            "distances": self.episode_distance,
            "success": self.episode_success,
            "imu_x": self.robot_x_position,
            "imu_y": self.robot_y_position,
            "imu_z": self.robot_z_position,
            "imu_x_vel": self.robot_x_velocity,
            "imu_y_vel": self.robot_y_velocity,
            "imu_z_vel": self.robot_z_velocity,
            "roll": self.robot_roll,
            "pitch": self.robot_pitch,
            "yaw": self.robot_yaw,
            "roll_vel": self.robot_roll_vel,
            "pitch_vel": self.robot_pitch_vel,
            "yaw_vel": self.robot_yaw_vel,
        }

        # Adicionar informações de fase DPG se disponível
        if hasattr(self.reward_system, "dpg_manager") and self.reward_system.dpg_manager is not None:
            try:
                dpg_manager = self.reward_system.dpg_manager
                dpg_status = dpg_manager.get_brain_status()
                advanced_metrics = dpg_manager.get_advanced_metrics()

                # Obter informações da fase atual 
                if hasattr(dpg_manager, 'phase_manager') and dpg_manager.phase_manager:
                    phase_info = dpg_manager.phase_manager.get_current_sub_phase_info()
                    episode_data.update({
                        "dpg_phase": phase_info.get('current_phase', 0), 
                        "dpg_phase_index": phase_info.get('phase_index', 0),  
                        "target_speed": phase_info.get('target_speed', 0.0),
                        "dpg_episodes_in_phase": phase_info.get('episodes_in_phase', 0),
                        "phase_name": "sub_fase",  
                        "dpg_success_rate": advanced_metrics.get("success_rate", 0.0),
                        "dpg_avg_distance": advanced_metrics.get("avg_distance", 0.0),
                    })

                episode_data.update({
                    "dass_samples": advanced_metrics.get("dass_samples", 0),
                    "irl_confidence": advanced_metrics.get("irl_confidence", 0.0),
                    "hdpg_convergence": advanced_metrics.get("hdpg_convergence", 0.0),
                    "hdpg_active": advanced_metrics.get("hdpg_active", False),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter status DPG detalhado: {e}")

            if self.episode_count % 100 == 0:
                try:
                    dpg_system = self.reward_system.dpg_manager
                    if dpg_manager and hasattr(dpg_manager, 'phase_manager') and dpg_manager.phase_manager:
                        phase_manager = dpg_manager.phase_manager
                        current_phase = phase_manager.current_sub_phase  
                        phase_config = phase_manager.current_sub_phase_config  
                        detailed_status = phase_manager.get_status()

                        print(f"\nDPG DIAGNÓSTICO - Ep: {self.episode_count}")
                        print(f"   Sub-fase: {current_phase} ({phase_config.name})")  
                        print(f"   Episódios na sub-fase: {detailed_status.get('episodes_in_sub_phase', 0)}")  
                        print(f"   Taxa de sucesso: {detailed_status.get('success_rate', 0):.1%}")
                        print(f"   Grupo atual: {detailed_status.get('current_group', 0)}")  

                        # Condições de transição
                        print(f"   REQUISITOS SUB-FASE {current_phase}:") 

                        conditions = phase_config.transition_conditions
                        for condition_name, required_value in conditions.items():
                            current_value = self._get_current_condition_value(condition_name, detailed_status, phase_manager)
                            met = self._is_condition_met(condition_name, current_value, required_value)
                            icon = "✅" if met else "❌"
                            if isinstance(current_value, float):
                                formatted_current = f"{current_value:.3f}"
                            else:
                                formatted_current = str(current_value)
                            print(f"     {icon} {condition_name}: {required_value} (Atual: {formatted_current})")

                        # Habilidades focadas
                        print("   HABILIDADES FOCADAS:")
                        focus_skills = phase_config.focus_skills
                        for skill in focus_skills:
                            print(f"     📍 {skill}")

                except Exception as e:
                    print(f"Erro no relatório DPG detalhado: {e}")
                    import traceback
                    traceback.print_exc()

        # Enviar para ipc_queue
        try:
            self.ipc_queue.put_nowait(episode_data)

        except Exception as e:
            self.logger.exception("Erro ao transmitir dados do episódio")
            # Ignorar erros de queue durante avaliação

        if self.episode_count % 10 == 0:
            self.logger.info(f"Episódio {self.episode_count} concluído")

    def _get_current_condition_value(self, condition_name, detailed_status, phase_manager):
        """Obtém o valor atual para uma condição específica"""
        try:
            performance_metrics = phase_manager.get_performance_metrics()
        
            if condition_name == "min_success_rate":
                return performance_metrics["success_rate"]
            elif condition_name == "min_avg_distance":
                return performance_metrics["avg_distance"]
            elif condition_name == "max_avg_roll":
                return performance_metrics["avg_roll"]
            elif condition_name == "min_avg_speed":
                return performance_metrics["avg_speed"]
            elif condition_name == "min_avg_steps":
                return performance_metrics["avg_steps"]
            elif condition_name == "min_alternating_score":
                return performance_metrics["alternating_score"]
            elif condition_name == "min_gait_coordination":
                return performance_metrics["gait_coordination"]
            elif condition_name == "min_positive_movement_rate":
                return performance_metrics["positive_movement_rate"]
            else:
                return "N/A"
        except Exception as e:
            print(f"Erro ao obter condição {condition_name}: {e}")
            return "N/A"

    def _is_condition_met(self, condition_name, current_value, required_value):
        """Verifica se uma condição está sendo atendida"""
        try:
            if condition_name.startswith("min_"):
                return current_value >= required_value
            elif condition_name.startswith("max_"):
                return current_value <= required_value
            else:
                return True  # Para condições não comparativas
        except:
            return False

    def _calculate_positive_movement_rate(self):
        """Calcula taxa de movimento positivo"""
        if not self.performance_history:
            return 0.0
        positive_movements = sum(1 for r in self.performance_history if r.get("distance", 0) > 0.1)
        return positive_movements / len(self.performance_history)

    def _calculate_avg_steps(self, phase_manager):
        """Calcula média de passos"""
        if not phase_manager.performance_history:
            return 0.0
        try:
            recent_steps = [r.get("steps", 0) for r in phase_manager.performance_history[-5:]]
            return np.mean(recent_steps) if recent_steps else 0.0
        except:
            return 0.0

    def _calculate_alternating_score(self, phase_manager):
        """Calcula score de alternância"""
        if len(phase_manager.performance_history) < 8:
            return 0.0
        try:
            alternations = sum(1 for r in phase_manager.performance_history[-8:] 
                              if r.get("alternating", False))
            return alternations / len(phase_manager.performance_history[-8:])
        except:
            return 0.0

    def _calculate_gait_coordination(self, phase_manager):
        """Calcula coordenação de marcha"""
        if not phase_manager.performance_history:
            return 0.3
        try:
            # Implementação simplificada - baseada em contato alternado dos pés
            recent_history = phase_manager.performance_history[-10:]
            alternations = sum(1 for r in recent_history 
                              if r.get("left_contact", False) != r.get("right_contact", False))
            return alternations / len(recent_history) if recent_history else 0.3
        except:
            return 0.3

    def _calculate_propulsion_efficiency(self, phase_manager):
        """Calcula eficiência propulsiva"""
        try:
            # Implementação simplificada - baseada em velocidade vs esforço
            if not phase_manager.performance_history:
                return 0.3
            recent_history = phase_manager.performance_history[-5:]
            avg_speed = np.mean([r.get("speed", 0) for r in recent_history])
            return min(avg_speed / 1.0, 1.0)  # Normalizado para velocidade máxima de 1.0 m/s
        except:
            return 0.3

    def _calculate_consistency_count(self, phase_manager):
        """Calcula contagem de consistência"""
        if len(phase_manager.performance_history) < 5:
            return 0
        try:
            recent_successes = sum(1 for r in phase_manager.performance_history[-5:] 
                                  if r.get("success", False))
            return recent_successes
        except:
            return 0

    def _calculate_avg_pitch(self, phase_manager):
        """Calcula pitch médio"""
        if not phase_manager.performance_history:
            return 0.0
        try:
            return np.mean([abs(r.get("pitch", 0)) for r in phase_manager.performance_history[-10:]])
        except:
            return 0.0

    def apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)  # Normalizar ação para evitar valores extremos

        joint_positions, joint_velocities = self.robot.get_joint_states()

        max_step_size = self.max_motor_velocity * self.time_step_s
        target_positions = [current_angle + action_value * max_step_size for current_angle, action_value in zip(joint_positions, action)]

        forces = [self.max_motor_torque] * self.action_dim

        p.setJointMotorControlArray(
            bodyIndex=self.robot.id,
            jointIndices=self.robot.revolute_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=forces,
            positionGains=[0.5] * self.action_dim,
        )

    def step(self, action):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        if self.should_save_model:
            self.ipc_queue.put({"type": "tracker_status", "tracker_status": self.tracker.get_status()})
            save_path = os.path.join(utils.TEMP_MODEL_SAVE_PATH, f"autosave_{self.tracker.auto_save_count}")
            utils.ensure_directory(save_path)
            model_full_path = os.path.join(save_path, f"autosave_model_{self.tracker.auto_save_count}.zip")
            self.save_and_confirm(save_path, model_full_path, autosave=True)
            self.should_save_model = False

        self.apply_action(action)

        # Avançar simulação
        for _ in range(self.physics_step_multiplier):
            p.stepSimulation()

        if self.is_visualization_enabled:
            self.update_com_marker()

        self.episode_steps += 1
        self.total_steps += 1

        # Obter observação
        self.has_gait_state_changed = self.robot.update_gait_state()
        obs = self.robot.get_observation()

        robot_position, robot_velocity, robot_orientation, robot_orientation_velocity = self.robot.get_imu_position_velocity_orientation()
        self.robot_x_position = robot_position[0]
        self.robot_y_position = robot_position[1]
        self.robot_z_position = robot_position[2]
        self.robot_x_velocity = robot_velocity[0]
        self.robot_y_velocity = robot_velocity[1]
        self.robot_z_velocity = robot_velocity[2]
        self.robot_roll = robot_orientation[0]
        self.robot_pitch = robot_orientation[1]
        self.robot_yaw = robot_orientation[2]
        self.robot_roll_vel = robot_orientation_velocity[0]
        self.robot_pitch_vel = robot_orientation_velocity[1]
        self.robot_yaw_vel = robot_orientation_velocity[2]
        self.robot_right_knee_angle, self.robot_left_knee_angle = self.robot.get_knee_angles()
        self.robot_right_hip_frontal_angle, self.robot_left_hip_frontal_angle = self.robot.get_hip_frontal_angles()
        self.robot_right_hip_lateral_angle, self.robot_left_hip_lateral_angle = self.robot.get_hip_lateral_angles()
        self.robot_left_foot_contact, self.robot_right_foot_contact = self.robot.get_foot_contact_states()
        self.robot_left_foot_height, self.robot_right_foot_height = self.robot.get_foot_heights()
        self.robot_right_shoulder_front_angle, self.robot_left_shoulder_front_angle = self.robot.get_shoulder_angles()
        self.robot_left_foot_x_velocity, self.robot_right_foot_x_velocity = self.robot.get_foot_x_velocities()
        self.robot_right_foot_orientation, self.robot_left_foot_orientation = self.robot.get_foot_global_angles()
        self.robot_right_foot_roll = self.robot_right_foot_orientation[0]
        self.robot_left_foot_roll = self.robot_left_foot_orientation[0]
        self.robot_right_foot_pitch = self.robot_right_foot_orientation[1]
        self.robot_left_foot_pitch = self.robot_left_foot_orientation[1]

        self.last_joint_velocities = self.joint_velocities
        self.joint_positions, self.joint_velocities = self.robot.get_joint_states()

        self.episode_distance = self.robot_x_position - self.episode_robot_x_initial_position

        info = {}

        # Condições de Termino
        self.episode_termination = "none"

        # Queda
        if self.robot_z_position < self.fall_threshold:
            self.episode_terminated = True
            self.episode_termination = "fell"

        # Desvio do caminho
        if abs(self.robot_yaw) >= self.yaw_threshold:
            self.episode_terminated = True
            self.episode_termination = "yaw_deviated"

        # Sucesso
        elif self.episode_distance >= self.success_distance:
            self.episode_terminated = True
            self.episode_success = True
            self.episode_termination = "success"

        # Timeout
        elif self.episode_steps * self.time_step_s >= self.episode_timeout_s:
            self.episode_truncated = True
            self.episode_termination = "timeout"

        self.episode_done = self.episode_truncated or self.episode_terminated

        # Calcular recompensa
        reward = self.reward_system.calculate_reward(self, action)
        self.episode_reward += reward

        if self.config_changed_value.value:  # Se houve mudança de configuração
            if self.pause_value.value:
                self.ipc_queue.put({"type": "tracker_status", "tracker_status": self.tracker.get_status()})

                while self.pause_value.value and not self.exit_value.value:
                    self.try_to_resolve_config_change()
                    time.sleep(0.1)

            self.try_to_resolve_config_change()

        else:  # Desabilita espera real-time quando há mudança de configuração pendente
            if self.is_visualization_enabled and self.is_real_time_enabled:
                time.sleep(self.time_step_s)

        self.should_save_model = self.tracker.update()

        # Coletar info final quando o episódio terminar
        if self.episode_done:
            if hasattr(self.reward_system, "dpg_manager") and self.reward_system.dpg_manager:

                episode_results = {
                    "distance": self.episode_distance,
                    "success": self.episode_success,
                    "duration": self.episode_steps * self.time_step_s,
                    "reward": self.episode_reward,
                    "roll": abs(self.robot_roll),
                    "steps": self.episode_steps,
                    "left_contact": self.robot_left_foot_contact,
                    "right_contact": self.robot_right_foot_contact,
                    "gait_pattern_score": self.robot.get_gait_pattern_score(),
                    "speed": abs(self.robot_x_velocity),
                    "energy_used": self.robot.get_energy_used(),
                    "flight_quality": self.robot.get_flight_phase_quality(),
                    "clearance_score": self.robot.get_clearance_score(),
                    "propulsion_efficiency": self.robot.get_propulsion_efficiency(),
                    "alternating": self.robot_left_foot_contact != self.robot_right_foot_contact,
                }
                try:
                    self.reward_system.dpg_manager.update_phase_progression(episode_results)
                except Exception as e:
                    self.logger.error(f"Erro ao chamar update_phase: {e}")

            self.transmit_episode_info()

        self.episode_last_action = action

        if self.tracker.should_pause():
            self.ipc_queue.put({"type": "autopause_request", "tracker_status": self.tracker.get_status()})
            self.tracker.patience_steps += self.tracker.original_patience

        if self.exit_value.value:
            self.logger.info("Sinal de saída recebido em step. Finalizando simulação.")
            info["exit"] = True

        return obs, reward, self.episode_terminated, self.episode_truncated, info

    def try_to_resolve_config_change(self):
        self.config_changed_value.value = 0
        self.is_real_time_enabled = self.enable_real_time_value.value

        try:
            msg = self.ipc_queue_main_to_process.get_nowait()
            data_type = msg.get("type")

            if data_type == "save_request":
                save_path = msg.get("save_session_path")
                model_full_path = os.path.join(save_path, "latest_model.zip")
                self.save_and_confirm(save_path, model_full_path, autosave=False)

        except queue.Empty:
            pass

        if self.last_selected_camera != self.camera_selection_value.value or self.is_visualization_enabled != self.enable_visualization_value.value:
            self.config_changed_value.value = 1  # Para esta mudança de configuração, precisamos aguardar o término do episódio atual para reiniciar a simulação

    def save_and_confirm(self, save_path, model_full_path, autosave):
        """Salva modelo do agente, solicita salvamento de dados da gui, pausa treinamento enquanto gui salva os dados"""
        self.pause_value.value = True
        self.config_changed_value.value = True
        self.agent.save_model(model_full_path)
        self.ipc_queue.put({"type": "agent_model_saved", "save_path": save_path, "autosave": autosave, "tracker_status": self.tracker.get_status()})

    def close(self):
        p.disconnect()
