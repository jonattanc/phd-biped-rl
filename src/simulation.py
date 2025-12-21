# simulation.py
import random
import pybullet as p
import gymnasium as gym
import time
import numpy as np
import math
import queue
import os
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
        num_episodes=1,
        initial_episode=0,
        is_fast_td3=True,
    ):
        super(Simulation, self).__init__()

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
        self.is_fast_td3 = is_fast_td3
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
        self.target_x_velocity = 1.0  # m/s
        self.fall_threshold = 0.5  # m
        self.success_distance = 9.0  # m
        self.yaw_threshold = math.radians(60)  # rad
        self.episode_training_timeout_s = 15  # s
        self.episode_pre_fill_timeout_s = 10  # s
        self.episode_timeout_s = self.episode_training_timeout_s
        self.physics_step_s = 1 / 240.0  # 240 Hz, ~4.16 ms
        self.physics_step_multiplier = 8
        self.time_step_s = self.physics_step_s * self.physics_step_multiplier  # 240/5 = 48 Hz, ~20.83 ms # 240/8 = 30 Hz, ~33.33 ms # 240/10 = 24 Hz, ~41.66 ms
        self.max_motor_velocity = 1.5  # rad/s
        self.max_training_steps = int(self.episode_training_timeout_s / self.time_step_s)
        self.max_pre_fill_steps = int(self.episode_pre_fill_timeout_s / self.time_step_s)
        self.max_steps = self.max_training_steps
        self.lock_per_second = 0.5  # lock/s
        self.lock_time = 0.5  # s
        self.action_noise_std = 1e-3
        if self.is_fast_td3:
            self.max_motor_torque = 350.0
        else:
            self.max_motor_torque = 250.0  # Nm

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

        self.lock_probability = (self.lock_per_second / (1 / self.time_step_s)) / self.action_dim
        self.lock_duration_steps = int(self.lock_time / self.time_step_s)

        self.logger.info(f"lock_probability: {self.lock_probability}")
        self.logger.info(f"lock_duration_steps: {self.lock_duration_steps} steps")

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

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

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
        self.wrapped_env = agent.env

    def pre_fill_buffer(self):
        obs, _ = self.reset()

        self.episode_timeout_s = self.episode_pre_fill_timeout_s
        self.max_steps = self.max_pre_fill_steps

        while self.total_steps < self.agent.prefill_steps and not self.exit_value.value:
            t = self.episode_steps * self.time_step_s
            action = self.robot.get_example_action(t)

            next_obs, reward, episode_terminated, episode_truncated, info = self.step(action)
            done = episode_terminated or episode_truncated

            action = np.array(action).flatten()

            # Adicionar ao buffer de replay do modelo
            if hasattr(self.agent.model, "replay_buffer"):
                self.agent.model.replay_buffer.add(obs, next_obs, action, reward, done, infos=[info])

            if done:
                obs, _ = self.reset()
            else:
                obs = next_obs

        self.episode_timeout_s = self.episode_training_timeout_s
        self.max_steps = self.max_training_steps

        self.logger.info("Pré-preenchimento do buffer de replay concluído.")

    def evaluate(self, episodes, deterministic):
        self.metrics = {}
        obs, _ = self.reset()
        self.metrics[str(self.episode_count + 1)] = {"step_data": {}}  # Criar espaço para primeiro episódio

        while self.episode_count < episodes and not self.exit_value.value:
            obs = self.wrapped_env.normalize_obs(obs)
            action, _ = self.agent.model.predict(obs, deterministic=deterministic)

            next_obs, reward, episode_terminated, episode_truncated, info = self.step(action, evaluation=True)
            done = episode_terminated or episode_truncated

            if done:
                obs, _ = self.reset()

            else:
                obs = next_obs

        self.metrics.pop(str(self.episode_count + 1))  # Remover espaço para próximo episódio, pois a avaliação terminou

        return {"episodes": self.metrics}

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
        self.episode_filtered_reward = 0.0
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
        self.joint_lock_timers = [0] * self.action_dim
        self.target_positions = [0] * self.action_dim
        self.robot_x_sum_velocity = 0
        self.robot_y_sum_velocity = 0
        self.robot_z_sum_velocity = 0

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
                self.robot.id,
                link_index,
                lateralFriction=self.environment.environment_settings["default"]["lateral_friction"],
                spinningFriction=self.environment.environment_settings["default"]["spinning_friction"],
                rollingFriction=self.environment.environment_settings["default"]["rolling_friction"],
                restitution=self.environment.environment_settings["default"]["restitution"],
            )

        link_map = self.environment.get_link_indices_by_name()

        for link_name, link_index in link_map.items():
            if link_name in self.environment.environment_settings:
                key = link_name

            else:
                key = "default"

            environment_selected_settings = {
                "lateralFriction": self.environment.environment_settings[key]["lateral_friction"],
                "spinningFriction": self.environment.environment_settings[key]["spinning_friction"],
                "rollingFriction": self.environment.environment_settings[key]["rolling_friction"],
                "restitution": self.environment.environment_settings[key]["restitution"],
            }

            if "contactStiffness" in self.environment.environment_settings[key]:
                environment_selected_settings["contactStiffness"] = self.environment.environment_settings[key]["contactStiffness"]
                environment_selected_settings["contactDamping"] = self.environment.environment_settings[key]["contactDamping"]

            p.changeDynamics(self.environment.id, link_index, **environment_selected_settings)

        # Reduzir damping para menos oscilação
        p.changeDynamics(self.robot.id, -1, linearDamping=0.04, angularDamping=0.04)

    def transmit_episode_info(self, evaluation):
        """Transmite informações do episódio via IPC"""

        if hasattr(self.agent, "model") and hasattr(self.agent.model, "ep_info_buffer"):
            if self.agent.model.ep_info_buffer is not None and len(self.agent.model.ep_info_buffer) > 0 and len(self.agent.model.ep_info_buffer[0]) > 0:
                self.agent.model.ep_info_buffer = []

        self.ipc_queue.put({"type": "tracker_status", "tracker_status": self.tracker.get_status()})

        # Criar dados do episódio
        episode_data = {
            "type": "episode_data",
            "episodes": self.episode_count,
            "rewards": self.episode_reward,
            "rewards_filtered": self.episode_filtered_reward,
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
            "imu_average_x_vel": self.robot_x_sum_velocity / self.episode_steps,
            "imu_average_y_vel": self.robot_y_sum_velocity / self.episode_steps,
            "imu_average_z_vel": self.robot_z_sum_velocity / self.episode_steps,
            "roll": self.robot_roll,
            "pitch": self.robot_pitch,
            "yaw": self.robot_yaw,
            "roll_vel": self.robot_roll_vel,
            "pitch_vel": self.robot_pitch_vel,
            "yaw_vel": self.robot_yaw_vel,
            "episode_environments": self.environment.env_list[self.environment.selected_env_index],
        }

        if evaluation:
            episode_extra_data = {
                "episode_termination": self.episode_termination,
                "episode_truncated": self.episode_truncated,
                "episode_terminated": self.episode_terminated,
                "episode_success": self.episode_success,
            }

            self.metrics[str(self.episode_count)]["episode_data"] = episode_data
            self.metrics[str(self.episode_count)]["episode_extra_data"] = episode_extra_data

        # Enviar para ipc_queue
        try:
            self.ipc_queue.put_nowait(episode_data)

        except Exception as e:
            self.logger.exception("Erro ao transmitir dados do episódio")

    def apply_action(self, action):
        noise = np.random.normal(0, self.action_noise_std, size=action.shape)
        action = np.clip(action + noise, -1, 1)

        joint_positions, joint_velocities = self.robot.get_joint_states()

        max_step_size = self.max_motor_velocity * self.time_step_s
        self.target_positions = [current_angle + action_value * max_step_size for current_angle, action_value in zip(joint_positions, action)]
        if self.environment.name == "PRB":
            for i in range(self.action_dim):
                if self.joint_lock_timers[i] > 0:
                    self.joint_lock_timers[i] -= 1
                if np.random.rand() < self.lock_probability:
                    self.joint_lock_timers[i] = self.lock_duration_steps
                if self.joint_lock_timers[i] > 0:
                    self.target_positions[i] = joint_positions[i]
        forces = [self.max_motor_torque] * self.action_dim

        p.setJointMotorControlArray(
            bodyIndex=self.robot.id,
            jointIndices=self.robot.revolute_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_positions,
            forces=forces,
            positionGains=[0.5] * self.action_dim,
        )

    def step(self, action, evaluation=False):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        # SALVAR O ESTADO ATUAL ANTES DA AÇÃO
        current_obs = self.robot.get_observation().copy()

        if self.should_save_model and not evaluation:
            self.ipc_queue.put({"type": "lock_for_saving"})
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
        next_obs = self.robot.get_observation().copy()

        robot_position, robot_velocity, robot_orientation, robot_orientation_velocity = self.robot.get_imu_position_velocity_orientation()
        self.robot_x_position = robot_position[0]
        self.robot_y_position = robot_position[1]
        self.robot_z_position = robot_position[2]
        self.robot_z_ramp_position = self.robot.get_fixed_height(self.robot_z_position, self.robot_x_position)
        self.robot_x_velocity = robot_velocity[0]
        self.robot_y_velocity = robot_velocity[1]
        self.robot_z_velocity = robot_velocity[2]
        self.robot_x_sum_velocity += self.robot_x_velocity
        self.robot_y_sum_velocity += self.robot_y_velocity
        self.robot_z_sum_velocity += self.robot_z_velocity
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
        self.xcom_ap, self.xcom_ml, self.mos_ap, self.mos_ml = self.robot.get_xcom_and_margin()
        self.mos_min = min(self.mos_ap, self.mos_ml)
        self.robot_right_shoulder_front_angle, self.robot_left_shoulder_front_angle = self.robot.get_shoulder_angles()
        self.robot_left_foot_x_velocity, self.robot_right_foot_x_velocity = self.robot.get_foot_x_velocities()
        self.robot_right_foot_orientation, self.robot_left_foot_orientation = self.robot.get_foot_global_angles()
        self.robot_right_foot_roll = self.robot_right_foot_orientation[0]
        self.robot_left_foot_roll = self.robot_left_foot_orientation[0]
        self.robot_right_foot_pitch = self.robot_right_foot_orientation[1]
        self.robot_left_foot_pitch = self.robot_left_foot_orientation[1]
        self.is_in_ramp = self.robot.is_in_ramp(self.robot_x_position)

        self.last_joint_velocities = self.joint_velocities
        self.joint_positions, self.joint_velocities = self.robot.get_joint_states()

        self.episode_distance = self.robot_x_position - self.episode_robot_x_initial_position

        info = {}

        # Condições de Termino
        self.episode_termination = "none"

        # Queda
        if self.robot_z_ramp_position < self.fall_threshold:
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
        self.episode_filtered_reward = 0.1 * self.episode_reward + 0.9 * self.episode_filtered_reward

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

        if self.is_fast_td3 and not evaluation:
            # Criar métricas do episódio para o FastTD3
            episode_results = {"reward": reward, "steps": self.episode_steps, "distance": self.episode_distance, "success": self.episode_success, "roll": self.robot_roll, "pitch": self.robot_pitch}

        if self.episode_done and not evaluation:
            episode_duration = self.episode_steps * self.time_step_s
            avg_x_velocity = self.robot_x_sum_velocity / max(self.episode_steps, 1)

            # Log a cada 50 episódios para todos
            if self.episode_count % 50 == 0:
                if self.is_fast_td3:
                    # LOG DETALHADO PARA FastTD3
                    phase_info = self.agent.model.get_phase_info()
                    phase_theme = phase_info.get("phase_theme", "DESCONHECIDA")
                    self.logger.info("=" * 60)
                    self.logger.info(f"FastTD3 - Episódio {self.episode_count} | " f"{phase_theme}")
                    self.logger.info(f"Distância: {self.episode_distance:.2f}m | " f"Duração: {episode_duration:.1f}s | " f"Recompensa: {self.episode_reward:.1f}")
                    self.logger.info(f"RP: {phase_info['current_rps']:.3f} | " f"DP: {phase_info['current_dps']:.3f} | " f"Suc: {phase_info['current_success']:.1%} | " f"Vel: {avg_x_velocity:.2f}m/s")

                else:
                    # LOG BÁSICO PARA PPO E TD3
                    success_status = "✅" if self.episode_success else "❌"
                    self.logger.info(f"{self.agent.algorithm} - Episódio {self.episode_count} | " f"Distância: {self.episode_distance:.2f}m | " f"Duração: {episode_duration:.1f}s")
                    self.logger.info(f"Recompensa: {self.episode_reward:.1f} | " f"Vel.X: {avg_x_velocity:.2f}m/s | " f"Sucesso: {success_status} | " f"Terminação: {self.episode_termination}")

        # ATUALIZAR PHASE MANAGER APENAS PARA FastTD3 NO FINAL DO EPISÓDIO
        if self.episode_done and not evaluation and self.is_fast_td3:
            episode_metrics = {"reward": self.episode_reward, "steps": self.episode_steps, "distance": self.episode_distance, "success": self.episode_success}

            # Atualizar métricas no phase manager do FastTD3
            try:
                transition_occurred = self.agent.model.update_phase_metrics(episode_metrics)

                # Verificar transição de fase (sempre logar transições)
                if transition_occurred:
                    phase_info = self.agent.model.get_phase_info()
        
                    # Aumentar timeout em 5 segundos
                    old_timeout = self.episode_training_timeout_s
                    self.episode_training_timeout_s += 5.0
                    self.episode_timeout_s = self.episode_training_timeout_s

                    # Atualizar max_steps com novo timeout
                    self.max_training_steps = int(self.episode_training_timeout_s / self.time_step_s)

                    self.logger.info(f"Timeout: {old_timeout}s → {self.episode_training_timeout_s}s | " f"Max steps: {self.max_training_steps}")

                    # Limpar metade inicial do buffer COM LOG DETALHADO
                    try:
                        if hasattr(self.agent.model, "clear_half_buffer"):
                            self.agent.model.clear_half_buffer()
                    except Exception as e:
                        self.logger.error(f"❌ FastTD3 - Erro ao limpar buffer: {e}")

                    # Enviar notificação para a GUI
                    self.ipc_queue.put({"type": "phase_transition", "algorithm": "FastTD3", "new_phase": phase_info["phase"], "phase_info": phase_info})
            except Exception as e:
                self.logger.error(f"Erro ao atualizar phase manager: {e}")

        if evaluation:
            self.add_episode_metrics("action", action)
            self.add_episode_metrics("obs", next_obs)
            self.add_episode_metrics("reward", reward)
            self.add_episode_metrics("has_gait_state_changed", self.has_gait_state_changed)
            self.add_episode_metrics("is_in_ramp", self.is_in_ramp)
            self.add_episode_metrics("robot_position", robot_position)
            self.add_episode_metrics("robot_z_ramp_position", self.robot_z_ramp_position)
            self.add_episode_metrics("robot_velocity", robot_velocity)
            self.add_episode_metrics("robot_orientation", robot_orientation)
            self.add_episode_metrics("robot_orientation_velocity", robot_orientation_velocity)
            self.add_episode_metrics("robot_right_knee_angle", self.robot_right_knee_angle)
            self.add_episode_metrics("robot_left_knee_angle", self.robot_left_knee_angle)
            self.add_episode_metrics("robot_right_hip_frontal_angle", self.robot_right_hip_frontal_angle)
            self.add_episode_metrics("robot_left_hip_frontal_angle", self.robot_left_hip_frontal_angle)
            self.add_episode_metrics("robot_right_hip_lateral_angle", self.robot_right_hip_lateral_angle)
            self.add_episode_metrics("robot_left_hip_lateral_angle", self.robot_left_hip_lateral_angle)
            self.add_episode_metrics("robot_left_foot_contact", self.robot_left_foot_contact)
            self.add_episode_metrics("robot_right_foot_contact", self.robot_right_foot_contact)
            self.add_episode_metrics("robot_left_foot_height", self.robot_left_foot_height)
            self.add_episode_metrics("robot_right_foot_height", self.robot_right_foot_height)
            self.add_episode_metrics("robot_left_foot_x_velocity", self.robot_left_foot_x_velocity)
            self.add_episode_metrics("robot_right_foot_x_velocity", self.robot_right_foot_x_velocity)
            self.add_episode_metrics("robot_right_foot_roll", self.robot_right_foot_roll)
            self.add_episode_metrics("robot_left_foot_roll", self.robot_left_foot_roll)
            self.add_episode_metrics("robot_right_foot_pitch", self.robot_right_foot_pitch)
            self.add_episode_metrics("robot_left_foot_pitch", self.robot_left_foot_pitch)
            self.add_episode_metrics("joint_positions", self.joint_positions)
            self.add_episode_metrics("joint_velocities", self.joint_velocities)
            self.add_episode_metrics("episode_distance", self.episode_distance)
            self.add_episode_metrics("joint_lock_timers", self.joint_lock_timers)
            self.add_episode_metrics("target_positions", self.target_positions)

        # Coletar info final quando o episódio terminar
        if self.episode_done:
            self.episode_count += 1

            if evaluation:
                self.metrics[str(self.episode_count + 1)] = {"step_data": {}}

            self.transmit_episode_info(evaluation)

        self.episode_last_action = action

        if self.tracker.should_pause() and not evaluation:
            self.ipc_queue.put({"type": "autopause_request", "tracker_status": self.tracker.get_status()})
            self.tracker.patience_steps += self.tracker.original_patience

        if self.exit_value.value:
            self.logger.info("Sinal de saída recebido em step. Finalizando simulação.")
            info["exit"] = True

        return next_obs, reward, self.episode_terminated, self.episode_truncated, info

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

    def add_episode_metrics(self, key, value):
        episode_key = str(self.episode_count + 1)

        if hasattr(value, "copy"):
            v = value.copy()

        else:
            v = value

        if key in self.metrics[episode_key]["step_data"]:
            self.metrics[episode_key]["step_data"][key].append(v)

        else:
            self.metrics[episode_key]["step_data"][key] = [v]

    def close(self):
        p.disconnect()
