# simulation.py
import pybullet as p
import gymnasium as gym
import time
import numpy as np
import random
import math
from reward_system import RewardSystem


class Simulation(gym.Env):
    def __init__(
        self, logger, robot, environment, ipc_queue, pause_value, exit_value, enable_visualization_value, enable_real_time_value, camera_selecion_value, num_episodes=1, seed=42, initial_episode=0
    ):
        super(Simulation, self).__init__()
        np.random.seed(seed)
        random.seed(seed)

        self.robot = robot
        self.environment = environment
        self.ipc_queue = ipc_queue
        self.pause_value = pause_value
        self.exit_value = exit_value
        self.enable_visualization_value = enable_visualization_value
        self.is_visualization_enabled = enable_visualization_value.value
        self.enable_real_time_value = enable_real_time_value
        self.camera_selection_value = camera_selecion_value
        self.last_camera_selection = camera_selecion_value.value
        self.num_episodes = num_episodes
        self.current_episode = 0
        self.total_steps = 0

        self.logger = logger
        self.agent = None
        self.physics_client = None
        self.com_marker = None
        self.reward_system = RewardSystem(logger)

        # Configurações de simulação
        self.target_pitch_rad = math.radians(1)  # rad
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
        self.apply_action = self.apply_position_action  # Escolher entre apply_velocity_action ou apply_position_action
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
        self.current_episode = initial_episode
        self.episode_count = initial_episode

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
        else:
            self.physics_client = p.connect(p.DIRECT)

        if self.camera_selection_value.value == 1:
            # Visão geral do ambiente
            p.resetDebugVisualizerCamera(cameraDistance=6.5, cameraYaw=35, cameraPitch=-45, cameraTargetPosition=[6.0, 0.0, 0.6])

        elif self.camera_selection_value.value == 2:
            # Visão próxima do robô
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=15, cameraPitch=-25, cameraTargetPosition=[0.0, 0.0, 0.0])

        elif self.camera_selection_value.value == 3:
            # Visão lateral do robô
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=180, cameraPitch=-15, cameraTargetPosition=[0.0, 0.0, 0.0])

        elif self.camera_selection_value.value == 4:
            # Visão frontal do robô
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=[0.0, 0.0, 0.5])

        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.physics_step_s)

        # Carregar ambiente primeiro
        self.environment.load_in_simulation(use_fixed_base=True)

        # Carregar robô após o ambiente
        self.robot.load_in_simulation()

        self.create_com_marker()

        if self.is_visualization_enabled:
            time.sleep(0.5)  # Aguarda a inicialização da janela do PyBullet
            self.ipc_queue.put({"type": "pybullet_window_ready"})

    def set_agent(self, agent):
        self.agent = agent

    def pre_fill_buffer(self, timesteps=100e3):
        self.logger.info(f"Pré-preenchendo buffer de replay com {timesteps} timesteps...")
        obs = self.reset()

        self.episode_timeout_s = self.episode_pre_fill_timeout_s
        self.max_steps = self.max_pre_fill_steps

        while self.total_steps < timesteps and not self.exit_value.value:
            t = self.episode_steps * self.time_step_s
            action = self.robot.get_example_action(t)

            next_obs, reward, episode_terminated, episode_truncated, info = self.step(action)
            done = episode_terminated or episode_truncated

            if isinstance(obs, (list, tuple)):
                obs = np.concatenate([np.ravel(o) for o in obs if not isinstance(o, dict)])

            if isinstance(next_obs, (list, tuple)):
                next_obs = np.concatenate([np.ravel(o) for o in next_obs if not isinstance(o, dict)])

            self.agent.model.replay_buffer.add(obs, next_obs, action, reward, done, infos=[info])
            obs = next_obs

            if done:
                obs = self.reset()

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
        self.episode_distance = 0.0
        self.joint_velocities = self.action_dim * [0.0]
        self.episode_success = False
        self.episode_terminated = False
        self.episode_truncated = False
        self.episode_episode_done = False
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
        if self.is_visualization_enabled != self.enable_visualization_value.value or self.last_camera_selection != self.camera_selection_value.value:
            self.is_visualization_enabled = self.enable_visualization_value.value
            self.last_camera_selection = self.camera_selection_value.value
            self.setup_sim_env()
        else:
            self.soft_env_reset()

        # Obter posição inicial
        robot_position, robot_velocity, robot_orientation = self.robot.get_imu_position_velocity_orientation()
        self.episode_robot_x_initial_position = robot_position[0]

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
        if self.episode_done:
            self.on_episode_end()

            if self.agent.model.ep_info_buffer is not None and len(self.agent.model.ep_info_buffer) > 0 and len(self.agent.model.ep_info_buffer[0]) > 0:
                self.agent.model.ep_info_buffer = []

    def on_episode_end(self):
        """Processa o final do episódio apenas se ipc_queue estiver disponível"""
        self.episode_count += 1

        # Obter posição e orientação final da IMU
        imu_position, robot_velocity, imu_orientation = self.robot.get_imu_position_velocity_orientation()

        actual_episode_number = self.current_episode + self.episode_count

        # SÓ enviar para ipc_queue se estiver disponível
        if self.ipc_queue is not None:
            try:
                self.ipc_queue.put(
                    {
                        "type": "episode_data",
                        "episode": actual_episode_number,
                        "reward": self.episode_reward,
                        "time": self.episode_steps * self.time_step_s,
                        "steps": self.episode_steps,
                        "distance": self.episode_distance,
                        "success": self.episode_success,
                        "imu_x": imu_position[0],
                        "imu_y": imu_position[1],
                        "imu_z": imu_position[2],
                        "roll": imu_orientation[0],
                        "pitch": imu_orientation[1],
                        "yaw": imu_orientation[2],
                    }
                )

                # Enviar contagem de steps para a GUI
                try:
                    self.ipc_queue.put_nowait({"type": "step_count", "steps": self.episode_steps})
                except Exception as e:
                    pass
            except Exception as e:
                self.logger.exception("Erro ao transmitir dados do episódio")
                # Ignorar erros de queue durante avaliação

        if actual_episode_number % 10 == 0:
            self.logger.info(f"Episódio {actual_episode_number} concluído")

    def apply_velocity_action(self, action):
        action = np.clip(action, -1.0, 1.0)  # Normalizar ação para evitar valores extremos

        target_velocities = action * self.max_motor_velocity
        forces = [self.max_motor_torque] * self.action_dim

        p.setJointMotorControlArray(bodyUniqueId=self.robot.id, jointIndices=self.robot.revolute_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=target_velocities, forces=forces)

    def apply_position_action(self, action):
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
        while self.pause_value.value and not self.exit_value.value:
            time.sleep(0.1)

        self.apply_action(action)

        # Avançar simulação
        for _ in range(self.physics_step_multiplier):
            p.stepSimulation()

        if self.is_visualization_enabled:
            self.update_com_marker()

        if self.is_visualization_enabled and self.enable_real_time_value.value:
            time.sleep(self.time_step_s)

        self.episode_steps += 1
        self.total_steps += 1

        # Obter observação
        obs = self.robot.get_observation()

        robot_position, robot_velocity, robot_orientation = self.robot.get_imu_position_velocity_orientation()
        self.robot_x_position = robot_position[0]
        self.robot_y_position = robot_position[0]
        self.robot_z_position = robot_position[2]
        self.robot_x_velocity = robot_velocity[0]
        self.robot_roll = robot_orientation[0]
        self.robot_pitch = robot_orientation[1]
        self.robot_yaw = robot_orientation[2]

        self.last_joint_velocities = self.joint_velocities
        self.joint_positions, self.joint_velocities = self.robot.get_joint_states()

        self.episode_distance = self.robot_x_position - self.episode_robot_x_initial_position

        # Condições de Termino
        info = {"distance": self.episode_distance, "termination": "none"}

        # Queda
        if self.robot_z_position < self.fall_threshold:
            self.episode_terminated = True
            info["termination"] = "fell"

        # Desvio do caminho
        if abs(self.robot_yaw) >= self.yaw_threshold:
            self.episode_terminated = True
            info["termination"] = "yaw_deviated"

        # Sucesso
        elif self.episode_distance >= self.success_distance:
            self.episode_terminated = True
            self.episode_success = True
            info["termination"] = "success"

        # Timeout
        elif self.episode_steps * self.time_step_s >= self.episode_timeout_s:
            self.episode_truncated = True
            info["termination"] = "timeout"

        info["success"] = self.episode_success
        self.episode_done = self.episode_truncated or self.episode_terminated

        # Recompensa
        reward = self.reward_system.calculate_reward(self, action, info)
        self.episode_reward += reward

        # Coletar info final quando o episódio terminar
        if self.episode_done:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_steps, "distance": self.episode_distance, "success": self.episode_success}

        self.transmit_episode_info()

        self.episode_last_action = action

        if self.exit_value.value:
            self.logger.info("Sinal de saída recebido em step. Finalizando simulação.")
            info["exit"] = True

        return obs, reward, self.episode_terminated, self.episode_truncated, info

    def close(self):
        p.disconnect()
