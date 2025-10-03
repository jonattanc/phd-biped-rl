# simulation.py

import pybullet as p
import gymnasium as gym
import time
import numpy as np
import random
from stable_baselines3.common.vec_env import DummyVecEnv


class Simulation(gym.Env):
    def __init__(self, logger, robot, environment, ipc_queue, pause_value, exit_value, enable_visualization_value, num_episodes=1, seed=42):
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
        self.num_episodes = num_episodes
        self.current_episode = 0

        self.logger = logger
        self.agent = None
        self.physics_client = None

        # Configurações de simulação
        self.fall_threshold = 0.5  # m
        self.yaw_threshold = 0.5  # rad
        self.episode_timeout_s = 20  # s
        self.physics_step_s = 1 / 240.0  # 240 Hz, ~4.16 ms
        self.physics_step_multiplier = 5
        self.time_step_s = self.physics_step_s * self.physics_step_multiplier  # 240/5 = 48 Hz, ~20.83 ms
        self.success_distance = 10.0  # m
        self.max_motor_velocity = 2.0  # rad/s
        self.max_motor_torque = 130.0  # Nm
        self.apply_action = self.apply_position_action
        self.max_steps = int(self.episode_timeout_s / self.time_step_s)

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
        self.logger.info(f"Action space: {self.action_dim}, Observation space: {self.observation_dim}")

        # Variáveis para coleta de dados
        self.reset_episode_vars()
        self.episode_count = 0

    def setup_sim_env(self):
        """Conecta ao PyBullet e carrega ambiente e robô"""
        if self.physics_client is not None:
            p.disconnect()

        # Usar visualização apenas se estiver habilitada
        if self.is_visualization_enabled:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.resetDebugVisualizerCamera(
            cameraDistance=6.5,
            cameraYaw=35,
            cameraPitch=-45,
            cameraTargetPosition=[6.0, 0.0, 0.6])

        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.physics_step_s)

        # Carregar ambiente primeiro
        self.environment.load_in_simulation(use_fixed_base=True)

        # Carregar robô diretamente na posição correta usando o método original
        start_position = [0.2, 0, 0.06]  # x=0.2m (centro da primeira seção), y=0, z=0.06m
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # Carregar URDF diretamente com posição/orientação
        self.robot.id = p.loadURDF(self.robot.urdf_path, start_position, start_orientation)

        # Configurar o robô após carregamento
        self.robot.imu_link_index = self.robot.get_link_index("imu_link")
        num_joints = p.getNumJoints(self.robot.id)
        self.robot.revolute_indices = [i for i in range(num_joints) if p.getJointInfo(self.robot.id, i)[2] == p.JOINT_REVOLUTE]
        self.robot.initial_position, self.robot.initial_orientation = p.getBasePositionAndOrientation(self.robot.id)

        self.robot.initial_joint_states = []
        for j in range(p.getNumJoints(self.robot.id)):
            joint_info = p.getJointState(self.robot.id, j)
            self.robot.initial_joint_states.append(joint_info[0])

    def set_agent(self, agent):
        self.agent = agent

    def set_initial_episode(self, initial_episode):
        """Configura o episódio inicial para continuar de onde parou"""
        self.current_episode = initial_episode
        self.episode_count = initial_episode
        self.logger.info(f"Episódio inicial configurado: {self.current_episode}")

    def run(self):
        """Executa múltiplos episódios e retorna métricas"""
        self.logger.info("Executando simulation.run")
        all_metrics = []

        for episode in range(self.num_episodes):
            if self.exit_value.value:
                self.logger.info("Sinal de saída recebido em run. Finalizando simulação.")
                break

            while self.pause_value.value and not self.exit_value.value:
                time.sleep(0.1)

            episode_number = self.current_episode + episode + 1
            self.logger.info(f"=== INICIANDO EPISÓDIO {episode_number}/{self.current_episode + self.num_episodes} ===")

            episode_metrics = self.run_episode()
            all_metrics.append(episode_metrics)

            self.logger.info(f"=== EPISÓDIO {episode_number} FINALIZADO ===")
            self.logger.info(f"Recompensa: {episode_metrics['reward']:.2f}")
            self.logger.info(f"Distância: {episode_metrics['distance']:.2f}m")
            self.logger.info(f"Sucesso: {episode_metrics['success']}")
            self.logger.info(f"Passos: {episode_metrics['steps']}")
            self.logger.info("")

        return all_metrics

    def run_episode(self):
        """Executa um episódio completo e retorna métricas"""
        start_time = time.time()
        distance_traveled = 0.0
        prev_x_pos = 0.0
        steps = 0
        success = False
        reward = 0.0

        # --- PASSO 1: REMOVER O ROBÔ E O AMBIENTE ANTIGOS ---
        if self.robot.id is not None:
            p.removeBody(self.robot.id)
        if hasattr(self.environment, "id") and self.environment.id is not None:
            p.removeBody(self.environment.id)

        # --- PASSO 2: RECRIAR O AMBIENTE ---
        self.plane.id = self.environment.load_in_simulation(use_fixed_base=True)

        # --- PASSO 3: RECRIAR O ROBÔ ---
        self.robot.id = self.robot.load_in_simulation()

        # --- PASSO 4 RESETAR A POSIÇÃO INICIAL DO ROBÔ ---
        # Forçar a referência de distância para 0.0
        episode_robot_x_initial_position = 0.0

        while steps < self.max_steps:
            action = np.random.uniform(-1, 1, size=self.action_dim)
            # Obter observação
            pos, _ = p.getBasePositionAndOrientation(self.robot.id)
            current_x_pos = pos[0]
            distance_traveled = current_x_pos - episode_robot_x_initial_position

            # Calcular recompensa
            progress = distance_traveled - prev_x_pos
            if progress > 0:
                progress_reward = progress * 10  # Recompensa menor para progresso positivo
            else:
                progress_reward = progress * 20  # Penalidade maior para movimento para trás
            reward += progress_reward
            prev_x_pos = distance_traveled

            # Verificar queda
            if pos[2] < self.fall_threshold:
                reward -= 100
                self.logger.info(f"Robô caiu após {steps} passos. Distância: {distance_traveled:.2f}m")
                break

            # Verificar sucesso
            if distance_traveled >= self.success_distance:
                reward += 50
                success = True
                self.logger.info(f"Sucesso! Percurso concluído em {steps} passos ({distance_traveled:.2f}m)")
                break

            # Aplicar ação
            p.setJointMotorControlArray(bodyUniqueId=self.robot.id, jointIndices=self.robot.revolute_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=action, forces=[100] * len(action))

            # Avançar simulação
            p.stepSimulation()
            steps += 1

            if self.is_visualization_enabled:
                time.sleep(self.time_step_s)

            if steps % 100 == 0:
                self.logger.debug(f"Passo {steps} | Distância: {distance_traveled:.2f}m")

        total_time = time.time() - start_time
        self.logger.info(f"Episódio finalizado. Distância: {distance_traveled:.2f}m | Tempo: {total_time:.2f}s | Sucesso: {success}")

        return {"reward": reward, "time_total": total_time, "distance": distance_traveled, "success": success, "steps": steps}

    def on_episode_end(self):
        self.episode_count += 1

        # Obter posição e orientação final da IMU
        imu_position, imu_orientation = self.robot.get_imu_position_and_orientation()

        # USAR O EPISÓDIO CORRETO (current_episode + episode_count)
        actual_episode_number = self.current_episode + self.episode_count

        self.ipc_queue.put(
            {
                "type": "episode_data",
                "episode": actual_episode_number,
                "reward": self.episode_reward,
                "time": self.episode_steps * self.time_step_s,
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

        if actual_episode_number % 10 == 0:
            self.logger.info(f"Episódio {actual_episode_number} concluído")

    def soft_env_reset(self):
        # Remover corpos antigos se existirem
        if hasattr(self, "robot") and self.robot.id is not None:
            p.removeBody(self.robot.id)

        if hasattr(self.environment, "id") and self.environment.id is not None:
            p.removeBody(self.environment.id)

        # Recarregar ambiente
        self.environment.load_in_simulation(use_fixed_base=True)

        # Recarregar robô diretamente na posição correta

        start_position = [0.2, 0, 0.06]  # x=0.2m, y=0, z=0.06m
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot.id = p.loadURDF(self.robot.urdf_path, start_position, start_orientation)

        # Reconfigurar o robô após carregamento
        self.robot.imu_link_index = self.robot.get_link_index("imu_link")
        num_joints = p.getNumJoints(self.robot.id)
        self.robot.revolute_indices = [i for i in range(num_joints) if p.getJointInfo(self.robot.id, i)[2] == p.JOINT_REVOLUTE]
        self.robot.initial_position, self.robot.initial_orientation = p.getBasePositionAndOrientation(self.robot.id)

        self.robot.initial_joint_states = []
        for j in range(p.getNumJoints(self.robot.id)):
            joint_info = p.getJointState(self.robot.id, j)
            self.robot.initial_joint_states.append(joint_info[0])

    def reset_episode_vars(self):
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.episode_robot_x_initial_position = 0.0
        self.episode_distance = 0.0
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
        if self.is_visualization_enabled != self.enable_visualization_value.value:
            self.is_visualization_enabled = self.enable_visualization_value.value
            self.setup_sim_env()
        else:
            self.soft_env_reset()

        # IMPORTANTE: Reconfigurar o agente no ambiente após o reset
        if self.agent is not None:
            # Garantir que o agente ainda está configurado
            self.agent.env = self
            if hasattr(self.agent, 'model') and self.agent.model is not None:
                # Reconfigurar o ambiente no modelo se necessário
                try:
                    if self.agent.model.get_env() is None:
                        # USAR DummyVecEnv CORRETAMENTE
                        vec_env = DummyVecEnv([lambda: self])
                        self.agent.model.set_env(vec_env)
                except Exception as e:
                    self.logger.warning(f"Não foi possível reconfigurar ambiente no reset: {e}")

        # Obter posição inicial
        robot_position, robot_orientation = self.robot.get_imu_position_and_orientation()
        self.episode_robot_x_initial_position = robot_position[0]

        # Configurar parâmetros físicos para estabilidade
        self._configure_robot_stability()

        # Retornar observação inicial
        obs = self.robot.get_observation()
        return obs, {}

    def _configure_robot_stability(self):
        """Configura parâmetros para melhorar a estabilidade inicial do robô"""
        # Aumentar o atrito dos pés
        for link_index in range(-1, self.robot.get_num_joints()):
            p.changeDynamics(self.robot.id, link_index, lateralFriction=1.0)

        # Reduzir damping para menos oscilação
        p.changeDynamics(self.robot.id, -1, linearDamping=0.04, angularDamping=0.04)

    def transmit_episode_info(self):
        """Transmite informações do episódio apenas se ipc_queue estiver disponível"""
        if len(self.agent.model.ep_info_buffer) > 0 and len(self.agent.model.ep_info_buffer[0]) > 0:
            if self.episode_done:
                self.on_episode_end()

                # Limpar buffer após processamento
                self.agent.model.ep_info_buffer = []

    def on_episode_end(self):
        """Processa o final do episódio apenas se ipc_queue estiver disponível"""
        self.episode_count += 1

        # Obter posição e orientação final da IMU
        imu_position, imu_orientation = self.robot.get_imu_position_and_orientation()

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
                except:
                    pass
            except:
                pass  # Ignorar erros de queue durante avaliação

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
        )

    def get_reward(self, action, robot_state, env_conditions=None):
        """
        Calcula a recompensa com parâmetro opcional para compatibilidade
        """
        reward = 0.0

        # Se env_conditions não for fornecido, criar um padrão
        if env_conditions is None:
            env_conditions = {
                "foot_slip": 0.0,
                "ramp_speed": 0.0,
                "com_drop": 0.0,
                "joint_failure": False
            }
    
        # Definir limites da plataforma
        PLATFORM_WIDTH = 1.0   # Largura total da plataforma
        PLATFORM_CENTER = 0.0  # Centro da plataforma
        SAFE_ZONE = 0.2       # Zona segura central
        WARNING_ZONE = 0.4    # Zona de aviso
    
        # Calcular distância do centro
        robot_position, robot_orientation = self.robot.get_imu_position_and_orientation()
        pos_y = robot_position[1]      # Posição lateral (eixo Y)
        distance_from_center = abs(pos_y - PLATFORM_CENTER)
        normalized_distance = distance_from_center / (PLATFORM_WIDTH / 2)
                
        # ===== FASE 1: RECOMPENSAS CRÍTICAS =====
        # 1. Avanço no percurso
        progress = self.episode_distance - self.episode_last_distance
        reward += progress * 9.0
        reward += self.episode_distance * 2.0  # incentivo de longo prazo

        # 2. Margem de Estabilidade Dinâmica (MOS)
        # Se robot_state for apenas orientação (tupla), converter para dict
        if isinstance(robot_state, (tuple, list)) and len(robot_state) == 3:
            # Assumindo que é (roll, pitch, yaw)
            roll, pitch, yaw = robot_state
            robot_state = {
                "mos": 0.1,  # valor padrão
                "orientation": (roll, pitch, yaw),
                "joint_torques": [0] * self.action_dim,
                "jerk": 0.0,
                "joint_velocities": [0] * self.action_dim,
                "step_time_left": 0.5,
                "step_time_right": 0.5,
                "foot_impact": 0.5,
                "foot_angle": 0.0,
                "foot_clearance": 0.1,
                "ankle_torque": 0.0,
                "com_gain": 1.0,
                "ds_left": 0.3,
                "ds_right": 0.3,
                "support_ratio": 0.6
            }

        mos = robot_state.get("mos", 0.1)
        if mos > 0:
            reward += mos * 3.0
        else:
            reward -= 100  # queda
            self.episode_terminated = True

        # 3. Orientação do torso
        orientation = robot_state.get("orientation", (0, 0, 0))
        roll, pitch, yaw = orientation
        reward += -0.01 * roll*2 - 0.04 * pitch*2
        if abs(yaw) > 20:  
            reward -= 2.0

        # ===== FASE 2: EFICIÊNCIA E ADAPTAÇÃO =====
        # 4. Economia de torque
        torque_sum = sum(abs(t) for t in robot_state.get("joint_torques", [0] * self.action_dim))
        reward += -0.001 * torque_sum

        # 5. Suavidade (jerk e velocidades articulares)
        jerk = robot_state.get("jerk", 0.0)
        reward += -0.05 * jerk
        for v in robot_state.get("joint_velocities", [0] * self.action_dim):
            if 60 <= abs(v) <= 120:
                reward += 0.1
            else:
                reward -= 0.05

        # 6. Ajuste ao atrito (escorregamento do pé)
        slip = env_conditions.get("foot_slip", 0.0)
        if slip < 0.01:
            reward += 1.0
        else:
            reward -= 5.0

        # 7. Eficiência em rampas
        ramp_speed = env_conditions.get("ramp_speed", 0.0)
        if ramp_speed > 0:
            reward += ramp_speed * 2.0
        else:
            reward -= 10.0

        # 8. Penetração controlada em piso granular
        com_drop = env_conditions.get("com_drop", 0.0)
        if com_drop < 0.05:
            reward += 2.0
        else:
            reward -= com_drop * 20.0

        # 9. Compensação articular para falhas
        if env_conditions.get("joint_failure", False):
            if progress > 0 and mos > 0:
                reward += 10.0

        # ===== FASE 3: REFINAMENTO =====
        # 10. Regularidade da marcha
        step_time_left = robot_state.get("step_time_left", 0.5)
        step_time_right = robot_state.get("step_time_right", 0.5)
        step_diff = abs(step_time_left - step_time_right)
        if step_diff / max(step_time_left, 1e-6) < 0.1:
            reward += 2.0
        else:
            reward -= 1.0

        # 11. Contato inicial adequado
        foot_impact = robot_state.get("foot_impact", 0.5)
        foot_angle = robot_state.get("foot_angle", 0.0)
        reward += -0.1 * abs(foot_impact - 0.5)
        if abs(foot_angle) < 10:
            reward += 1.0

        # 12. Clearance do pé
        clearance = robot_state.get("foot_clearance", 0.1)
        if 0.05 <= clearance <= 0.15:
            reward += 1.5
        else:
            reward -= 0.5

        # 13. Propulsão eficiente no pré-balanço
        ankle_torque = robot_state.get("ankle_torque", 0.0)
        com_gain = robot_state.get("com_gain", 1.0)
        reward += 0.1 * (ankle_torque * com_gain)

        # 14. Simetria de duplo apoio
        ds_left = robot_state.get("ds_left", 0.3)
        ds_right = robot_state.get("ds_right", 0.3)
        double_support_diff = abs(ds_left - ds_right)
        if double_support_diff < 0.05:
            reward += 1.0
        else:
            reward -= 1.0

        # 15. Tempo relativo apoio/balanço
        support_ratio = robot_state.get("support_ratio", 0.6)
        if 0.5 <= support_ratio <= 0.7:  # próximo de 60/40
            reward += 1.0

        # 16. Manutenção no centro da plataforma
        if distance_from_center <= SAFE_ZONE:
            # Zona segura: recompensa máxima no centro, decaindo suavemente
            safe_factor = 1.0 - (distance_from_center / SAFE_ZONE)
            center_reward = safe_factor * 5.0 
            reward += center_reward

        elif distance_from_center <= WARNING_ZONE:
            # Zona de aviso: penalidade leve que aumenta com a distância
            warning_factor = (distance_from_center - SAFE_ZONE) / (WARNING_ZONE - SAFE_ZONE)
            warning_penalty = -3.0 * warning_factor  
            reward += warning_penalty

        # ===== SUCESSO OU FALHA =====
        if self.episode_terminated:
            if self.episode_success:
                reward += 100
            else:
                reward -= 250

        return reward

    def step(self, action):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        while self.pause_value.value and not self.exit_value.value:
            time.sleep(0.1)
    
        if self.exit_value.value:
            self.logger.info("Sinal de saída recebido em step. Finalizando simulação.")
            return None, 0.0, True, False, {"exit": True}
    
        # Atualizar configurações de visualização e tempo real se necessário
        if self.is_visualization_enabled != self.enable_visualization_value.value:
            self.is_visualization_enabled = self.enable_visualization_value.value
            self.setup_sim_env()
    
        self.apply_action(action)
    
        # Avançar simulação
        for _ in range(self.physics_step_multiplier):
            p.stepSimulation()
    
        if self.is_visualization_enabled:
            time.sleep(self.time_step_s)
    
        self.episode_steps += 1
    
        # Obter observação
        obs = self.robot.get_observation()
    
        robot_position, robot_orientation = self.robot.get_imu_position_and_orientation()
        robot_x_position = robot_position[0]
        robot_z_position = robot_position[2]
        robot_yaw = robot_orientation[2]
        self.episode_last_distance = self.episode_distance
        self.episode_distance = robot_x_position - self.episode_robot_x_initial_position
    
        # Condições de Termino
        info = {"distance": self.episode_distance, "termination": "none"}
    
        # Queda
        if robot_z_position < self.fall_threshold:
            self.episode_terminated = True
            info["termination"] = "fell"
    
        # Desvio do caminho
        if abs(robot_yaw) >= self.yaw_threshold:
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

        # Obter dados do robô para o cálculo de recompensa
        joint_positions, joint_velocities = self.robot.get_joint_states()
        robot_position, robot_orientation = self.robot.get_imu_position_and_orientation()

        # Criar robot_state básico com orientação
        robot_state = {
            "orientation": robot_orientation,
            "mos": 0.1,  # valor padrão - você pode calcular isso se tiver dados
            "joint_torques": [0] * self.action_dim,  # placeholder
            "jerk": 0.0,  # placeholder
            "joint_velocities": joint_velocities
        }

        # Criar env_conditions básico
        env_conditions = {
            "foot_slip": 0.0,
            "ramp_speed": 0.0, 
            "com_drop": 0.0,
            "joint_failure": False
        }
    
        reward = self.get_reward(action, robot_state, env_conditions)
        self.episode_reward += reward
    
        # Coletar info final quando o episódio terminar
        if self.episode_done:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_steps, "distance": self.episode_distance, "success": self.episode_success}
    
        # MODIFICAÇÃO: Só chamar transmit_episode_info se ipc_queue estiver disponível
        if self.ipc_queue is not None:
            self.transmit_episode_info()
    
        self.episode_last_action = action
    
        return obs, reward, self.episode_terminated, self.episode_truncated, info

    def get_episode_info(self):
        """Retorna informações do episódio atual"""
        return self.episode_info.copy()

    def evaluate(self, num_episodes=5):
        """Método específico para avaliação, ignorando sinais de pause/exit"""
        self.logger.info("Executando simulation.evaluate")
        all_metrics = []

        for episode in range(num_episodes):
            # IGNORAR sinais de pause/exit durante avaliação
            self.logger.info(f"=== INICIANDO EPISÓDIO DE AVALIAÇÃO {episode + 1}/{num_episodes} ===")

            # Executar episódio sem verificar pause/exit
            episode_metrics = self._run_evaluation_episode()
            all_metrics.append(episode_metrics)

            self.logger.info(f"=== EPISÓDIO {episode + 1} FINALIZADO ===")
            self.logger.info(f"Recompensa: {episode_metrics['reward']:.2f}")
            self.logger.info(f"Distância: {episode_metrics['distance']:.2f}m")
            self.logger.info(f"Sucesso: {episode_metrics['success']}")

        return self._compile_evaluation_metrics(all_metrics)

    def _run_evaluation_episode(self):
        """Executa um episódio de avaliação sem verificar sinais externos"""
        # Reset manual do ambiente
        if self.robot.id is not None:
            p.removeBody(self.robot.id)
        if hasattr(self.environment, "id") and self.environment.id is not None:
            p.removeBody(self.environment.id)

        self.environment.load_in_simulation(use_fixed_base=True)

        # Carregar robô na posição correta
        start_position = [0.5, 0, 0.7]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot.id = p.loadURDF(self.robot.urdf_path, start_position, start_orientation)

        # Configuração inicial
        pos, _ = p.getBasePositionAndOrientation(self.robot.id)
        episode_robot_x_initial_position = pos[0]
        distance_traveled = 0.0
        steps = 0
        success = False
        reward = 0.0

        while steps < self.max_steps:
            # Ação aleatória para teste
            action = np.random.uniform(-1, 1, size=self.action_dim)

            # Aplicar ação
            p.setJointMotorControlArray(bodyUniqueId=self.robot.id, jointIndices=self.robot.revolute_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=action, forces=[100] * len(action))

            # Avançar simulação
            p.stepSimulation()
            steps += 1

            # Calcular progresso
            pos, _ = p.getBasePositionAndOrientation(self.robot.id)
            current_x_pos = pos[0]
            distance_traveled = current_x_pos - episode_robot_x_initial_position

            # Verificar condições de término
            if pos[2] < self.fall_threshold:
                reward -= 100
                break

            if distance_traveled >= self.success_distance:
                reward += 50
                success = True
                break

        return {"reward": reward, "time_total": steps * self.time_step_s, "distance": distance_traveled, "success": success, "steps": steps}

    def _compile_evaluation_metrics(self, all_metrics):
        """Compila métricas de todos os episódios"""
        self.logger.info("Executando simulation._compile_evaluation_metrics")
        total_times = [m["time_total"] for m in all_metrics]
        successes = [m["success"] for m in all_metrics]

        return {
            "avg_time": np.mean(total_times) if total_times else 0,
            "std_time": np.std(total_times) if len(total_times) > 1 else 0,
            "success_rate": np.mean(successes) if successes else 0,
            "success_count": sum(successes),
            "total_times": total_times,
            "total_rewards": [m["reward"] for m in all_metrics],
            "num_episodes": len(all_metrics),
        }

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()