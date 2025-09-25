# simulation.py

import pybullet as p
import gymnasium as gym
import time
import numpy as np
import random


class Simulation(gym.Env):
    def __init__(self, logger, robot, environment, pause_value, exit_value, enable_real_time_value, enable_gui=True, num_episodes=1, seed=42):
        super(Simulation, self).__init__()
        np.random.seed(seed)
        random.seed(seed)

        self.robot = robot
        self.environment = environment
        self.pause_value = pause_value
        self.exit_value = exit_value
        self.enable_real_time_value = enable_real_time_value
        self.enable_gui = enable_gui
        self.num_episodes = num_episodes

        self.logger = logger
        self.physics_client = None

        # Configurações de simulação
        self.physics_client = None
        self.steps = 0
        self.fall_threshold = 0.3
        self.max_steps = 5000  # ~20.8 segundos (240 * 20.8)
        self.time_step_s = 1 / 240.0
        self.success_distance = 10.0
        self.initial_x_pos = 0.0
        self.prev_distance = 0.0

        # Variáveis para coleta de dados
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.episode_distance = 0.0
        self.episode_success = False
        self.episode_info = {}

        self.setup_sim_env()

        # Definir espaço de ação e observação
        self.action_dim = self.robot.get_num_revolute_joints()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.observation_dim = len(self.robot.get_observation())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

        self.logger.info(f"Simulação configurada: {self.robot.name} no {self.environment.name}")
        self.logger.info(f"Action space: {self.action_dim}, Observation space: {self.observation_dim}")

        # Variáveis de estado
        self.steps = 0
        self.max_steps = 5000
        self.success_distance = 10.0
        self.fall_threshold = 0.3
        self.initial_x_pos = 0.0

    def setup_sim_env(self):
        """Conecta ao PyBullet e carrega ambiente e robô"""
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)

        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.time_step_s)

        self.environment.load_in_simulation(use_fixed_base=True)
        self.robot.load_in_simulation()

        self.logger.info("Simulação configurada")
        self.logger.info(f"Robô: {self.robot.name}")
        self.logger.info(f"DOF: {self.robot.get_num_revolute_joints()}")
        self.logger.info(f"Ambiente: {self.environment.name}")

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

            self.logger.info(f"=== INICIANDO EPISÓDIO {episode + 1}/{self.num_episodes} ===")

            episode_metrics = self.run_episode()
            all_metrics.append(episode_metrics)

            self.logger.info(f"=== EPISÓDIO {episode + 1} FINALIZADO ===")
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
        initial_x_pos = 0.0

        # Atualizar os índices das juntas
        self.set_revolute_indices(self.robot.revolute_indices)

        while steps < self.max_steps:
            action = np.random.uniform(-1, 1, size=self.action_dim)
            # Obter observação
            pos, _ = p.getBasePositionAndOrientation(self.robot.id)
            current_x_pos = pos[0]
            distance_traveled = current_x_pos - initial_x_pos

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

            if self.enable_real_time_value.value:
                time.sleep(self.time_step_s)

            if steps % 100 == 0:
                self.logger.debug(f"Passo {steps} | Distância: {distance_traveled:.2f}m")

        total_time = time.time() - start_time
        self.logger.info(f"Episódio finalizado. Distância: {distance_traveled:.2f}m | Tempo: {total_time:.2f}s | Sucesso: {success}")

        return {"reward": reward, "time_total": total_time, "distance": distance_traveled, "success": success, "steps": steps}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        Reinicia o ambiente e retorna o estado inicial.
        """
        # Reiniciar variáveis do episódio
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.episode_distance = 0.0
        self.episode_success = False
        self.steps = 0
        self.prev_distance = 0.0

        # Remover corpos antigos se existirem
        if hasattr(self, "robot") and self.robot.id is not None:
            p.removeBody(self.robot.id)
        if hasattr(self.environment, "id") and self.environment.id is not None:
            p.removeBody(self.environment.id)

        # Recarregar ambiente e robô
        self.environment.load_in_simulation(use_fixed_base=True)
        self.robot.load_in_simulation()

        # Obter posição inicial
        robot_position = self.robot.get_imu_position()
        self.initial_x_pos = robot_position[0]
        self.prev_distance = 0.0

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

    def step(self, action):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        while self.pause_value.value and not self.exit_value.value:
            time.sleep(0.1)

        if self.exit_value.value:
            self.logger.info("Sinal de saída recebido em step. Finalizando simulação.")
            return None, 0.0, True, False, {"exit": True}

        # Normalizar ação para evitar valores extremos
        action = np.clip(action, -1.0, 1.0)

        # Aplicar ação com controle de posição em vez de velocidade para mais suavidade
        current_positions = []
        for i in self.robot.revolute_indices:
            joint_state = p.getJointState(self.robot.id, i)
            current_positions.append(joint_state[0])

        target_positions = current_positions + action * 0.1

        # Aplicar ação
        p.setJointMotorControlArray(bodyUniqueId=self.robot.id, jointIndices=self.robot.revolute_indices, controlMode=p.POSITION_CONTROL, targetPositions=target_positions, forces=[50] * len(action))

        # Avançar simulação
        p.stepSimulation()
        self.steps += 1

        if self.enable_real_time_value.value:
            time.sleep(self.time_step_s)

        # Obter observação
        obs = self.robot.get_observation()
        pos, _ = p.getBasePositionAndOrientation(self.robot.id)
        current_x_pos = pos[0]
        distance_traveled = current_x_pos - self.initial_x_pos

        # --- Estratégia de Recompensa Melhorada ---
        reward = 0.0

        # 1. Recompensa por estar em pé (mais importante no início)
        standing_reward = max(0, (pos[2] - self.fall_threshold)) * 10.0
        reward += standing_reward

        # 2. Recompensa principal por progresso
        progress = distance_traveled - self.prev_distance
        reward += progress * 5.0

        # 3. Recompensa por distância total (incentivo de longo prazo)
        reward += distance_traveled * 2.0

        # 4. Recompensa incremental por movimento para frente
        if hasattr(self, "prev_distance"):
            step_progress = distance_traveled - self.prev_distance
            if step_progress > 0:
                reward += step_progress * 2.0  # Grande recompensa por progresso positivo
            else:
                reward += step_progress * 1.0  # Penalidade por movimento para trás

        # 5. Recompensa por estabilidade (menor penalidade por movimento)
        joint_velocities = []
        for i in self.robot.revolute_indices:
            joint_state = p.getJointState(self.robot.id, i)
            joint_velocities.append(abs(joint_state[1]))
        energy_penalty = -0.001 * sum(joint_velocities)
        reward += energy_penalty

        # Condições de Termino
        done = False
        truncated = False
        info = {"distance": distance_traveled, "success": False, "termination": "none"}

        # Cond1: Queda
        if pos[2] < self.fall_threshold + 0.1:
            reward -= 5  # Penalidade maior por queda
            done = True
            info["termination"] = "fell"
            self.logger.debug(f"Robô caiu! Altura: {pos[2]:.3f}")

        # Cond2: Sucesso
        elif distance_traveled >= self.success_distance:
            reward += 50  # Bônus maior por sucesso
            done = True
            info["success"] = True
            info["termination"] = "success"
            self.logger.debug("Sucesso! Percurso concluído")

        # Cond3: Timeout
        elif self.steps >= self.max_steps:
            truncated = True
            info["termination"] = "timeout"
            reward += distance_traveled * 2.0
            self.logger.debug("Timeout do episódio")

        # Atualizar distância anterior
        self.prev_distance = distance_traveled
        self.episode_reward += reward
        self.episode_distance = distance_traveled

        # Coletar info final quando o episódio terminar
        if done or truncated:
            info["episode"] = {"r": self.episode_reward, "l": self.steps, "distance": distance_traveled, "success": info["success"]}

        return obs, reward, done, truncated, info

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
        self.robot.load_in_simulation()

        # Configuração inicial
        pos, _ = p.getBasePositionAndOrientation(self.robot.id)
        initial_x_pos = pos[0]
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
            distance_traveled = current_x_pos - initial_x_pos

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
        pass  # O modo GUI é controlado por enable_gui no construtor

    def close(self):
        p.disconnect()
