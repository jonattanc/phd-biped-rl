# simulation.py

import pybullet as p
import pybullet_data
import gymnasium as gym
import time
import logging
from datetime import datetime
import numpy as np
import random


class Simulation(gym.Env):
    def __init__(self, robot, environment, pause_value, exit_value, enable_real_time_value, enable_gui=True, num_episodes=1, seed=42):
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

        self.logger = logging.getLogger(__name__)
        self.physics_client = None
        self.plane_id = None
        self.robot_id = None

        self.revolute_indices = []
        self.len_revolute_indices = 0

        # Configurações de simulação
        self.time_step_s = 1 / 240.0
        self.max_steps = 5000  # ~20.8 segundos (240 * 20.8)
        self.success_distance = 10.0
        self.fall_threshold = 0.0  # altura mínima para considerar queda

        self.setup()

        # Definir espaço de ação e observação
        # Ação: velocidade alvo para cada junta revolute (4 juntas)
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)  # robot_stage1 tem 4 juntas revolute

        # Observação: [posição X, velocidade X, orientação (roll, pitch, yaw), 4 ângulos de junta, 4 velocidades de junta]
        # Total: 1 + 1 + 3 + 4 + 4 = 13 valores
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # Variáveis de estado
        self.steps = 0
        self.max_steps = 5000
        self.success_distance = 10.0
        self.fall_threshold = 0.3
        self.initial_x_pos = 0.0

    def setup(self):
        """Conecta ao PyBullet e carrega ambiente e robô"""
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.time_step_s)

        # Carregar ambiente
        self.plane_id = self.environment.load_in_simulation(use_fixed_base=True)
        # Carregar robô
        self.robot_id = self.robot.load_in_simulation()

        # Passar índices das juntas para o agente
        self.set_revolute_indices(self.robot.revolute_indices)
        self.logger.info(f"Simulação configurada: {len(self.robot.revolute_indices)} DOFs")
        self.logger.info(f"Robô: {self.robot.name}")
        self.logger.info(f"Ambiente: {self.environment.name}")

    def set_revolute_indices(self, revolute_indices):
        self.revolute_indices = revolute_indices
        self.len_revolute_indices = len(revolute_indices)

    def run(self):
        """Executa múltiplos episódios e retorna métricas"""
        all_metrics = []

        for episode in range(self.num_episodes):
            if self.exit_value.value:
                self.logger.info("Sinal de saída recebido. Finalizando simulação.")
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
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        if self.plane_id is not None:
            p.removeBody(self.plane_id)

        # --- PASSO 2: RECRIAR O AMBIENTE ---
        self.plane_id = self.environment.load_in_simulation(use_fixed_base=True)

        # --- PASSO 3: RECRIAR O ROBÔ ---
        self.robot_id = self.robot.load_in_simulation()

        # --- PASSO 4 RESETAR A POSIÇÃO INICIAL DO ROBÔ ---
        # Forçar a referência de distância para 0.0
        initial_x_pos = 0.0

        # Atualizar os índices das juntas
        self.set_revolute_indices(self.robot.revolute_indices)

        while steps < self.max_steps:
            # Obter observação
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
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

            # Obter ação do agente
            # action = self.agent.get_action() # TODO: Arrumar

            # Aplicar ação
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.robot.revolute_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=action, forces=[100] * len(action))

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
        """
        Reinicia o ambiente e retorna o estado inicial.
        """
        # Se seed for fornecida, configure o gerador de números aleatórios
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reiniciar simulação (o método run() já faz isso, mas vamos garantir)
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        if self.plane_id is not None:
            p.removeBody(self.plane_id)

        self.plane_id = self.environment.load_in_simulation(use_fixed_base=True)
        self.robot_id = self.robot.load_in_simulation()
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.45], p.getQuaternionFromEuler([0, 0, 0]))

        # Resetar contadores
        self.steps = 0
        self.initial_x_pos = 0.0

        # Retornar observação inicial
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        # Aplicar ação
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.robot.revolute_indices, controlMode=p.VELOCITY_CONTROL, targetVelocities=action, forces=[100] * len(action))

        # Avançar simulação
        p.stepSimulation()
        self.steps += 1

        # Obter observação
        obs = self._get_observation()
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        current_x_pos = pos[0]
        distance_traveled = current_x_pos - self.initial_x_pos

        # --- Estratégia de Recompensa Melhorada ---
        # 1. Recompensa principal por progresso (mais generosa)
        progress_reward = distance_traveled * 5.0  # Recompensa por distância total percorrida

        # 2. Recompensa incremental por movimento para frente
        if hasattr(self, "prev_distance"):
            step_progress = distance_traveled - self.prev_distance
            if step_progress > 0:
                progress_reward += step_progress * 20.0  # Grande recompensa por progresso positivo
            else:
                progress_reward += step_progress * 10.0  # Penalidade por movimento para trás

        # 3. Recompensa por estabilidade (menor penalidade por movimento)
        joint_velocities = []
        for i in self.robot.revolute_indices:
            joint_state = p.getJointState(self.robot_id, i)
            joint_velocities.append(abs(joint_state[1]))

        # Penalidade muito menor por movimento
        movement_penalty = -0.001 * sum(joint_velocities)

        # 4. Recompensa por permanecer em pé
        standing_reward = 0.5 if pos[2] > 0.4 else -1.0

        # 5. Combina todas as recompensas
        reward = progress_reward + movement_penalty + standing_reward

        # 6. Penalidades e bônus finais
        done = False
        info = {"distance": distance_traveled, "success": False}

        # Queda
        if pos[2] < self.fall_threshold:
            reward -= 200  # Penalidade maior por queda
            done = True
            info["termination"] = "fell"
        # Sucesso
        elif distance_traveled >= self.success_distance:
            reward += 500  # Bônus muito maior por sucesso
            done = True
            info["success"] = True
            info["termination"] = "success"
        # Timeout
        elif self.steps >= self.max_steps:
            done = True
            info["termination"] = "timeout"
            # Recompensa adicional baseada no progresso final
            reward += distance_traveled * 2.0

        # Atualizar distância anterior
        self.prev_distance = distance_traveled

        return obs, reward, done, False, info

    def _get_observation(self):
        """
        Retorna o vetor de observação baseado em sensores IMU (simulados).
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)

        # Posição e velocidade linear no eixo X
        x_pos = pos[0]
        x_vel = vel[0]

        # Orientação (roll, pitch, yaw)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler

        # Ângulos e velocidades das juntas
        joint_angles = []
        joint_velocities = []
        for i in self.robot.revolute_indices:
            joint_state = p.getJointState(self.robot_id, i)
            joint_angles.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        # Montar vetor de observação
        obs = np.array([x_pos, x_vel, roll, pitch, yaw, *joint_angles, *joint_velocities], dtype=np.float32)

        return obs

    def render(self, mode="human"):
        pass  # O modo GUI é controlado por enable_gui no construtor

    def close(self):
        p.disconnect()
