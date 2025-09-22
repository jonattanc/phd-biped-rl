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

        # Configurações de simulação
        self.physics_client = None
        self.steps = 0
        self.max_steps = 5000  # ~20.8 segundos (240 * 20.8)
        self.time_step_s = 1 / 240.0
        self.success_distance = 10.0
        self.fall_threshold = 0.0
        self.initial_x_pos = 0.0
        self.prev_distance = 0.0

        # Variáveis para coleta de dados
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.episode_distance = 0.0
        self.episode_success = False
        self.episode_info = {}

        self.logger = logging.getLogger(__name__)
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
        if self.robot.id is not None:
            p.removeBody(self.robot.id)
        if hasattr(self.environment, 'id') and self.environment.id is not None:
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
        """
        Reinicia o ambiente e retorna o estado inicial.
        """
        # Se seed for fornecida, configure o gerador de números aleatórios
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if hasattr(self, 'episode_start_time') and self.episode_start_time > 0:
            episode_duration = time.time() - self.episode_start_time
            self.episode_info = {
                'reward': self.episode_reward,
                'time': episode_duration,
                'distance': self.episode_distance,
                'success': self.episode_success,
                'steps': self.steps
            }
        
        # Reiniciar variáveis do episódio
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.episode_distance = 0.0
        self.episode_success = False
        self.steps = 0
        self.prev_distance = 0.0
        
        # Reiniciar simulação
        self.robot.reset_base_position_and_orientation()
        self.initial_x_pos = 0.0

        # Retornar observação inicial
        obs = self.robot.get_observation()
        return obs, {}

    def step(self, action):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        while self.pause_value.value and not self.exit_value.value:
            time.sleep(0.1)

        if self.exit_value.value:
            self.logger.info("Sinal de saída recebido. Finalizando simulação.")
            return None, 0.0, True, False, {"exit": True}

        # Aplicar ação
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.id, 
            jointIndices=self.robot.revolute_indices, 
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocities=action, 
            forces=[100] * len(action)
            )

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
        self.episode_distance = distance_traveled

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
            joint_state = p.getJointState(self.robot.id, i)
            joint_velocities.append(abs(joint_state[1]))
        movement_penalty = -0.001 * sum(joint_velocities)

        # 4. Recompensa por permanecer em pé
        standing_reward = 0.5 if pos[2] > 0.4 else -1.0

        # 5. Combina todas as recompensas
        reward = progress_reward + movement_penalty + standing_reward
        self.episode_reward += reward

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

        # Coletar info final quando o episódio terminar
        info = {
            "distance": distance_traveled, 
            "success": False,
            "episode": {
                "r": self.episode_reward,
                "l": self.steps
            }
        }

        return obs, reward, done, False, info

    def get_episode_info(self):
        """Retorna informações do episódio atual"""
        return self.episode_info.copy()
    
    def render(self, mode="human"):
        pass  # O modo GUI é controlado por enable_gui no construtor

    def close(self):
        p.disconnect()
