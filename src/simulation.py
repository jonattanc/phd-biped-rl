# simulation.py

import pybullet as p
import pybullet_data
import time
import logging
from datetime import datetime


class Simulation:
    def __init__(self, robot, environment, agent, pause_value, exit_value, enable_real_time_value, enable_gui=True, num_episodes=1):
        self.robot = robot
        self.environment = environment
        self.agent = agent
        self.pause_value = pause_value
        self.exit_value = exit_value
        self.enable_real_time_value = enable_real_time_value
        self.enable_gui = enable_gui
        self.num_episodes = num_episodes

        self.logger = logging.getLogger(__name__)
        self.physics_client = None
        self.plane_id = None
        self.robot_id = None

        # Configurações de simulação
        self.time_step_s = 1 / 240.0
        self.max_steps = 5000  # ~20.8 segundos (240 * 20.8)
        self.success_distance = 10.0
        self.fall_threshold = 0.0  # altura mínima para considerar queda

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
        self.agent.set_revolute_indices(self.robot.revolute_indices)
        self.logger.info(f"Simulação configurada: {len(self.robot.revolute_indices)} DOFs")
        self.logger.info(f"Robô: {self.robot.name}")
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
        self.agent.set_revolute_indices(self.robot.revolute_indices)

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
            action = self.agent.get_action()

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
