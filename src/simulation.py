# simulation.py

import pybullet as p
import pybullet_data
import time
import logging
from datetime import datetime


class Simulation:
    def __init__(self, robot, environment, agent, enable_gui=True):
        self.robot = robot
        self.environment = environment
        self.agent = agent
        self.enable_gui = enable_gui
        self.logger = logging.getLogger(__name__)
        self.physics_client = None
        self.plane_id = None
        self.robot_id = None
        # Configurações de simulação
        self.time_step = 1 / 240.0
        self.max_steps = 5000  # ~20.8 segundos (240 * 20.8)
        self.success_distance = 10.0
        self.fall_threshold = 0.3  # altura mínima para considerar queda

    def setup(self):
        """Conecta ao PyBullet e carrega ambiente e robô"""
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.time_step)
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
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.45], p.getQuaternionFromEuler([0, 0, 0]))
        # Forçar a referência de distância para 0.0
        initial_x_pos = 0.0

        # Atualizar os índices das juntas (caso tenham mudado)
        self.agent.set_revolute_indices(self.robot.revolute_indices)

        while steps < self.max_steps:
            # Obter observação
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            print(f"[DEBUG] Posição inicial do robô (base_link): x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            current_x_pos = pos[0]
            distance_traveled = current_x_pos - initial_x_pos  # Agora, initial_x_pos é SEMPRE 0.0

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

            if steps % 100 == 0:
                self.logger.debug(f"Passo {steps} | Distância: {distance_traveled:.2f}m")

        total_time = time.time() - start_time
        self.logger.info(f"Episódio finalizado. Distância: {distance_traveled:.2f}m | Tempo: {total_time:.2f}s | Sucesso: {success}")

        return {"reward": reward, "time_total": total_time, "distance": distance_traveled, "success": success, "steps": steps}
