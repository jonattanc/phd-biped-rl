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

        # ✅ PASSO CRÍTICO: REMOVER O ROBÔ ANTIGO E CARREGAR UM NOVO
        if self.robot_id is not None:
            p.removeBody(self.robot_id)

        # Recarregar ambiente (opcional, mas recomendado para garantir piso limpo)
        if self.plane_id is not None:
            p.removeBody(self.plane_id)
        self.plane_id = self.environment.load_in_simulation(use_fixed_base=True)

        # Recarregar o robô → isso garante que ele começa sempre na posição original definida no URDF
        self.robot_id = self.robot.load_in_simulation()

        # Atualizar os índices das juntas (caso tenham mudado)
        self.agent.set_revolute_indices(self.robot.revolute_indices)

        # Obter posição inicial do novo robô
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        initial_x_pos = pos[0]  # Posição inicial do episódio

        while steps < self.max_steps:
            # Obter observação
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            print(f"[DEBUG] Posição inicial do robô (base_link): x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            current_x_pos = pos[0]
            distance_traveled = current_x_pos - initial_x_pos  # Variação relativa ao início!

            # Calcular recompensa
            progress_reward = (distance_traveled - prev_x_pos) * 100
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
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.robot.revolute_indices,
                controlMode=p.TORQUE_CONTROL,
                forces=action
            )

            # Avançar simulação
            p.stepSimulation()
            steps += 1

            if steps % 100 == 0:
                self.logger.debug(f"Passo {steps} | Distância: {distance_traveled:.2f}m")

        total_time = time.time() - start_time
        self.logger.info(f"Episódio finalizado. Distância: {distance_traveled:.2f}m | Tempo: {total_time:.2f}s | Sucesso: {success}")

        return {
            "reward": reward,
            "time_total": total_time,
            "distance": distance_traveled,
            "success": success,
            "steps": steps
        }