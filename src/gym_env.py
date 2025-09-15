# gym_env.py
import gymnasium as gym
import numpy as np
import pybullet as p
import random
from simulation import Simulation
from robot import Robot
from environment import Environment
from agent import Agent

class ExoskeletonPRst1(gym.Env):
    """
    Ambiente Gym para o robô robot_stage1 no circuito PR.
    """

    def __init__(self, enable_gui=False):
        super(ExoskeletonPRst1, self).__init__()

        # Inicializar componentes
        self.robot = Robot(name="robot_stage1")
        self.environment = Environment(name="PR")
        self.agent = Agent()  # Será usado apenas para obter índices das juntas
        self.sim = Simulation(self.robot, self.environment, self.agent, enable_gui=enable_gui)

        # Configurar simulação
        self.sim.setup()

        # Definir espaço de ação e observação
        # Ação: velocidade alvo para cada junta revolute (4 juntas)
        self.action_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(4,),  # robot_stage1 tem 4 juntas revolute
            dtype=np.float32
        )

        # Observação: [posição X, velocidade X, orientação (roll, pitch, yaw), 4 ângulos de junta, 4 velocidades de junta]
        # Total: 1 + 1 + 3 + 4 + 4 = 13 valores
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        # Variáveis de estado
        self.steps = 0
        self.max_steps = 5000
        self.success_distance = 10.0
        self.fall_threshold = 0.3
        self.initial_x_pos = 0.0

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente e retorna o estado inicial.
        """
        # Se seed for fornecida, configure o gerador de números aleatórios
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reiniciar simulação (o método run() já faz isso, mas vamos garantir)
        if self.sim.robot_id is not None:
            p.removeBody(self.sim.robot_id)
        if self.sim.plane_id is not None:
            p.removeBody(self.sim.plane_id)

        self.sim.plane_id = self.sim.environment.load_in_simulation(use_fixed_base=True)
        self.sim.robot_id = self.sim.robot.load_in_simulation()
        p.resetBasePositionAndOrientation(self.sim.robot_id, [0, 0, 0.45], p.getQuaternionFromEuler([0, 0, 0]))

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
        p.setJointMotorControlArray(
            bodyUniqueId=self.sim.robot_id,
            jointIndices=self.sim.robot.revolute_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=action,
            forces=[100] * len(action)
        )

        # Avançar simulação
        p.stepSimulation()
        self.steps += 1

        # Obter observação
        obs = self._get_observation()
        pos, _ = p.getBasePositionAndOrientation(self.sim.robot_id)
        current_x_pos = pos[0]
        distance_traveled = current_x_pos - self.initial_x_pos

        # Calcular recompensa (mesma lógica da simulação)
        progress = distance_traveled - (self.initial_x_pos if self.steps == 1 else self.prev_distance)
        if progress > 0:
            progress_reward = progress * 10
        else:
            progress_reward = progress * 20
        reward = progress_reward

        # Penalidades e bônus
        done = False
        if pos[2] < self.fall_threshold:
            reward -= 100
            done = True
        elif distance_traveled >= self.success_distance:
            reward += 50
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # Atualizar distância anterior
        self.prev_distance = distance_traveled

        # Informações adicionais
        info = {
            "distance": distance_traveled,
            "success": distance_traveled >= self.success_distance and not done
        }

        return obs, reward, done, False, info

    def _get_observation(self):
        """
        Retorna o vetor de observação baseado em sensores IMU (simulados).
        """
        pos, orn = p.getBasePositionAndOrientation(self.sim.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.sim.robot_id)

        # Posição e velocidade linear no eixo X
        x_pos = pos[0]
        x_vel = vel[0]

        # Orientação (roll, pitch, yaw)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler

        # Ângulos e velocidades das juntas
        joint_angles = []
        joint_velocities = []
        for i in self.sim.robot.revolute_indices:
            joint_state = p.getJointState(self.sim.robot_id, i)
            joint_angles.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        # Montar vetor de observação
        obs = np.array([
            x_pos, x_vel,
            roll, pitch, yaw,
            *joint_angles,
            *joint_velocities
        ], dtype=np.float32)

        return obs

    def render(self, mode='human'):
        pass  # O modo GUI é controlado por enable_gui no construtor

    def close(self):
        p.disconnect()