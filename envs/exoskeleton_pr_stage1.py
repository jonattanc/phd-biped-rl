import os
import time
import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, Dict, Any
from robot import Robot

class ExoskeletonPRst1:
    """
    Ambiente de Simulação para o circuito PR (Piso Regular).
    Usa o robô 'robot_stage1' com apenas 2 articulações.
    """

    def __init__(
        self,
        robot_name: str = "robot_stage1",
        plane_name: str = "PR",  
        enable_gui: bool = False,
        time_limit: float = 20.0,
        target_distance: float = 10.0,
        timestep: float = 1/240.0,
        control_freq: int = 30,
    ):
        # Parâmetros
        self.robot_name = robot_name
        self.plane_name = plane_name 
        self.enable_gui = enable_gui
        self.time_limit = time_limit
        self.target_distance = target_distance
        self.timestep = timestep
        self.control_freq = control_freq
        self.control_steps_per_env_step = max(1, int(1 / (control_freq * timestep)))

        # Define o caminho base do projeto (assumindo que este script está em 'envs/')
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_env_path = os.path.join(self.project_root, "models", "environments")

        # Estado da simulação
        self.physics_client = None
        self.robot = None
        self.plane_id = None
        self.current_time = 0.0
        self.distance_traveled = 0.0
        self.prev_x_pos = 0.0

        # Conectar ao PyBullet
        self._connect_physics_client()
        self.reset() # Inicializa o ambiente

    def _connect_physics_client(self):
        """Conecta ao servidor PyBullet."""
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.807)
        p.setTimeStep(self.timestep)
        # Adiciona o caminho dos modelos de ambiente à lista de busca
        p.setAdditionalSearchPath(self.models_env_path)
        # Também mantém o caminho padrão do PyBullet para outros assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reseta o ambiente para o estado inicial.
        Returns:
            obs: Observação inicial.
            info: Dicionário com informações.
        """
        # Reinicia a simulação
        if self.robot is not None:
            p.resetSimulation()
            self.robot = None
            self.plane_id = None

        # Constrói o caminho completo para o arquivo URDF do ambiente
        plane_urdf_path = os.path.join(self.models_env_path, f"{self.plane_name}.urdf")
        
        # Verifica se o arquivo existe
        if not os.path.exists(plane_urdf_path):
            raise FileNotFoundError(f"Arquivo de ambiente não encontrado: {plane_urdf_path}")

        # Carrega o piso a partir do arquivo URDF personalizado
        self.plane_id = p.loadURDF(f"{self.plane_name}.urdf")

        # Se precisar configurar o atrito dinamicamente após o carregamento, descomente:
        # p.changeDynamics(self.plane_id, -1, lateralFriction=0.8)

        # Posição inicial do robô
        initial_base_position = [0, 0, 0.7]
        initial_base_orientation = [0, 0, 0]

        # Cria e carrega o robô
        self.robot = Robot(
            name=self.robot_name,
            base_position=initial_base_position,
            base_orientation=initial_base_position
        )
        self.robot.load_in_simulation()

        # Pequena perturbação aleatória nas articulações (±5°)
        for joint_index in [0, 1]:
            noise = np.random.uniform(-5, 5) * np.pi / 180.0
            p.resetJointState(self.robot.get_body_id(), joint_index, targetValue=noise)

        # Reseta variáveis
        self.current_time = 0.0
        self.distance_traveled = 0.0
        pos, _ = p.getBasePositionAndOrientation(self.robot.get_body_id())
        self.prev_x_pos = pos[0]

        # Retorna observação inicial
        obs = self._get_observation()
        info = {'reset': True}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa um passo na simulação.
        Args:
            action: Array numpy de torques para as 2 articulações.
        Returns:
            obs: Observação do estado atual.
            reward: Recompensa recebida.
            terminated: True se o episódio terminou (sucesso ou falha).
            truncated: True se o episódio foi truncado (timeout).
            info: Dicionário com informações adicionais.
        """
        # Aplica a ação (torques)
        p.setJointMotorControlArray(
            self.robot.get_body_id(),
            jointIndices=[0, 1],
            controlMode=p.TORQUE_CONTROL,
            forces=action
        )

        # Executa a simulação
        for _ in range(self.control_steps_per_env_step):
            p.stepSimulation()
            if self.enable_gui:
                time.sleep(self.timestep)

        # Atualiza tempo
        self.current_time += self.control_steps_per_env_step * self.timestep

        # Obtém nova observação
        obs = self._get_observation()

        # Calcula recompensa
        reward, distance_reward, penalty = self._compute_reward()

        # Verifica término
        terminated, truncated, info = self._check_termination()

        # Informações adicionais
        info.update({
            'distance_traveled': self.distance_traveled,
            'reward_components': {
                'distance': distance_reward,
                'penalty': penalty,
            },
            'time': self.current_time,
        })

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Obtém a observação do ambiente."""
        body_id = self.robot.get_body_id()
        pos, orn = p.getBasePositionAndOrientation(body_id)
        vel, ang_vel = p.getBaseVelocity(body_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        joint_states = p.getJointStates(body_id, jointIndices=[0, 1])
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        obs = np.array([
            pos[0], pos[2],
            vel[0], vel[2],
            pitch, ang_vel[1],
            joint_positions[0], joint_velocities[0],
            joint_positions[1], joint_velocities[1],
        ], dtype=np.float32)
        return obs

    def _compute_reward(self) -> Tuple[float, float, float]:
        """Função de recompensa simples."""
        current_x_pos, _, _ = p.getBasePositionAndOrientation(self.robot.get_body_id())[0]
        distance_this_step = current_x_pos - self.prev_x_pos
        self.distance_traveled += distance_this_step
        self.prev_x_pos = current_x_pos
        distance_reward = distance_this_step * 10.0
        step_penalty = -0.1
        fall_penalty = -100.0 if self._is_fallen() else 0.0
        total_reward = distance_reward + step_penalty + fall_penalty
        return total_reward, distance_reward, fall_penalty + step_penalty

    def _is_fallen(self) -> bool:
        """Verifica se o robô caiu."""
        pos, orn = p.getBasePositionAndOrientation(self.robot.get_body_id())
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        is_too_low = pos[2] < 0.3
        is_too_tilted = abs(pitch) > 1.0 or abs(roll) > 1.0
        return is_too_low or is_too_tilted

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        """Verifica as condições de término do episódio."""
        info = {}
        if self.distance_traveled >= self.target_distance:
            return True, False, info
        if self._is_fallen():
            return True, False, info
        if self.current_time >= self.time_limit:
            return False, True, info
        return False, False, info

    def close(self):
        """Fecha a conexão com o PyBullet."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
