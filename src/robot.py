# robot.py
import utils
import os
import pybullet as p
from xacrodoc import XacroDoc
import numpy as np


class Robot:
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

        self.id = None
        self.revolute_indices = None

        self.robots_dir = os.path.join(utils.PROJECT_ROOT, "robots")
        self.robots_tmp_dir = os.path.join(utils.TMP_PATH, "robots")

        if not os.path.exists(self.robots_tmp_dir):
            os.makedirs(self.robots_tmp_dir, exist_ok=True)

        self.urdf_path = self._generate_urdf()

    def _generate_urdf(self):
        xacro_path = os.path.join(self.robots_dir, f"{self.name}.xacro")
        urdf_path = os.path.join(self.robots_tmp_dir, f"{self.name}.urdf")

        if not os.path.exists(urdf_path):
            XacroDoc.from_file(xacro_path).to_urdf_file(urdf_path)

        return urdf_path

    def get_link_index(self, link_name):
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)

            if info[12].decode("utf-8") == link_name:  # Looks the name of the child link connected to the joint
                return i

        raise (f"Link index not found for {link_name}")

    def load_in_simulation(self):
        self.id = p.loadURDF(self.urdf_path)

        self.imu_link_index = self.get_link_index("imu_link")

        num_joints = p.getNumJoints(self.id)
        self.revolute_indices = [i for i in range(num_joints) if p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE]

        return self.id

    def get_num_joints(self):
        return p.getNumJoints(self.id)

    def get_num_revolute_joints(self):
        if self.revolute_indices is None:
            raise ValueError("O robô ainda não foi carregado na simulação. Chame load_in_simulation() primeiro.")

        return len(self.revolute_indices)

    def get_joint_states(self):
        """Retorna posições e velocidades das juntas COM VERIFICAÇÃO"""
        joint_states = p.getJointStates(self.id, self.revolute_indices)
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]
        return joint_positions, joint_velocities

    def get_observation(self):
        """Retorna observação"""
        if self.id is None:
            return np.zeros(10, dtype=np.float32)

        link_state = p.getLinkState(self.id, self.imu_link_index, computeLinkVelocity=1)
        position, orientation = link_state[0], link_state[1]
        linear_velocity, angular_velocity = link_state[6], link_state[7]
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        x_velocity = linear_velocity[0]
        roll_velocity = angular_velocity[0]
        pitch_velocity = angular_velocity[1]

        joint_positions, joint_velocities = self.get_joint_states()

        obs = np.array([roll, pitch, yaw, x_velocity, roll_velocity, pitch_velocity] + joint_positions, dtype=np.float32)
        return obs

    def get_imu_position_velocity_orientation(self):
        """Retorna posição e orientação do IMU COM VERIFICAÇÃO"""
        link_state = p.getLinkState(self.id, self.imu_link_index, computeLinkVelocity=1)
        position = link_state[0]
        orientation = p.getEulerFromQuaternion(link_state[1])
        linear_velocity = link_state[6]
        return position, linear_velocity, orientation

    def get_base_position_and_orientation(self):
        """Retorna a posição e orientação atual da base do robô"""
        return p.getBasePositionAndOrientation(self.id)

    def get_example_action(self, t):
        """Gera uma ação de exemplo baseada no tempo"""
        num_joints = self.get_num_revolute_joints()

        if num_joints == 4:
            hip_right = 0
            knee_right = 0
            hip_left = 0
            knee_left = 0

            action_list = [hip_right, knee_right, hip_left, knee_left]

        elif num_joints == 6:
            f = 0.8  # Frequência do movimento
            w = 2 * np.pi * f  # Velocidade angular

            hip_right = -1.0 * np.sin(w * t + 0.0 * np.pi)
            knee_right = 1.0 * np.sin(w * t + 0.0 * np.pi)
            ankle_right = 0.0 * np.sin(w * t + 0.0 * np.pi)
            hip_left = 0.0 * np.sin(w * t + 0.0 * np.pi)
            knee_left = 0.0 * np.sin(w * t + 0.0 * np.pi)
            ankle_left = -1.0 * np.sin(w * t + 0.0 * np.pi)

            action_list = [hip_right, knee_right, ankle_right, hip_left, knee_left, ankle_left]

        elif num_joints == 8:
            t1 = 1.55
            t2 = t1 + 0.5
            t3 = t2 + 0.75
            t4 = t3 + 0.2

            hip_right_front = 0  # Positivo para trás
            hip_right_lateral = 0  # Positivo para dentro
            knee_right = 0  # Positivo para dobrar
            ankle_right = 0  # Positivo para baixo
            hip_left_front = 0
            hip_left_lateral = 0  # Positivo para fora
            knee_left = 0
            ankle_left = 0  # Positivo para baixo

            if t < t1:
                hip_right_lateral = -0.05
                # ankle_right = 0.05
                hip_left_lateral = -0.05

            elif t < t2:
                ankle_right = -0.05
                ankle_left = -0.05

            elif t < t3:
                hip_right_front = -0.5
                knee_right = 0.5

            elif t < t4:
                hip_right_front = 0.5
                knee_right = -0.5
                hip_right_lateral = 0.01
                hip_left_lateral = 0.01

            action_list = [hip_right_front, hip_right_lateral, knee_right, ankle_right, hip_left_front, hip_left_lateral, knee_left, ankle_left]

        else:
            raise ValueError(f"Número de juntas não suportado para ação de exemplo: {num_joints}")

        return np.array(action_list, dtype=np.float32)
