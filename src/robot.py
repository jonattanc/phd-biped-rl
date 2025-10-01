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
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.id)
        self.initial_joint_states = []

        for j in range(p.getNumJoints(self.id)):
            joint_info = p.getJointState(self.id, j)
            self.initial_joint_states.append(joint_info[0])

        return self.id

    def reset_base_position_and_orientation(self):
        p.resetBasePositionAndOrientation(self.id, self.initial_position, self.initial_orientation)

        for j, angle in enumerate(self.initial_joint_states):
            p.resetJointState(self.id, j, angle)

    def get_num_joints(self):
        return p.getNumJoints(self.id)

    def get_num_revolute_joints(self):
        if self.revolute_indices is None:
            raise ValueError("O robô ainda não foi carregado na simulação. Chame load_in_simulation() primeiro.")

        return len(self.revolute_indices)

    def get_joint_states(self):
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

        joint_positions, joint_velocities = self.get_joint_states()

        obs = np.array([roll, pitch, yaw] + list(angular_velocity) + joint_positions, dtype=np.float32)
        return obs

    def get_imu_position_and_orientation(self):
        link_state = p.getLinkState(self.id, self.imu_link_index, computeLinkVelocity=1)
        position = link_state[0]
        orientation = p.getEulerFromQuaternion(link_state[1])
        return position, orientation

    def get_base_position_and_orientation(self):
        """Retorna a posição e orientação atual da base do robô"""
        if self.id is None:
            return [0, 0, 0], [0, 0, 0, 1]

        return p.getBasePositionAndOrientation(self.id)
