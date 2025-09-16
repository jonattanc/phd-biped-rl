# robot.py
import os
import logging
import pybullet as p
from xacrodoc import XacroDoc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Robot:
    def __init__(self, name):
        self.name = name

        self.tmp_dir = "tmp"
        self.models_dir = os.path.join(PROJECT_ROOT, "robots")
        self.models_tmp_dir = os.path.join(self.tmp_dir, self.models_dir)

        if not os.path.exists(self.models_tmp_dir):
            os.makedirs(self.models_tmp_dir, exist_ok=True)

        self.urdf_path = self._generate_urdf()
        self.body_id = None
        self.logger = logging.getLogger(__name__)

    def _generate_urdf(self):
        xacro_path = os.path.join(self.models_dir, f"{self.name}.xacro")
        urdf_path = os.path.join(self.models_tmp_dir, f"{self.name}.urdf")

        if os.path.exists(urdf_path):
            os.remove(urdf_path)

        XacroDoc.from_file(xacro_path).to_urdf_file(urdf_path)
        return urdf_path

    def load_in_simulation(self):
        self.body_id = p.loadURDF(self.urdf_path)

        num_joints = p.getNumJoints(self.body_id)
        self.revolute_indices = [i for i in range(num_joints) if p.getJointInfo(self.body_id, i)[2] == p.JOINT_REVOLUTE]
        self.logger.info(f"Robot {self.name} loaded with {num_joints} joints, revolute joints at indices: {self.revolute_indices}")

        initial_position = (0, 0, 0.45)  # Eleva o centro do corpo
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.body_id, initial_position, initial_orientation)
        for j in range(p.getNumJoints(self.body_id)):
            p.resetJointState(self.body_id, j, 0.0)

        # Salva o estado inicial para resets futuros
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation

        self.initial_joint_states = []

        for j in range(p.getNumJoints(self.body_id)):
            joint_info = p.getJointState(self.body_id, j)
            self.initial_joint_states.append(joint_info[0])

        return self.body_id

    def get_body_id(self):
        return self.body_id

    def reset_base_position_and_orientation(self):
        p.resetBasePositionAndOrientation(self.body_id, self.initial_position, self.initial_orientation)

        for j, angle in enumerate(self.initial_joint_states):
            p.resetJointState(self.body_id, j, angle)
