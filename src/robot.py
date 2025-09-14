import os
import logging
import pybullet as p
from xacrodoc import XacroDoc


class Robot:
    def __init__(self, name, base_position=[0, 0, 0], base_orientation=[0, 0, 0]):
        self.name = name
        self.base_position = base_position
        self.base_orientation = p.getQuaternionFromEuler(base_orientation)  # x roll, y pitch, z yaw -> x, y, z, w

        self.tmp_dir = "tmp"
        self.models_dir = os.path.join("models", "robots")
        self.models_tmp_dir = os.path.join(self.tmp_dir, self.models_dir)

        if not os.path.exists(self.models_tmp_dir):
            os.makedirs(self.models_tmp_dir, exist_ok=True)

        self.urdf_path = self._generate_urdf()
        self.body_id = None
        self.logger = logging.getLogger(__name__)

    def _generate_urdf(self):
        xacro_path = os.path.join(self.models_dir, f"{self.name}.xacro")
        urdf_path = os.path.join(self.models_tmp_dir, f"{self.name}.urdf")
        XacroDoc.from_file(xacro_path).to_urdf_file(urdf_path)
        return urdf_path

    def load_in_simulation(self):
        self.body_id = p.loadURDF(self.urdf_path, self.base_position, self.base_orientation)
        return self.body_id

    def get_body_id(self):
        return self.body_id

    def reset_base_position_and_orientation(self, base_position=None, base_orientation=None):
        if base_position is None:
            base_position = self.base_position

        if base_orientation is None:
            base_orientation = self.base_orientation

        p.resetBasePositionAndOrientation(self.body_id, base_position, base_orientation)
