# environment.py
import utils
import os
import pybullet as p
from xacrodoc import XacroDoc


class Environment:
    def __init__(self, name):
        self.name = name

        self.environment_dir = os.path.join(utils.PROJECT_ROOT, "environments")
        self.environment_tmp_dir = os.path.join(utils.TMP_PATH, "environments")

        if not os.path.exists(self.environment_tmp_dir):
            os.makedirs(self.environment_tmp_dir, exist_ok=True)

        self.urdf_path = self._generate_urdf()
        self.plane_id = None

    def _generate_urdf(self):
        xacro_path = os.path.join(self.environment_dir, f"{self.name}.xacro")
        urdf_path = os.path.join(self.environment_tmp_dir, f"{self.name}.urdf")

        if not os.path.exists(urdf_path):
            XacroDoc.from_file(xacro_path).to_urdf_file(urdf_path)

        return urdf_path

    def get_urdf_path(self):
        return self.urdf_path

    def load_in_simulation(self, use_fixed_base=True):
        self.plane_id = p.loadURDF(self.urdf_path, useFixedBase=use_fixed_base)
        return self.plane_id

    def get_plane_id(self):
        return self.plane_id
