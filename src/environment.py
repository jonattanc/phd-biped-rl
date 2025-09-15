#environment.py
import os
import pybullet as p
from xacrodoc import XacroDoc
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Environment:
    def __init__(self, name):
        self.name = name
        self.tmp_dir = "tmp"
        self.models_dir = os.path.join(PROJECT_ROOT, "models", "environments")
        self.models_tmp_dir = os.path.join(self.tmp_dir, self.models_dir)

        if not os.path.exists(self.models_tmp_dir):
            os.makedirs(self.models_tmp_dir, exist_ok=True)

        self.urdf_path = self._generate_urdf()
        self.plane_id = None

    def _generate_urdf(self):
        xacro_path = os.path.join(self.models_dir, f"{self.name}.xacro")
        urdf_path = os.path.join(self.models_tmp_dir, f"{self.name}.urdf")
        XacroDoc.from_file(xacro_path).to_urdf_file(urdf_path)
        return urdf_path

    def get_urdf_path(self):
        return self.urdf_path

    def load_in_simulation(self, use_fixed_base=True):
        self.plane_id = p.loadURDF(self.urdf_path, useFixedBase=use_fixed_base)
        return self.plane_id

    def get_plane_id(self):
        return self.plane_id
