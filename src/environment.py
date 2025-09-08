import os
import pybullet as p
import pybullet_data


class Environment:
    def __init__(self, name):
        self.name = name
        self.models_dir = "models/environments"

        self.urdf_path = self._get_urdf_path()
        self.plane_id = None

    def _get_urdf_path(self):
        urdf_path = os.path.join(self.models_dir, f"{self.name}.urdf")

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"Environment URDF not found: {urdf_path}")

        return urdf_path

    def get_urdf_path(self):
        return self.urdf_path

    def load_in_simulation(self, use_fixed_base=True):
        self.plane_id = p.loadURDF(self.urdf_path, useFixedBase=use_fixed_base)
        return self.plane_id

    def get_plane_id(self):
        return self.plane_id
