# environment.py
import utils
import os
import pybullet as p
from xacrodoc import XacroDoc
import trimesh
import numpy as np


def get_env_file_variable(filename, variable_name):
    file_path = os.path.join(utils.PROJECT_ROOT, "environments", filename)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if f'name="{variable_name}"' in line and "value=" in line:
                start = line.find('value="') + len('value="')
                end = line.find('"', start)
                value = line[start:end]
                return float(value)

    return None


def create_ramp_stl(input_filename, output_filename, ascending=True):
    hypotenuse = get_env_file_variable(input_filename, "ramp_hypotenuse")
    width = get_env_file_variable(input_filename, "plane_width")
    angle_deg = get_env_file_variable(input_filename, "ramp_angle_deg")
    height = np.sin(np.radians(angle_deg)) * hypotenuse
    length = np.cos(np.radians(angle_deg)) * hypotenuse

    vertices = np.array(
        [
            [length, width, height],  # topo traseiro direito
            [0, width, 0],  # base traseiro direito
            [length, 0, height],  # topo frente direito
            [0, 0, 0],  # base frente direito
            [length, width, 0],  # base traseiro esquerdo
            [length, 0, 0],  # base frente esquerdo
        ]
    )

    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # base inferior
            [0, 4, 1],  # lado frontal inclinado
            [2, 3, 5],  # lado traseiro inclinado
            [0, 2, 4],
            [2, 5, 4],  # parede esquerda
            [1, 4, 3],
            [3, 4, 5],  # parede direita
        ]
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if not ascending:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle=np.radians(180), direction=[0, 0, 1], point=[0, 0, 0]))

    output_folder = os.path.join(utils.TMP_PATH, "environments")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    mesh.export(output_path)


class Environment:
    def __init__(self, logger, name, robot, is_fast_td3=False):
        self.logger = logger
        self.name = name
        self.robot = robot
        self.is_fast_td3 = is_fast_td3
        self.current_phase = 1

        self.id = None
        self.selected_env_index = -1
        self.urdf_path = None

        self.environment_dir = os.path.join(utils.PROJECT_ROOT, "environments")
        self.environment_tmp_dir = os.path.join(utils.TMP_PATH, "environments")

        if not os.path.exists(self.environment_tmp_dir):
            os.makedirs(self.environment_tmp_dir, exist_ok=True)

        create_ramp_stl("PRA1.xacro", "ramp_up_p1.stl", ascending=True)
        create_ramp_stl("PRA.xacro", "ramp_up.stl", ascending=True)
        
        self._init_env_list()

        # Gerar URDF e configurações para cada ambiente na lista
        self.environment_dict_settings = []
        for env_name in self.env_list:
            urdf_path = self._generate_urdf(env_name)
            self.robot.update_env(env_name)
            self.environment_dict_settings.append(self.get_environment_dict_settings(env_name))

            # Definir o primeiro URDF como caminho padrão
            if self.urdf_path is None:
                self.urdf_path = urdf_path

        self.environment_settings = self.environment_dict_settings[0]

    def _init_env_list(self):
        """Atualiza a lista de ambientes com base na fase atual (apenas para FastTD3)"""
        if self.is_fast_td3 and self.current_phase in [1, 2]:
            if self.name == "CC":
                self.env_list = ["PR", "PBA", "PG", "PRA1", "PRB", "PRD"]
            else:
                self.env_list = [self.name]
        else:
            if self.name == "CC":
                self.env_list = ["PR", "PBA", "PG", "PRA", "PRB", "PRD"]
            else:     
                self.env_list = [self.name]
       
    def _update_env_list(self):
        """Atualiza a lista de ambientes (apenas para FastTD3 ao mudar de fase)"""
        if not self.is_fast_td3:
            return  # Não precisa atualizar para não-FastTD3

        # Salvar a lista anterior para comparação
        old_env_list = self.env_list.copy() if hasattr(self, 'env_list') else []

        # Definir nova lista
        self._init_env_list()

        # Verificar se a lista mudou
        if old_env_list != self.env_list:
            self.logger.info(f"🔄 Lista de ambientes atualizada: {self.env_list}")

            # Regenerar URDFs e configurações apenas se a lista mudou
            self.environment_dict_settings = []
            for env_name in self.env_list:
                self._generate_urdf(env_name)
                self.robot.update_env(env_name)
                self.environment_dict_settings.append(self.get_environment_dict_settings(env_name))

            # Ajustar índice se necessário (não resetar para -1!)
            if self.selected_env_index >= len(self.env_list):
                self.selected_env_index = len(self.env_list) - 1

            # Atualizar ambiente atual com o novo índice
            env_name = self.env_list[self.selected_env_index]
            self.urdf_path = os.path.join(self.environment_tmp_dir, f"{env_name}.urdf")
            self.environment_settings = self.environment_dict_settings[self.selected_env_index]
            self.robot.update_env(env_name)

    def set_phase(self, phase):
        """Apenas para FastTD3: atualiza a fase atual das rampas"""
        if self.is_fast_td3:
            self.current_phase = phase
            self._update_env_list()
            
    def _generate_urdf(self, env_name):
        xacro_path = os.path.join(self.environment_dir, f"{env_name}.xacro")
        urdf_path = os.path.join(self.environment_tmp_dir, f"{env_name}.urdf")

        if not os.path.exists(urdf_path):
            XacroDoc.from_file(xacro_path).to_urdf_file(urdf_path)

        return urdf_path

    def get_next_env(self):
        if self.name != "CC":
            return

        # Incrementar índice
        self.selected_env_index += 1

        # Verificar se ultrapassou o final da lista
        if self.selected_env_index >= len(self.env_list):
            self.selected_env_index = 0  # Reiniciar do primeiro ambiente

        env_name = self.env_list[self.selected_env_index]

        # Atualizar caminho do URDF
        self.urdf_path = os.path.join(self.environment_tmp_dir, f"{env_name}.urdf")

        # Atualizar configurações do ambiente
        self.environment_settings = self.environment_dict_settings[self.selected_env_index]

        # Atualizar o robô para o novo ambiente
        self.robot.update_env(env_name)

    def get_environment_dict_settings(self, env):
        if self.is_fast_td3:
            environment_settings = {"default": {"lateral_friction": 2.0, "spinning_friction": 1.0, "rolling_friction": 0.001, "restitution": 0.0}}
            if env == "PBA":
                environment_settings["middle_link"] = {
                    "lateral_friction": 0.4,
                    "spinning_friction": 0.2,
                    "rolling_friction": environment_settings["default"]["rolling_friction"],
                    "restitution": environment_settings["default"]["restitution"],
                }
            elif env == "PG":
                environment_settings["middle_link"] = {
                    "lateral_friction": 1.5,
                    "spinning_friction": 0.75,
                    "rolling_friction": environment_settings["default"]["rolling_friction"],
                    "restitution": environment_settings["default"]["restitution"],
                    "contactStiffness": 5e4,
                    "contactDamping": 500,
                }
            elif env == "PRA" or env == "PRD":
                environment_settings["middle_link"] = {"lateral_friction": 2.0, "spinning_friction": 2.0, "rolling_friction": 0.01, "restitution": 0.0}

        else:
            environment_settings = {"default": {"lateral_friction": 2.0, "spinning_friction": 1.0, "rolling_friction": 0.001, "restitution": 0.0}}
            if env == "PBA":
                environment_settings["middle_link"] = {
                    "lateral_friction": 0.85,
                    "spinning_friction": 0.425,
                    "rolling_friction": environment_settings["default"]["rolling_friction"],
                    "restitution": environment_settings["default"]["restitution"],
                }
            elif env == "PG":
                environment_settings["middle_link"] = {
                    "lateral_friction": environment_settings["default"]["lateral_friction"],
                    "spinning_friction": environment_settings["default"]["spinning_friction"],
                    "rolling_friction": environment_settings["default"]["rolling_friction"],
                    "restitution": environment_settings["default"]["restitution"],
                    "contactStiffness": 5e4,
                    "contactDamping": 500,
                }
            elif env == "PRA" or env == "PRD":
                environment_settings["middle_link"] = {"lateral_friction": 10.0, "spinning_friction": 2.0, "rolling_friction": 0.01, "restitution": 0.0}

        self.logger.info(f"{env} environment_settings: {environment_settings}")
        return environment_settings

    def load_in_simulation(self, use_fixed_base=True):
        self.get_next_env()
        self.id = p.loadURDF(self.urdf_path, useFixedBase=use_fixed_base)

        return self.id

    def get_num_joints(self):
        return p.getNumJoints(self.id)

    def get_link_indices_by_name(self):
        name_to_index = {"base_link": -1}

        for i in range(p.getNumJoints(self.id)):
            name = p.getJointInfo(self.id, i)[12].decode("utf-8")
            name_to_index[name] = i

        return name_to_index
