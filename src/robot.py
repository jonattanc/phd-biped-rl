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

    def get_joint_index(self, joint_name):
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)

            if info[1].decode("utf-8") == joint_name:
                return i

        raise (f"Joint index not found for {joint_name}")

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

    def get_joint_angle(self, joint_name):
        """Retorna o ângulo de uma junta específica"""
        joint_index = self.get_joint_index(joint_name)
        joint_angle = p.getJointState(self.id, joint_index)[0]
        return joint_angle

    def get_knee_angles(self):
        """Retorna os ângulos dos joelhos direito e esquerdo"""
        return self.get_joint_angle("right_knee_ball_to_shin"), self.get_joint_angle("left_knee_ball_to_shin")

    def get_hip_front_angles(self):
        """Retorna os ângulos dos quadris direito e esquerdo"""
        return self.get_joint_angle("base_to_right_hip_ball"), self.get_joint_angle("base_to_left_hip_ball")

    def get_foot_contact_states(self):
        """Retorna os estados de contato dos pés direito e esquerdo"""
        right_foot_contacts = p.getContactPoints(bodyA=self.id, linkIndexA=self.get_link_index("right_foot_link"))
        left_foot_contacts = p.getContactPoints(bodyA=self.id, linkIndexA=self.get_link_index("left_foot_link"))
        return (len(right_foot_contacts) > 0), (len(left_foot_contacts) > 0)

    def get_foot_heights(self):
        """Retorna as alturas dos pés direito e esquerdo em relação ao solo"""
        right_foot_state = p.getLinkState(self.id, self.get_link_index("right_foot_link"))
        left_foot_state = p.getLinkState(self.id, self.get_link_index("left_foot_link"))
        right_foot_height = right_foot_state[0][2]
        left_foot_height = left_foot_state[0][2]
        return right_foot_height, left_foot_height

    def get_center_of_mass(self):
        total_mass = 0.0
        weighted_pos = np.zeros(3)

        base_pos, _ = p.getBasePositionAndOrientation(self.id)
        base_mass = p.getDynamicsInfo(self.id, -1)[0]
        weighted_pos += base_mass * np.array(base_pos)
        total_mass += base_mass

        num_joints = p.getNumJoints(self.id)

        for i in range(num_joints):
            link_state = p.getLinkState(self.id, i, computeForwardKinematics=True)
            link_com_pos = np.array(link_state[0])
            link_mass = p.getDynamicsInfo(self.id, i)[0]
            weighted_pos += link_mass * link_com_pos
            total_mass += link_mass

        return weighted_pos / total_mass if total_mass > 0 else np.zeros(3)

    def get_shoulder_angles(self):
        """Retorna os ângulos dos ombros direito e esquerdo"""
        try:
            right_shoulder_angle = self.get_joint_angle("base_to_right_shoulder_front")
            left_shoulder_angle = self.get_joint_angle("base_to_left_shoulder_front")
            return right_shoulder_angle, left_shoulder_angle
        except:
            # Fallback se as juntas não existirem
            return 0.0, 0.0
    
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
            t1 = 3.5
            t2 = t1 + 1.5
            t3 = t2 + 1.2
            t4 = t3 + 0.2
            t5 = t4 + 0.5
            t6 = t5 + 1.0
            t7 = t6 + 0.5
            t8 = t7 + 1.0

            hip_right_front = 0  # Positivo para trás
            hip_right_lateral = 0  # Positivo para dentro
            knee_right = 0  # Positivo para dobrar
            ankle_right = 0  # Positivo para baixo
            hip_left_front = 0  # Positivo para trás
            hip_left_lateral = 0  # Positivo para fora
            knee_left = 0  # Positivo para dobrar
            ankle_left = 0  # Positivo para baixo

            if t < t1:
                hip_right_lateral = -0.061
                hip_left_lateral = -0.061

            elif t < t2:
                ankle_right = -0.05
                ankle_left = -0.05

            elif t < t3:
                hip_right_front = -0.5
                knee_right = 0.5
                ankle_right = -0.11
                hip_left_front = -0.1
                ankle_left = -0.01
                hip_right_lateral = -0.09
                hip_left_lateral = 0.10

            action_list = [hip_right_front, hip_right_lateral, knee_right, ankle_right, hip_left_front, hip_left_lateral, knee_left, ankle_left]

        elif num_joints == 10:
            t1 = 1.5
            t2 = t1 + 0.9
            t3 = t2 + 0.5
            t4 = t3 + 2.0
            t5 = t4 + 0.5
            t6 = t5 + 1.0
            t7 = t6 + 0.5
            t8 = t7 + 1.0
            t9 = t8 + 1.0

            hip_right_front = 0  # Positivo para trás
            hip_right_lateral = 0  # Positivo para dentro
            knee_right = 0  # Positivo para dobrar
            ankle_right_front = 0  # Positivo para baixo
            ankle_right_lateral = 0  # Positivo para dentro
            hip_left_front = 0  # Positivo para trás
            hip_left_lateral = 0  # Positivo para fora
            knee_left = 0  # Positivo para dobrar
            ankle_left_front = 0  # Positivo para baixo
            ankle_left_lateral = 0  # Positivo para fora

            if t < t1:
                lateral_inclination = 0.12
                frontal_inclination = 0.05

                hip_right_lateral = -lateral_inclination
                hip_left_lateral = -lateral_inclination
                ankle_right_lateral = lateral_inclination
                ankle_left_lateral = lateral_inclination

                ankle_right_front = -frontal_inclination
                ankle_left_front = -frontal_inclination

            elif t < t2:
                hip_right_front = -0.5  # Negativo para frente
                hip_right_lateral = (
                    0.05  # Positivo para dentro # TODO: Deixa o pé desalinhado com o chão, mas não posso mexer no tornozelo ao mesmo tempo que tiro o pé do chão, para não desequilibrar
                )
                knee_right = 0.5  # Positivo para dobrar
                hip_left_front = -0.1  # Negativo para frente

                lateral_inclination = -0.03
                hip_left_lateral = -lateral_inclination
                ankle_left_lateral = lateral_inclination

            elif t < t3:
                # hip_right_front = 0.5  # Positivo para trás
                knee_right = -0.5  # Negativo para estender
                ankle_right_front = 0.11  # Positivo para baixo
                # hip_left_front = -0.5
                # hip_left_lateral = -0.03  # Negativo para dentro
                # knee_left = 0.5
                # ankle_left_front = -0.13

                lateral_inclination = -0.03
                hip_left_lateral = -lateral_inclination
                ankle_left_lateral = lateral_inclination

            elif t < t4:
                lateral_inclination = -0.06
                hip_right_lateral = -lateral_inclination
                hip_left_lateral = -lateral_inclination
                ankle_right_lateral = lateral_inclination
                ankle_left_lateral = lateral_inclination

            action_list = [hip_right_front, hip_right_lateral, knee_right, ankle_right_front, ankle_right_lateral, hip_left_front, hip_left_lateral, knee_left, ankle_left_front, ankle_left_lateral]

        elif num_joints == 14:
            f = 0.8  # Frequência do movimento
            w = 2 * np.pi * f  # Velocidade angular

            # Ações para pernas (10 juntas)
            hip_right_front = -0.3 * np.sin(w * t + 0.0 * np.pi)
            hip_right_lateral = 0.1 * np.sin(w * t + 0.5 * np.pi)
            knee_right = 0.4 * np.sin(w * t + 0.2 * np.pi)
            ankle_right_front = -0.2 * np.sin(w * t + 0.3 * np.pi)
            ankle_right_lateral = 0.05 * np.sin(w * t + 0.7 * np.pi)

            hip_left_front = -0.3 * np.sin(w * t + 1.0 * np.pi)  # Fase oposta
            hip_left_lateral = 0.1 * np.sin(w * t + 1.5 * np.pi)
            knee_left = 0.4 * np.sin(w * t + 1.2 * np.pi)
            ankle_left_front = -0.2 * np.sin(w * t + 1.3 * np.pi)
            ankle_left_lateral = 0.05 * np.sin(w * t + 1.7 * np.pi)

            # Ações para braços (4 juntas) - apenas ombros
            shoulder_right_front = 0.2 * np.sin(w * t + 0.5 * np.pi)  # Balanço frontal
            shoulder_right_lateral = 0.05  # Pequena abertura lateral fixa
            shoulder_left_front = 0.2 * np.sin(w * t + 1.5 * np.pi)  # Fase oposta
            shoulder_left_lateral = -0.05  # Pequena abertura lateral fixa

            action_list = [
                # Pernas direita (5 juntas)
                hip_right_front,
                hip_right_lateral,
                knee_right,
                ankle_right_front,
                ankle_right_lateral,
                # Pernas esquerda (5 juntas)
                hip_left_front,
                hip_left_lateral,
                knee_left,
                ankle_left_front,
                ankle_left_lateral,
                # Braços (4 juntas)
                shoulder_right_front,
                shoulder_right_lateral,
                shoulder_left_front,
                shoulder_left_lateral,
            ]

        else:
            raise ValueError(f"Número de juntas não suportado para ação de exemplo: {num_joints}")

        # Add noise to the actions
        noise_amplitude = 0.05
        action_list = [a + np.random.uniform(-noise_amplitude, noise_amplitude) for a in action_list]

        return np.array(action_list, dtype=np.float32)
