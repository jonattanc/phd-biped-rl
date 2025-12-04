# robot.py
import utils
import os
import pybullet as p
from xacrodoc import XacroDoc
import numpy as np
import math


class Robot:
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

        self.id = None
        self.revolute_indices = None
        self.gait_state = 0

        self.gait_step_size = 0.2
        self.min_knee_angle = math.radians(2)

        self.initial_section_length = 1
        self.ramp_hypotenuse = 8
        self.ramp_angle_deg = 8.33

        self.ramp_angle_rad = math.radians(self.ramp_angle_deg)
        self.ramp_start = self.initial_section_length / 2
        self.middle_section_length = math.cos(self.ramp_angle_rad) * self.ramp_hypotenuse
        self.ramp_end = self.ramp_start + self.middle_section_length
        self.ramp_height = math.sin(self.ramp_angle_rad) * self.ramp_hypotenuse

        self.robots_dir = os.path.join(utils.PROJECT_ROOT, "robots")
        self.robots_tmp_dir = os.path.join(utils.TMP_PATH, "robots")

        if not os.path.exists(self.robots_tmp_dir):
            os.makedirs(self.robots_tmp_dir, exist_ok=True)

        # Verificar se o arquivo .xacro existe
        xacro_path = os.path.join(self.robots_dir, f"{self.name}.xacro")
        if not os.path.exists(xacro_path):
            raise FileNotFoundError(f"Arquivo {self.name}.xacro não encontrado em {self.robots_dir}")
        
        self.urdf_path = self._generate_urdf()

    def update_env(self, env):
        self.env_name = env

        if self.env_name == "PRA":
            self.ramp_signal = 1

        elif self.env_name == "PRD":
            self.ramp_signal = -1

        else:
            self.ramp_signal = 0

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
        self.gait_state = 0.5

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
        y_position = position[0]
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        x_velocity = linear_velocity[0]
        roll_velocity = angular_velocity[0]
        pitch_velocity = angular_velocity[1]

        joint_positions, joint_velocities = self.get_joint_states()

        obs = np.array([self.gait_state, y_position, roll, pitch, yaw, x_velocity, roll_velocity, pitch_velocity] + joint_positions, dtype=np.float32)
        obs += np.random.normal(0, 1e-6, size=obs.shape)
        return obs

    def update_gait_state(self):
        """Atualiza estado da marcha e retorna se houve transição de estado"""
        right_foot_state = p.getLinkState(self.id, self.get_link_index("right_foot_link"))
        left_foot_state = p.getLinkState(self.id, self.get_link_index("left_foot_link"))
        right_foot_x_position = right_foot_state[0][0]
        left_foot_x_position = left_foot_state[0][0]
        feet_frontal_distance = right_foot_x_position - left_foot_x_position

        right_knee_angle, left_knee_angle = self.get_knee_angles()

        new_state = self.gait_state

        if self.gait_state == 0.5:  # Pés paralelos (pé direito deve avançar)
            if feet_frontal_distance > self.gait_step_size / 2 and abs(right_knee_angle) >= self.min_knee_angle:
                new_state = 0.25

        elif self.gait_state == 0.25:  # Pé direito um pouco avançado
            if feet_frontal_distance > self.gait_step_size:
                new_state = -1.0

        elif self.gait_state == -1.0:  # Pé direito bem avançado
            if feet_frontal_distance < self.gait_step_size / 2 and abs(left_knee_angle) >= self.min_knee_angle:
                new_state = -0.75

        elif self.gait_state == -0.75:  # Pé esquerdo começa a avançar
            if feet_frontal_distance < 0 and abs(left_knee_angle) >= self.min_knee_angle:
                new_state = -0.5

        elif self.gait_state == -0.5:  # Pés paralelos (pé esquerdo deve avançar)
            if feet_frontal_distance < -self.gait_step_size / 2 and abs(left_knee_angle) >= self.min_knee_angle:
                new_state = -0.25

        elif self.gait_state == -0.25:  # Pé esquerdo um pouco avançado
            if feet_frontal_distance < -self.gait_step_size:
                new_state = 1.0

        elif self.gait_state == 1.0:  # Pé esquerdo bem avançado
            if feet_frontal_distance > -self.gait_step_size / 2 and abs(right_knee_angle) >= self.min_knee_angle:
                new_state = 0.75

        elif self.gait_state == 0.75:  # Pé direito começa a avançar
            if feet_frontal_distance > 0 and abs(right_knee_angle) >= self.min_knee_angle:
                new_state = 0.5

        if new_state != self.gait_state:
            self.gait_state = new_state
            return True

        else:
            return False

    def get_imu_position_velocity_orientation(self):
        """Retorna posição, orientação e velocidades linear e angular do IMU"""
        link_state = p.getLinkState(self.id, self.imu_link_index, computeLinkVelocity=1)
        position = link_state[0]
        orientation = p.getEulerFromQuaternion(link_state[1])
        linear_velocity = link_state[6]
        angular_velocity = link_state[7]
        return position, linear_velocity, orientation, angular_velocity

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

    def get_hip_frontal_angles(self):
        """Retorna os ângulos dos quadris direito e esquerdo"""
        return self.get_joint_angle("base_to_right_hip_ball"), self.get_joint_angle("base_to_left_hip_ball")

    def get_hip_lateral_angles(self):
        """Retorna os ângulos laterais dos quadris direito e esquerdo"""
        return self.get_joint_angle("hip_ball_to_right_thigh"), self.get_joint_angle("hip_ball_to_left_thigh")

    def get_foot_contact_states(self):
        """Retorna os estados de contato dos pés direito e esquerdo"""
        right_foot_contacts = p.getContactPoints(bodyA=self.id, linkIndexA=self.get_link_index("right_foot_link"))
        left_foot_contacts = p.getContactPoints(bodyA=self.id, linkIndexA=self.get_link_index("left_foot_link"))
        return (len(right_foot_contacts) > 0), (len(left_foot_contacts) > 0)

    def get_foot_heights(self):
        """Retorna as alturas dos pés direito e esquerdo em relação ao solo"""
        right_foot_state = p.getLinkState(self.id, self.get_link_index("right_foot_link"))
        left_foot_state = p.getLinkState(self.id, self.get_link_index("left_foot_link"))

        right_foot_x = right_foot_state[0][0]
        right_foot_z = right_foot_state[0][2]
        left_foot_x = left_foot_state[0][0]
        left_foot_z = left_foot_state[0][2]

        right_foot_height = self.get_fixed_height(right_foot_z, right_foot_x)
        left_foot_height = self.get_fixed_height(left_foot_z, left_foot_x)

        return right_foot_height, left_foot_height

    def get_foot_x_velocities(self):
        """Retorna as velocidades em x dos pés direito e esquerdo"""
        right_foot_state = p.getLinkState(self.id, self.get_link_index("right_foot_link"), computeLinkVelocity=1)
        left_foot_state = p.getLinkState(self.id, self.get_link_index("left_foot_link"), computeLinkVelocity=1)
        right_foot_x_velocity = right_foot_state[6][0]
        left_foot_x_velocity = left_foot_state[6][0]
        return left_foot_x_velocity, right_foot_x_velocity

    def get_foot_global_angles(self):
        """Retorna os ângulos globais (roll, pitch, yaw) dos pés direito e esquerdo"""
        right_foot_state = p.getLinkState(self.id, self.get_link_index("right_foot_link"))
        left_foot_state = p.getLinkState(self.id, self.get_link_index("left_foot_link"))

        right_foot_orientation = list(p.getEulerFromQuaternion(right_foot_state[1]))
        left_foot_orientation = list(p.getEulerFromQuaternion(left_foot_state[1]))

        right_foot_x = right_foot_state[0][0]
        left_foot_x = left_foot_state[0][0]

        if self.is_in_ramp(right_foot_x):
            right_foot_orientation[1] -= self.ramp_signal * self.ramp_angle_rad

        if self.is_in_ramp(left_foot_x):
            left_foot_orientation[1] -= self.ramp_signal * self.ramp_angle_rad

        return right_foot_orientation, left_foot_orientation

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

    def get_gait_pattern_score(self):
        """
        Calcula um score que indica a qualidade do padrão de marcha.
        Retorna um valor entre 0.0 (ruim) e 1.0 (bom).
        """
        try:
            # Obter estados dos pés
            right_foot_contact, left_foot_contact = self.get_foot_contact_states()

            # Score baseado no padrão alternado (um pé no chão, outro no ar)
            if right_foot_contact != left_foot_contact:
                alternating_score = 1.0  # Padrão alternado ideal
            else:
                if right_foot_contact and left_foot_contact:
                    alternating_score = 0.3  # Ambos no chão - fase de duplo suporte
                else:
                    alternating_score = 0.1  # Ambos no ar - fase de voo (pode ser bom em corrida)

            # Score baseado na diferença de altura dos pés
            right_foot_height, left_foot_height = self.get_foot_heights()
            height_diff = abs(right_foot_height - left_foot_height)

            if height_diff > 0.05:  # Mais de 5cm de diferença
                clearance_score = 1.0  # Bom clearance
            elif height_diff > 0.02:  # 2-5cm de diferença
                clearance_score = 0.7  # Moderado
            else:
                clearance_score = 0.3  # Baixo clearance

            # Combinar scores
            gait_score = alternating_score * 0.6 + clearance_score * 0.4

            return max(0.0, min(1.0, gait_score))

        except Exception as e:
            self.logger.warning(f"Erro ao calcular gait pattern score: {e}")
            return 0.5  # Valor padrão

    def get_energy_used(self):
        """
        Calcula uma estimativa do 'energia' usada baseada nas velocidades das juntas.
        Quanto maior o valor, mais 'energia' foi gasta.
        """
        try:
            joint_positions, joint_velocities = self.get_joint_states()

            # Energia proporcional à soma dos quadrados das velocidades
            # (aproximação simples para esforço)
            energy = sum(v**2 for v in joint_velocities) / len(joint_velocities) if joint_velocities else 0.0

            return energy

        except Exception as e:
            self.logger.warning(f"Erro ao calcular energia: {e}")
            return 1.0  # Valor padrão

    def get_flight_phase_quality(self):
        """
        Calcula a qualidade da fase de voo (para corrida).
        Retorna um valor entre 0.0 (ruim) e 1.0 (bom).
        """
        try:
            right_foot_contact, left_foot_contact = self.get_foot_contact_states()

            # Fase de voo = nenhum pé no chão
            if not right_foot_contact and not left_foot_contact:
                # Verificar altura dos pés durante voo
                right_foot_height, left_foot_height = self.get_foot_heights()
                avg_flight_height = (right_foot_height + left_foot_height) / 2.0

                if avg_flight_height > 0.08:  # Mais de 8cm de altura
                    flight_quality = 1.0
                elif avg_flight_height > 0.04:  # 4-8cm
                    flight_quality = 0.7
                else:
                    flight_quality = 0.3
            else:
                flight_quality = 0.0  # Não está em fase de voo

            return flight_quality

        except Exception as e:
            self.logger.warning(f"Erro ao calcular flight phase quality: {e}")
            return 0.0

    def get_propulsion_efficiency(self):
        """
        Calcula a eficiência propulsiva baseada na velocidade vs esforço.
        """
        try:
            # Obter velocidade atual
            position, linear_velocity, orientation, robot_orientation_velocity = self.get_imu_position_velocity_orientation()
            speed = abs(linear_velocity[0])  # Velocidade em x

            # Obter energia usada
            energy = self.get_energy_used()

            # Eficiência = velocidade / energia (com proteção contra divisão por zero)
            if energy > 0:
                efficiency = speed / (energy + 0.1)  # +0.1 para evitar divisão por zero
            else:
                efficiency = speed

            # Normalizar para 0-1
            normalized_efficiency = min(efficiency / 2.0, 1.0)

            return normalized_efficiency

        except Exception as e:
            self.logger.warning(f"Erro ao calcular propulsion efficiency: {e}")
            return 0.5

    def get_clearance_score(self):
        """
        Score específico para clearance dos pés durante a marcha.
        """
        try:
            right_foot_height, left_foot_height = self.get_foot_heights()

            # Considerar o pé que está mais alto (em swing)
            max_foot_height = max(right_foot_height, left_foot_height)

            if max_foot_height > 0.08:  # Mais de 8cm
                clearance_score = 1.0
            elif max_foot_height > 0.05:  # 5-8cm
                clearance_score = 0.8
            elif max_foot_height > 0.03:  # 3-5cm
                clearance_score = 0.5
            else:
                clearance_score = 0.2

            return clearance_score

        except Exception as e:
            self.logger.warning(f"Erro ao calcular clearance score: {e}")
            return 0.3

    def get_fixed_height(self, z, x):
        if self.env_name == "PRA" or self.env_name == "PRD":
            if x < self.ramp_start:
                ramp_height = 0

            elif x < self.ramp_end:
                ramp_height = -self.ramp_signal * (x - self.ramp_start) * math.tan(self.ramp_angle_rad)

            else:
                ramp_height = -self.ramp_signal * self.ramp_height

            return z + ramp_height

        else:
            return z

    def is_in_ramp(self, x):
        if self.env_name == "PRA" or self.env_name == "PRD":
            if x < self.ramp_start:
                return False

            elif x < self.ramp_end:
                return True

            else:
                return False

        else:
            return False

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

        elif num_joints == 14:  # 6 juntas por perna (3 quadril + 1 joelho + 2 tornozelo) × 2 = 12 + 2 ombros = 14
            # ROBÔ STAGE 5 - Modelo Biomecânico Completo (1.70m, 100kg)
            # Ações mais conservadoras para evitar overflow
            f = 0.5  # Frequência reduzida
            w = 2 * np.pi * f
            
            # PERNA DIREITA (6 juntas) - Amplitudes reduzidas
            hip_right_flexion = -0.2 * np.sin(w * t)  # Flexão/extensão quadril
            hip_right_abduction = 0.08 * np.sin(w * t + 0.5 * np.pi)  # Abdução/adução
            hip_right_rotation = 0.03 * np.sin(w * t + 0.25 * np.pi)  # Rotação quadril
            knee_right = 0.3 * np.sin(w * t + 0.3 * np.pi)  # Flexão joelho
            ankle_right_flexion = -0.15 * np.sin(w * t + 0.6 * np.pi)  # Dorsiflexão/flexão plantar
            ankle_right_inversion = 0.04 * np.sin(w * t + 0.8 * np.pi)  # Inversão/eversão
            
            # PERNA ESQUERDA (6 juntas) - Fase oposta
            hip_left_flexion = -0.2 * np.sin(w * t + np.pi)
            hip_left_abduction = 0.08 * np.sin(w * t + 1.5 * np.pi)
            hip_left_rotation = 0.03 * np.sin(w * t + 1.25 * np.pi)
            knee_left = 0.3 * np.sin(w * t + 1.3 * np.pi)
            ankle_left_flexion = -0.15 * np.sin(w * t + 1.6 * np.pi)
            ankle_left_inversion = 0.04 * np.sin(w * t + 1.8 * np.pi)
            
            # OMBROS (1 junta cada) - Movimento frontal apenas
            shoulder_right = 0.15 * np.sin(w * t + 0.5 * np.pi)  # Balanço frontal
            shoulder_left = 0.15 * np.sin(w * t + 1.5 * np.pi)  # Fase oposta
            
            action_list = [
                # Pernas direita (6 juntas)
                hip_right_flexion, hip_right_abduction, hip_right_rotation,
                knee_right, ankle_right_flexion, ankle_right_inversion,
                # Pernas esquerda (6 juntas)  
                hip_left_flexion, hip_left_abduction, hip_left_rotation,
                knee_left, ankle_left_flexion, ankle_left_inversion,
                # Ombros (2 juntas)
                shoulder_right, shoulder_left
            ]
        
            # Noise reduzido
            noise_amplitude = 0.05 
            action_list = [a + np.random.uniform(-noise_amplitude, noise_amplitude) for a in action_list]

        else:
            raise ValueError(f"Número de juntas não suportado para ação de exemplo: {num_joints}")

        # Add noise to the actions
        noise_amplitude = 0.05
        action_list = [a + np.random.uniform(-noise_amplitude, noise_amplitude) for a in action_list]

        return np.array(action_list, dtype=np.float32)
