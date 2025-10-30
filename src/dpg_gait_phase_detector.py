# gait_phases_detector.py
import numpy as np
from enum import Enum
import pybullet as p

class GaitPhase(Enum):
    IC = "Initial Contact"
    LR = "Loading Response"
    MS = "Mid Stance" 
    TS = "Terminal Stance"
    PS = "Pre-Swing"
    ISw = "Initial Swing"
    MSw = "Mid Swing"
    TSw = "Terminal Swing"

class GaitPhaseDetector:
    def __init__(self, robot, logger):
        self.robot = robot
        self.logger = logger
        
        # Configurações baseadas no documento
        self.force_contact_threshold = 30.0  # N
        self.force_release_threshold = 10.0  # N
        self.swing_velocity_threshold = 0.05  # m/s
        
        # Estados
        self.phase_state = {"left": GaitPhase.IC, "right": GaitPhase.IC}
        self.contact_history = {"left": [], "right": []}
        self.phase_start_time = {"left": 0.0, "right": 0.0}
        self.last_contact_state = {"left": False, "right": False}
        
        # Cache para velocidades
        self.last_foot_positions = {"left": None, "right": None}
        self.last_ankle_angles = {"left": 0.0, "right": 0.0}
        self.last_knee_angles = {"left": 0.0, "right": 0.0}
    
    def detect_phase_transition(self, foot_side, current_time):
        """Detecta a fase atual do pé"""
        try:
            # Verificar contato atual
            in_contact = self._check_foot_contact(foot_side)
            foot_velocity = self._get_foot_velocity(foot_side, current_time)
            
            previous_phase = self.phase_state[foot_side]
            new_phase = previous_phase
            
            # Lógica de transição baseada no documento
            if in_contact:
                if previous_phase in [GaitPhase.TSw, GaitPhase.ISw]:
                    new_phase = GaitPhase.IC
                elif previous_phase == GaitPhase.IC and current_time - self.phase_start_time[foot_side] > 0.05:
                    new_phase = GaitPhase.LR
                elif previous_phase == GaitPhase.LR and self._get_force_ratio(foot_side) > 0.4:
                    new_phase = GaitPhase.MS
                elif previous_phase == GaitPhase.MS and self._get_force_ratio(foot_side) > 0.2:
                    new_phase = GaitPhase.TS
                elif previous_phase == GaitPhase.TS and self._get_force_ratio(foot_side) < 0.1:
                    new_phase = GaitPhase.PS
            else:
                # Fase de oscilação
                if previous_phase in [GaitPhase.PS, GaitPhase.TS] and foot_velocity > self.swing_velocity_threshold:
                    new_phase = GaitPhase.ISw
                elif previous_phase == GaitPhase.ISw and self._get_foot_height(foot_side) > 0.05:
                    new_phase = GaitPhase.MSw
                elif previous_phase == GaitPhase.MSw and foot_velocity < -self.swing_velocity_threshold:
                    new_phase = GaitPhase.TSw
            
            # Atualizar se houve mudança
            if new_phase != previous_phase:
                self.phase_state[foot_side] = new_phase
                self.phase_start_time[foot_side] = current_time
                self.logger.debug(f"Fase {foot_side}: {previous_phase.name} -> {new_phase.name}")
            
            self.last_contact_state[foot_side] = in_contact
            return new_phase, in_contact
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de fase {foot_side}: {e}")
            return self.phase_state[foot_side], self.last_contact_state[foot_side]
    
    def _check_foot_contact(self, foot_side):
        """Verifica se o pé está em contato com o solo usando PyBullet"""
        try:
            foot_link_name = f"{foot_side}_foot_link"
            foot_link_index = self.robot.get_link_index(foot_link_name)
            contact_points = p.getContactPoints(bodyA=self.robot.id, linkIndexA=foot_link_index)
            total_force = 0.0
            for contact in contact_points:
                total_force += contact[9]  

            return total_force > self.force_contact_threshold  

        except Exception as e:
            self.logger.warning(f"Erro ao verificar contato: {e}")
            # Fallback
            if foot_side == "right":
                return self.robot.robot_right_foot_contact
            else:
                return self.robot.robot_left_foot_contact

    def _get_foot_velocity(self, foot_side, current_time):
        """Calcula velocidade vertical do pé usando diferença de posição"""
        try:
            foot_link_name = f"{foot_side}_foot_link"
            foot_link_index = self.robot.get_link_index(foot_link_name)
            
            # Obter estado atual do link
            link_state = p.getLinkState(self.robot.id, foot_link_index, computeLinkVelocity=True)
            current_position = np.array(link_state[0])  # Posição mundial
            current_velocity = np.array(link_state[6])  # Velocidade linear
            
            # Retornar componente vertical da velocidade
            return current_velocity[2]  # vz
            
        except Exception as e:
            self.logger.warning(f"Erro ao obter velocidade do pé {foot_side}: {e}")
            return 0.0
    
    def _get_foot_height(self, foot_side):
        """Obtém altura do pé em relação ao solo"""
        try:
            if foot_side == "right":
                return self.robot.robot_right_foot_height
            else:
                return self.robot.robot_left_foot_height
        except:
            # Fallback usando PyBullet diretamente
            try:
                foot_link_name = f"{foot_side}_foot_link"
                foot_link_index = self.robot.get_link_index(foot_link_name)
                link_state = p.getLinkState(self.robot.id, foot_link_index)
                return link_state[0][2]  # Posição Z
            except:
                return 0.0
    
    def _get_force_ratio(self, foot_side):
        """Estima ratio de força normal/peso (simplificado)"""
        try:
            foot_link_name = f"{foot_side}_foot_link"
            foot_link_index = self.robot.get_link_index(foot_link_name)
            
            contact_points = p.getContactPoints(bodyA=self.robot.id, linkIndexA=foot_link_index)
            total_force = 0.0
            
            for contact in contact_points:
                # contact[9] é a força normal no PyBullet
                total_force += contact[9]
            
            # Assumir peso total de ~500N para um robô humanoide
            estimated_weight = 500.0
            return min(total_force / estimated_weight, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular força do pé {foot_side}: {e}")
            # Fallback baseado no contato
            return 1.0 if self._check_foot_contact(foot_side) else 0.0
    
    def get_ankle_velocity(self, foot_side, current_time, dt=0.01):
        """Calcula velocidade angular do tornozelo"""
        try:
            current_angle = self._get_ankle_angle(foot_side)
            
            # Calcular velocidade por diferença
            if hasattr(self, f'last_ankle_time_{foot_side}'):
                last_time = getattr(self, f'last_ankle_time_{foot_side}')
                last_angle = self.last_ankle_angles[foot_side]
                
                if current_time - last_time > 0:
                    velocity = (current_angle - last_angle) / (current_time - last_time)
                else:
                    velocity = 0.0
            else:
                velocity = 0.0
            
            # Atualizar cache
            self.last_ankle_angles[foot_side] = current_angle
            setattr(self, f'last_ankle_time_{foot_side}', current_time)
            
            return velocity
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular velocidade do tornozelo {foot_side}: {e}")
            return 0.0
    
    def get_knee_velocity(self, foot_side, current_time):
        """Calcula velocidade angular do joelho"""
        try:
            current_angle = self._get_knee_angle(foot_side)
            
            # Calcular velocidade por diferença
            if hasattr(self, f'last_knee_time_{foot_side}'):
                last_time = getattr(self, f'last_knee_time_{foot_side}')
                last_angle = self.last_knee_angles[foot_side]
                
                if current_time - last_time > 0:
                    velocity = (current_angle - last_angle) / (current_time - last_time)
                else:
                    velocity = 0.0
            else:
                velocity = 0.0
            
            # Atualizar cache
            self.last_knee_angles[foot_side] = current_angle
            setattr(self, f'last_knee_time_{foot_side}', current_time)
            
            return velocity
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular velocidade do joelho {foot_side}: {e}")
            return 0.0
    
    def _get_ankle_angle(self, foot_side):
        """Obtém ângulo do tornozelo baseado na estrutura do seu robô"""
        try:
            # Baseado no seu robot_stage5.xacro, o tornozelo tem juntas frontais e laterais
            if foot_side == "right":
                # Para robô stage5, usamos a junta frontal do tornozelo como referência principal
                ankle_front_angle = self.robot.get_joint_angle("right_shin_to_ankle_ball")
                return ankle_front_angle
            else:
                ankle_front_angle = self.robot.get_joint_angle("left_shin_to_ankle_ball")
                return ankle_front_angle
                
        except Exception as e:
            self.logger.warning(f"Erro ao obter ângulo do tornozelo {foot_side}: {e}")
            return 0.0
    
    def _get_knee_angle(self, foot_side):
        """Obtém ângulo do joelho"""
        try:
            if foot_side == "right":
                return self.robot.robot_right_knee_angle
            else:
                return self.robot.robot_left_knee_angle
        except:
            try:
                if foot_side == "right":
                    return self.robot.get_joint_angle("right_knee_ball_to_shin")
                else:
                    return self.robot.get_joint_angle("left_knee_ball_to_shin")
            except:
                return 0.0
            
    def get_grf_forces(self, foot_side):
        """Obtém as forças de reação do solo (GRF) para um pé específico"""
        try:
            foot_link_name = f"{foot_side}_foot_link"
            foot_link_index = self.robot.get_link_index(foot_link_name)

            contact_points = p.getContactPoints(bodyA=self.robot.id, linkIndexA=foot_link_index)

            total_force = np.zeros(3)  # [Fx, Fy, Fz]
            total_torque = np.zeros(3)  # [Tx, Ty, Tz]

            for contact in contact_points:
                # contact[9] = força normal, contact[10-11] = forças de atrito
                # contact[12-14] = torque de atrito
                normal_force = contact[9]
                friction_force1 = contact[10]
                friction_force2 = contact[11]

                # Direção da força normal (do corpo B para A)
                normal_direction = contact[7]

                # Calcular componentes da força
                Fz = normal_force * abs(normal_direction[2])  # Componente vertical
                Fx = friction_force1
                Fy = friction_force2

                total_force[0] += Fx
                total_force[1] += Fy
                total_force[2] += Fz

            return total_force

        except Exception as e:
            self.logger.warning(f"Erro ao calcular GRF para {foot_side}: {e}")
            return np.zeros(3)

    def get_propulsion_impulse(self, foot_side, current_time, dt=0.01):
        """Calcula o impulso de propulsão horizontal na fase TS"""
        try:
            if not hasattr(self, 'last_grf_time'):
                self.last_grf_time = current_time
                self.last_grf_forces = {"left": np.zeros(3), "right": np.zeros(3)}
                self.propulsion_integrals = {"left": 0.0, "right": 0.0}
                return 0.0

            # Só calcular durante a fase TS
            current_phase = self.phase_state[foot_side]
            if current_phase != GaitPhase.TS:
                self.propulsion_integrals[foot_side] = 0.0  # Resetar integral
                return 0.0

            # Obter forças atuais
            current_grf = self.get_grf_forces(foot_side)
            Fx = current_grf[0]  # Força horizontal anterior-posterior

            # Integrar impulso (F * dt)
            time_diff = current_time - self.last_grf_time
            if time_diff > 0:
                impulse = Fx * time_diff
                # Só considerar forças propulsivas (positivas)
                if impulse > 0:
                    self.propulsion_integrals[foot_side] += impulse

            self.last_grf_time = current_time
            self.last_grf_forces[foot_side] = current_grf

            return self.propulsion_integrals[foot_side]

        except Exception as e:
            self.logger.warning(f"Erro ao calcular impulso de propulsão {foot_side}: {e}")
            return 0.0
        
    def detect_flight_phase(self):
        """Detecta quando ambos os pés estão sem contato (fase de voo)"""
        try:
            left_contact = self._check_foot_contact("left")
            right_contact = self._check_foot_contact("right")

            # Fase de voo = nenhum pé em contato
            flight_phase = not left_contact and not right_contact

            # Atualizar histórico de voo
            if not hasattr(self, 'flight_history'):
                self.flight_history = []

            self.flight_history.append(flight_phase)
            if len(self.flight_history) > 10:  # Manter apenas últimos 10 frames
                self.flight_history.pop(0)

            # Considerar voo estável se ocorrer por vários frames consecutivos
            stable_flight = sum(self.flight_history) >= 3  # Pelo menos 3 frames de voo

            return stable_flight

        except Exception as e:
            self.logger.warning(f"Erro ao detectar fase de voo: {e}")
            return False

    def get_flight_duration(self, current_time):
        """Calcula duração da fase de voo atual"""
        try:
            if not hasattr(self, 'flight_start_time'):
                self.flight_start_time = None
                self.last_flight_state = False

            current_flight = self.detect_flight_phase()

            # Iniciar temporizador quando começar o voo
            if current_flight and not self.last_flight_state:
                self.flight_start_time = current_time
            # Parar temporizador quando voo terminar
            elif not current_flight and self.last_flight_state:
                self.flight_start_time = None

            self.last_flight_state = current_flight

            # Retornar duração atual do voo
            if current_flight and self.flight_start_time is not None:
                return current_time - self.flight_start_time

            return 0.0

        except Exception as e:
            self.logger.warning(f"Erro ao calcular duração do voo: {e}")
            return 0.0
    
    def detect_foot_slap(self, foot_side, current_time):
        """Detecta 'foot-slap' - plantarflexão rápida após contato inicial"""
        try:
            # Só verificar nas primeiras fases de contato
            current_phase = self.phase_state[foot_side]
            if current_phase not in [GaitPhase.IC, GaitPhase.LR]:
                return 0.0

            # Verificar se é um contato recente (últimos 80ms)
            if not hasattr(self, 'contact_start_times'):
                self.contact_start_times = {"left": 0.0, "right": 0.0}

            # Registrar início do contato
            in_contact = self._check_foot_contact(foot_side)
            if in_contact and self.contact_start_times[foot_side] == 0:
                self.contact_start_times[foot_side] = current_time

            contact_duration = current_time - self.contact_start_times[foot_side]

            # Só verificar nos primeiros 80ms após contato
            if contact_duration > 0.08:  # 80ms
                return 0.0

            # Calcular velocidade angular do tornozelo
            ankle_velocity = self.get_ankle_velocity(foot_side, current_time)

            # Foot-slap = velocidade negativa rápida (plantarflexão)
            if ankle_velocity < -2.0:  # rad/s threshold
                slap_intensity = min(abs(ankle_velocity) / 5.0, 1.0)  # Normalizar 0-1
                return slap_intensity

            return 0.0

        except Exception as e:
            self.logger.warning(f"Erro ao detectar foot-slap {foot_side}: {e}")
            return 0.0

    def get_ankle_plantarflexion_velocity(self, foot_side, current_time):
        """Velocidade específica de plantarflexão (negativa)"""
        try:
            ankle_velocity = self.get_ankle_velocity(foot_side, current_time)
            # Retornar apenas velocidades negativas (plantarflexão)
            return min(ankle_velocity, 0.0)
        except:
            return 0.0
    
    def detect_foot_slip(self, foot_side, current_time):
        """Detecta escorregamento do pé baseado em velocidade tangencial"""
        try:
            # Só verificar durante fases de contato
            if not self._check_foot_contact(foot_side):
                return 0.0
                
            # Obter velocidade linear do pé
            foot_link_name = f"{foot_side}_foot_link"
            foot_link_index = self.robot.get_link_index(foot_link_name)
            
            link_state = p.getLinkState(self.robot.id, foot_link_index, computeLinkVelocity=True)
            linear_velocity = np.array(link_state[6])  # Velocidade linear [vx, vy, vz]
            
            # Calcular velocidade tangencial (horizontal)
            tangential_velocity = np.sqrt(linear_velocity[0]**2 + linear_velocity[1]**2)
            
            # Threshold de escorregamento: > 2cm/s (0.02 m/s)
            slip_threshold = 0.02
            
            if tangential_velocity > slip_threshold:
                # Intensidade normalizada do escorregamento
                slip_intensity = min((tangential_velocity - slip_threshold) / 0.1, 1.0)
                return slip_intensity
                
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Erro ao detectar escorregamento {foot_side}: {e}")
            return 0.0
    
    def get_foot_tangential_velocity(self, foot_side):
        """Retorna velocidade tangencial do pé para debug"""
        try:
            foot_link_name = f"{foot_side}_foot_link"
            foot_link_index = self.robot.get_link_index(foot_link_name)
            
            link_state = p.getLinkState(self.robot.id, foot_link_index, computeLinkVelocity=True)
            linear_velocity = np.array(link_state[6])
            
            return np.sqrt(linear_velocity[0]**2 + linear_velocity[1]**2)
        except:
            return 0.0
        
    def get_current_phases(self):
        return self.phase_state.copy()