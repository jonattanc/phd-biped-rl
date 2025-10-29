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
        self.force_contact_threshold = 50.0  # N
        self.force_release_threshold = 20.0  # N
        self.swing_velocity_threshold = 0.1  # m/s
        
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
            
            # Considerar contato se houver qualquer ponto de contato
            return len(contact_points) > 0
            
        except Exception as e:
            self.logger.warning(f"Erro ao verificar contato do pé {foot_side}: {e}")
            # Fallback para o método existente do robô
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
    
    def get_current_phases(self):
        return self.phase_state.copy()