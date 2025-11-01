# dpg_manager.py
import numpy as np
import pybullet as p
from dataclasses import dataclass
from typing import Dict, List, Optional
from dpg_gait_phases import GaitPhaseDPG


@dataclass
class DPGConfig:
    """Configuração completa do DPG"""
    enabled: bool = False
    initial_posture: Dict = None
    phase_targets: Dict = None
    phase_weights: Dict = None
    

class DPGManager:
    """
    Gerenciador central para todo o sistema DPG
    """
    
    def __init__(self, logger, robot, reward_system):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.config = DPGConfig()
        
        # Componentes do DPG
        self.phase_detector = None
        self.gait_phase_dpg = None
        
        # Inicializar configurações padrão
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Inicializa configurações padrão do DPG"""
        # Postura inicial otimizada para DPG
        self.config.initial_posture = {
            "hip_frontal": 0.0,
            "hip_lateral": 0.0, 
            "knee": 0.4,
            "ankle_frontal": -0.15,
            "ankle_lateral": 0.0,
            "body_pitch": 0.1
        }
        
        # Metas articulares por fase
        self.config.phase_targets = {
            "IC": {"hip": +0.15, "knee": +0.05, "ankle": +0.02, "sigma": 0.15},
            "LR": {"hip": +0.10, "knee": +0.08, "ankle": +0.02, "sigma": 0.15},
            "MS": {"hip": +0.05, "knee": +0.03, "ankle": +0.05, "sigma": 0.15},
            "TS": {"hip": -0.20, "knee": +0.05, "ankle": -0.30, "sigma": 0.15},
            "PS": {"hip": 0.00, "knee": +0.40, "ankle": -0.10, "sigma": 0.15},
            "ISw": {"hip": +0.25, "knee": +1.00, "ankle": +0.08, "sigma": 0.20},
            "MSw": {"hip": +0.40, "knee": +0.50, "ankle": +0.05, "sigma": 0.20},
            "TSw": {"hip": +0.30, "knee": +0.10, "ankle": +0.02, "sigma": 0.12},
        }
        
        # Pesos das recompensas por fase
        self.config.phase_weights = {
            "velocity": 2.0,
            "phase_angles": 1.0,
            "propulsion": 0.5,
            "clearance": 0.2,
            "stability": 3.0,
            "symmetry": 0.3,
            "effort_torque": 1e-4,
            "effort_power": 1e-5,
            "action_smoothness": 1e-3,
            "lateral_penalty": 1.0,
            "slip_penalty": 0.5,
        }
    
    def enable(self, enabled=True):
        """Ativa/desativa o sistema DPG completo"""
        self.config.enabled = enabled
        
        if enabled:
            self._setup_dpg_components()
            self.logger.info("Sistema DPG ativado com sucesso")
        else:
            self._teardown_dpg_components()
            self.logger.info("Sistema DPG desativado")
    
    def _setup_dpg_components(self):
        """Configura todos os componentes do DPG"""
        try:        
            # 1. Configurar DPG de fases da marcha
            self.gait_phase_dpg = GaitPhaseDPG(self.logger, self.reward_system)
            self.reward_system.gait_phase_dpg = self.gait_phase_dpg

            # 2. Configurar componentes avançados
            self.setup_advanced_dpg_components()

            # 3. Aplicar configuração inicial de fase
            self.gait_phase_dpg._apply_phase_config()

            self.logger.info("Todos os componentes DPG configurados (incluindo avançados)")

        except Exception as e:
            self.logger.error(f"Erro ao configurar componentes DPG: {e}")
            raise
    
    def _teardown_dpg_components(self):
        """Remove todos os componentes do DPG"""
        self.phase_detector = None
        self.gait_phase_dpg = None
        self.reward_system.phase_detector = None
        self.reward_system.gait_phase_dpg = None
    
    def apply_initial_posture(self):
        """Aplica a postura inicial otimizada para DPG"""
        if not self.config.enabled:
            return
        
        try:
            posture = self.config.initial_posture
            
            # Aqui você implementaria a lógica para aplicar a postura inicial
            # ao robô. Isso depende da estrutura específica do seu robô.
            # Exemplo genérico:
            self.logger.info(f"Aplicando postura inicial DPG: {posture}")
            
            # Em uma implementação real, você setaria as juntas do robô
            # para os valores especificados em posture
            
        except Exception as e:
            self.logger.warning(f"Erro ao aplicar postura inicial DPG: {e}")
    
    def calculate_reward(self, sim, action):
        """Calcula recompensa usando o sistema DPG"""
        if not self.config.enabled:
            return 0.0
        
        try:
            return self._calculate_dpg_reward(sim, action)
        except Exception as e:
            self.logger.error(f"Erro no cálculo de recompensa DPG: {e}")
            return 0.0
    
    def _calculate_dpg_reward(self, sim, action):
        """Recompensas mais focadas no aprendizado gradual"""
        total_reward = 0.0

        # BÔNUS MASSIVO PARA FASE INICIAL
        if hasattr(self, 'gait_phase_dpg') and self.gait_phase_dpg:
            current_phase = self.gait_phase_dpg.current_phase

            if current_phase == 0:
                # Bônus generoso por qualquer progresso
                if sim.episode_distance > 0.1:
                    progress_bonus = min(sim.episode_distance * 10, 5.0)
                    total_reward += progress_bonus

                # Bônus por estabilidade básica
                if abs(sim.robot_roll) < 0.5 and abs(sim.robot_pitch) < 0.4:
                    stability_bonus = 2.0
                    total_reward += stability_bonus

            elif current_phase == 1:
                # Bônus por alternância e padrão cruzado
                alternation = sim.robot_left_foot_contact != sim.robot_right_foot_contact
                if alternation:
                    total_reward += 1.0
            total_reward = 0.0
            w = self.config.phase_weights
        
        # Detectar se está preso
        if hasattr(self, 'stagnation_counter'):
            if sim.episode_distance < 0.1:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0

        # RECOMPENSA DE EMERGÊNCIA se estagnado
        if self.stagnation_counter > 30:
            emergency_bonus = 2.0 
            total_reward += emergency_bonus
        
        # 1. Componente de Velocidade (w_v * r_vel)
        velocity_reward = self._calculate_velocity_reward(sim)
        total_reward += w["velocity"] * velocity_reward

        # 2. Componente de Fases e Ângulos Articulares (w_phase * r_ângulos)
        phase_angle_reward = self._calculate_phase_angle_reward(sim)
        total_reward += w["phase_angles"] * phase_angle_reward

        # 3. Componente de Propulsão (w_prop * r_TS) 
        propulsion_reward = self._calculate_propulsion_reward(sim)
        total_reward += w["propulsion"] * propulsion_reward

        # 4. Componente de Clearance (w_clr * r_clr)
        clearance_reward = self._calculate_clearance_reward(sim)
        total_reward += w["clearance"] * clearance_reward

        # 5. Componente de Estabilidade (w_stab * r_MoS)
        stability_reward = self._calculate_stability_reward(sim)
        total_reward += w["stability"] * stability_reward

        # 6. Componente de Simetria (w_sym * r_simetria)
        symmetry_reward = self._calculate_symmetry_reward(sim)
        total_reward += w["symmetry"] * symmetry_reward

        # 7. Componente de Coordenação Braço-Perna
        coordination_reward = self._calculate_arm_leg_coordination(sim)
        total_reward += 0.4 * coordination_reward

        # 8. NOVO: Recompensa por Fase de Voo Controlada
        flight_reward = self._calculate_flight_phase_reward(sim)
        total_reward += 0.2 * flight_reward  # Peso leve para voo

        # 9. Penalidades de Eficiência
        effort_cost = self._calculate_effort_cost(sim, action)
        total_reward -= effort_cost

        # 10. Penalidades de Estabilidade Lateral
        lateral_cost = self._calculate_lateral_cost(sim)
        total_reward -= lateral_cost

        # 11. Penalidades por Problemas de Contato
        foot_slap_penalty = self._calculate_anti_foot_slap_penalty(sim)
        total_reward -= 0.8 * foot_slap_penalty  # Peso forte contra foot-slap

        slip_penalty = self._calculate_slip_penalty(sim)
        total_reward -= 0.6 * slip_penalty  # Peso moderado contra escorregamento

        # 12. Penalidades fora da marcha
        if hasattr(sim, 'episode_termination'):  
            if sim.episode_termination == "fell":  
                total_reward -= 50.0
            elif sim.episode_termination == "yaw_deviated":  
                total_reward -= 30.0  

        if hasattr(sim, 'has_gait_state_changed') and getattr(sim, 'has_gait_state_changed', False):
            total_reward += 25.0

        # BÔNUS ADICIONAL PARA FASE INICIAL
        if hasattr(self, 'gait_phase_dpg') and self.gait_phase_dpg and self.gait_phase_dpg.current_phase == 0:
            # Bônus por manter estabilidade (evitar quedas)
            stability_bonus = 0.0
            if abs(sim.robot_roll) < 0.3:  # Roll estável
                stability_bonus += 0.5
            if abs(sim.robot_pitch) < 0.3:  # Pitch estável  
                stability_bonus += 0.5
            if sim.robot_z_position > 0.6:  # Não caiu
                stability_bonus += 1.0

            total_reward += stability_bonus * 0.5  # Peso moderado

        return total_reward
    
    def _calculate_velocity_reward(self, sim):
        """Recompensa de velocidade adaptada para DPG"""
        vx = getattr(sim, "robot_x_velocity", 0)
        
        if self.gait_phase_dpg and self.gait_phase_dpg.current_phase == 0:
            v_min, v_max = 0.1, 1.0
            gamma = 1.0
        else:
            v_min, v_max = 1.2, 2.8
            gamma = 1.4
        
        if v_max - v_min > 0:
            normalized_vel = (vx - v_min) / (v_max - v_min)
            clipped_vel = np.clip(normalized_vel, 0.0, 1.0)
            return clipped_vel ** gamma
        else:
            return 0.0
    
    def _calculate_phase_angle_reward(self, sim):
        """Recompensa por seguir metas articulares da fase atual"""
        if not self.phase_detector:
            return 0.0
            
        total_angle_reward = 0.0
        current_time = sim.episode_steps * sim.time_step_s
        
        for foot_side in ["left", "right"]:
            phase, _ = self.phase_detector.detect_phase_transition(foot_side, current_time)
            phase_name = phase.name
            
            if phase_name in self.config.phase_targets:
                target = self.config.phase_targets[phase_name]
                
                # Obter ângulos atuais
                hip_angle = self._get_hip_angle(sim, foot_side)
                knee_angle = self._get_knee_angle(sim, foot_side)
                ankle_angle = self._get_ankle_angle(sim, foot_side)
                
                # Recompensas gaussianas
                hip_reward = np.exp(-0.5 * (hip_angle - target["hip"])**2 / target["sigma"]**2)
                knee_reward = np.exp(-0.5 * (knee_angle - target["knee"])**2 / target["sigma"]**2)
                ankle_reward = np.exp(-0.5 * (ankle_angle - target["ankle"])**2 / target["sigma"]**2)
                
                phase_reward = (hip_reward + knee_reward + ankle_reward) / 3.0
                total_angle_reward += phase_reward
        
        return total_angle_reward / 2.0
    
    def _calculate_propulsion_reward(self, sim):
        """Recompensa de propulsão na fase TS"""
        if not self.phase_detector:
            return 0.0
            
        propulsion_reward = 0.0
        current_time = sim.episode_steps * sim.time_step_s
        
        for foot_side in ["left", "right"]:
            phase, in_contact = self.phase_detector.detect_phase_transition(foot_side, current_time)
            
            if phase.name == "TS" and in_contact:
                ankle_velocity = self._get_ankle_velocity(sim, foot_side)
                propulsion = max(ankle_velocity * 0.1, 0.0)
                propulsion_reward += np.tanh(0.002 * propulsion)
        
        return propulsion_reward
    
    def _calculate_clearance_reward(self, sim):
        """Recompensa por clearance adequado na oscilação"""
        if not self.phase_detector:
            return 0.0
            
        clearance_reward = 0.0
        current_time = sim.episode_steps * sim.time_step_s
        
        for foot_side in ["left", "right"]:
            phase, in_contact = self.phase_detector.detect_phase_transition(foot_side, current_time)
            
            if not in_contact:
                foot_height = getattr(sim, f"robot_{foot_side}_foot_height")
                clearance = 1.0 / (1.0 + np.exp(-100 * (foot_height - 0.025)))
                clearance_reward += clearance
        
        return clearance_reward
    
    def _calculate_stability_reward(self, sim):
        """Recompensa de estabilidade"""
        pitch, roll = getattr(sim, "robot_pitch", 0), getattr(sim, "robot_roll", 0)
        stability_penalty = abs(pitch) + abs(roll)
        return np.exp(-stability_penalty / 0.35)
    
    def _calculate_symmetry_reward(self, sim):
        """Recompensa por simetria temporal entre membros"""
        left_hip = abs(getattr(sim, "robot_left_hip_frontal_angle", 0))
        right_hip = abs(getattr(sim, "robot_right_hip_frontal_angle", 0))
        
        if left_hip + right_hip == 0:
            return 1.0
            
        symmetry = 1.0 - abs(left_hip - right_hip) / (left_hip + right_hip)
        return symmetry
    
    def _calculate_effort_cost(self, sim, action):
        """Custo de esforço"""
        w = self.config.phase_weights
        
        joint_velocities = getattr(sim, "joint_velocities", [0])
        torque_cost = w["effort_torque"] * sum(v**2 for v in joint_velocities)
        
        power_cost = w["effort_power"] * sum(max(0, v)**2 for v in joint_velocities)
        
        last_action = getattr(sim, "episode_last_action", np.zeros_like(action))
        action_smoothness_cost = w["action_smoothness"] * np.sum((action - last_action)**2)
        
        return torque_cost + power_cost + action_smoothness_cost
    
    def _calculate_lateral_cost(self, sim):
        """Penalidade por deriva lateral"""
        w = self.config.phase_weights
        vy = abs(getattr(sim, "robot_y_velocity", 0))
        return w["lateral_penalty"] * vy
    
    # Métodos auxiliares para obter ângulos articulares
    def _get_hip_angle(self, sim, foot_side):
        try:
            if foot_side == "right":
                return sim.robot_right_hip_frontal_angle
            else:
                return sim.robot_left_hip_frontal_angle
        except:
            return 0.0
    
    def _get_knee_angle(self, sim, foot_side):
        try:
            if foot_side == "right":
                return sim.robot_right_knee_angle
            else:
                return sim.robot_left_knee_angle
        except:
            return 0.0
    
    def _get_ankle_angle(self, sim, foot_side):
        try:
            if self.phase_detector:
                return self.phase_detector._get_ankle_angle(foot_side)
            return 0.0
        except:
            return 0.0
    
    def _get_ankle_velocity(self, sim, foot_side):
        try:
            if self.phase_detector:
                current_time = sim.episode_steps * sim.time_step_s
                return self.phase_detector.get_ankle_velocity(foot_side, current_time)
            return 0.0
        except:
            return 0.0
    
    def _calculate_arm_leg_coordination(self, sim):
        """Calcula recompensa por coordenação braço-perna"""
        try:
            left_arm_angle = getattr(sim, "robot_left_shoulder_front_angle", 0)
            right_arm_angle = getattr(sim, "robot_right_shoulder_front_angle", 0)
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)

            coordination = 0.0

            # Coordenação contralateral: braço esquerdo com pé direito e vice-versa
            if not right_contact and left_arm_angle > 0.1:  # Braço esquerdo avança quando pé direito está no ar
                coordination += 0.5
            if not left_contact and right_arm_angle > 0.1:  # Braço direito avança quando pé esquerdo está no ar
                coordination += 0.5

            return coordination
        except:
            return 0.0

    def _calculate_flight_phase_reward(self, sim):
        """Recompensa por fase de voo controlada"""
        if not self.phase_detector:
            return 0.0

        flight_duration = self.phase_detector.get_flight_duration(sim.episode_steps * sim.time_step_s)

        # Recompensa por fase de voo curta e controlada (ideal para corrida)
        if 0.05 < flight_duration < 0.2:  # 50-200ms é ideal
            return 1.0
        elif flight_duration > 0:  # Qualquer fase de voo é melhor que nenhuma
            return 0.5 * min(flight_duration / 0.3, 1.0)  # Normalizado
        return 0.0

    def _calculate_anti_foot_slap_penalty(self, sim):
        """Penalidade por foot-slap (plantarflexão rápida no contato inicial)"""
        if not self.phase_detector:
            return 0.0

        current_time = sim.episode_steps * sim.time_step_s
        foot_slap_intensity = 0.0

        for foot_side in ["left", "right"]:
            slap = self.phase_detector.detect_foot_slap(foot_side, current_time)
            foot_slap_intensity += slap

        return foot_slap_intensity

    def _calculate_slip_penalty(self, sim):
        """Penalidade por escorregamento dos pés"""
        if not self.phase_detector:
            return 0.0

        current_time = sim.episode_steps * sim.time_step_s
        slip_intensity = 0.0

        for foot_side in ["left", "right"]:
            slip = self.phase_detector.detect_foot_slip(foot_side, current_time)
            slip_intensity += slip

        return slip_intensity

    def _calculate_alternating_score(self, sim):
        """Calcula score de alternância baseado no DPG"""
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)

        alternation = left_contact != right_contact
        return 1.0 if alternation else 0.0

    def _calculate_gait_coordination(self, sim):
        """Calcula coordenação baseado no padrão cruzado do DPG"""
        coordination_score = self._calculate_arm_leg_coordination(sim)
        return coordination_score  

    def _calculate_consistency_score(self, sim):
        """Calcula consistência baseada no histórico do DPG"""
        if not hasattr(self, 'recent_distances'):
            self.recent_distances = []

        self.recent_distances.append(sim.episode_distance)
        if len(self.recent_distances) > 10:
            self.recent_distances.pop(0)

        if len(self.recent_distances) < 3:
            return 0.5

        consistency = 1.0 - (np.std(self.recent_distances) / 2.0)
        return max(0.0, consistency)

    def update_phase_progression(self, episode_results):
        """Atualiza progressão de fases do DPG"""
        if self.gait_phase_dpg:
            self.gait_phase_dpg.update_phase(episode_results)
    
    def get_status(self):
        """Retorna status atual do DPG"""
        if not self.config.enabled:
            return {"enabled": False}
        
        status = {
            "enabled": True,
            "phase_detector": self.phase_detector is not None,
            "gait_phase_dpg": self.gait_phase_dpg is not None,
        }
        
        if self.gait_phase_dpg:
            status.update(self.gait_phase_dpg.get_status())
        
        return status
    
    def setup_advanced_dpg_components(self):
        """Configura componentes avançados do DPG"""
        if not self.config.enabled or not self.gait_phase_dpg:
            return

        try:
            # Inicializar componentes avançados
            self.gait_phase_dpg._initialize_adaptive_reward_components()
            self.logger.info("Componentes avançados DPG configurados (HDPG, IRL, DASS)")
        except Exception as e:
            self.logger.error(f"Erro ao configurar componentes avançados DPG: {e}")

    def get_advanced_metrics(self):
        """Retorna métricas avançadas do sistema"""
        if not self.config.enabled or not self.gait_phase_dpg:
            return {}
        
        metrics = {
            "dpg_phase": self.gait_phase_dpg.current_phase,
            "phase_name": self.gait_phase_dpg.phases[self.gait_phase_dpg.current_phase].name,
            "dass_samples": len(self.gait_phase_dpg.dass_samples),
            "hdpg_active": self.gait_phase_dpg.current_phase >= 3,
            "irl_confidence": 0.0,
            "hdpg_convergence": 0.0
        }
        
        if hasattr(self.gait_phase_dpg, 'learned_reward_model') and self.gait_phase_dpg.learned_reward_model:
            metrics["irl_confidence"] = self.gait_phase_dpg.learned_reward_model.get('confidence', 0.0)
        
        if self.gait_phase_dpg.current_phase >= 3:
            metrics["hdpg_convergence"] = self.gait_phase_dpg._validate_hdpg_convergence()
        
        return metrics

