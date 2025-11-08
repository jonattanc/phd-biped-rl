# dpg_reward.py
import time
import numpy as np
from typing import Any, Dict, List, Callable, Tuple
from dataclasses import dataclass


@dataclass
class RewardComponent:
    name: str
    weight: float
    calculator: Callable
    enabled: bool = True
    adaptive_weight: float = 1.0


class RewardCalculator:
    """
    ESPECIALISTA EM RECOMPENSAS
    """
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.components = self._initialize_components()
    
    def _initialize_components(self) -> Dict[str, RewardComponent]:
        """Inicializa componentes de recompensa"""
        return {
            "stability": RewardComponent("stability", 3.0, self._calculate_stability_reward),
            "basic_progress": RewardComponent("basic_progress", 5.0, self._calculate_basic_progress_reward),
            "posture": RewardComponent("posture", 2.5, self._calculate_posture_reward),
            "velocity": RewardComponent("velocity", 1.5, self._calculate_velocity_reward),
            "phase_angles": RewardComponent("phase_angles", 1.0, self._calculate_phase_angles_reward),
            "propulsion": RewardComponent("propulsion", 1.0, self._calculate_propulsion_reward),
            "clearance": RewardComponent("clearance", 0.8, self._calculate_clearance_reward),
            "coordination": RewardComponent("coordination", 1.2, self._calculate_coordination_reward),
            "efficiency": RewardComponent("efficiency", 0.8, self._calculate_efficiency_reward),
            "success_bonus": RewardComponent("success_bonus", 5.0, self._calculate_success_bonus),
            "effort_penalty": RewardComponent("effort_penalty", 0.005, self._calculate_effort_penalty),
            "dynamic_balance": RewardComponent("dynamic_balance", 1.5, self._calculate_dynamic_balance_reward),
            "smoothness": RewardComponent("smoothness", 1.0, self._calculate_smoothness_reward),
            "rhythm": RewardComponent("rhythm", 1.2, self._calculate_rhythm_reward),
            "gait_pattern": RewardComponent("gait_pattern", 1.0, self._calculate_gait_pattern_reward),
            "biomechanics": RewardComponent("biomechanics", 0.8, self._calculate_biomechanics_reward),
            "robustness": RewardComponent("robustness", 0.5, self._calculate_robustness_reward),
            "adaptation": RewardComponent("adaptation", 0.5, self._calculate_adaptation_reward),
            "recovery": RewardComponent("recovery", 0.8, self._calculate_recovery_reward),
            "movement_priority": RewardComponent("movement_priority", 10.0, self._calculate_movement_priority_reward),
        }
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
        """RECOMPENSA FOCADA EM PROGRESSO CONSISTENTE"""
        base_reward = 0.0
        distance = max(getattr(sim, "episode_distance", 0), 0)
        
        # BÔNUS PRIMÁRIO: DISTÂNCIA (mais progressivo)
        if distance > 0:
            distance_reward = distance * 200.0  # Reduzido de 500.0
            
            # BÔNUS PROGRESSIVO MAIS REALISTA
            if distance > 2.0: base_reward += 800.0
            elif distance > 1.5: base_reward += 400.0
            elif distance > 1.0: base_reward += 200.0
            elif distance > 0.7: base_reward += 100.0
            elif distance > 0.5: base_reward += 50.0
            elif distance > 0.3: base_reward += 20.0
            elif distance > 0.1: base_reward += 10.0
    
            base_reward += distance_reward
    
        # BÔNUS DE SEQUÊNCIA DA MARCHA (reduzido)
        sequence_bonus = self.calculate_marcha_sequence_bonus(sim, action)
        base_reward += sequence_bonus * 0.5  # Antes: bonus completo
    
        # BÔNUS DE SOBREVIVÊNCIA (reduzido)
        if not getattr(sim, "episode_terminated", True) and distance > 0.01:
            base_reward += 100.0  # Antes: 500.0
    
        return max(base_reward, 0.0)
    
    def _calculate_global_penalties(self, sim, action) -> float:
        """Calcula penalidades globais mais balanceadas"""
        penalties = 0.0
        progress = getattr(sim, "learning_progress", 0.0)
        penalty_multiplier = 1.0 - min(progress * 0.7, 0.6)

        # 1. Penalidade de ação extrema 
        if hasattr(action, '__len__'):
            action_magnitude = np.sqrt(np.sum(np.square(action)))
            action_penalty = action_magnitude * 0.005 * penalty_multiplier  
            penalties += min(action_penalty, 0.3)  

        # 2. Penalidade por queda iminente 
        height = getattr(sim, "robot_z_position", 0.8)
        if height < 0.5:
            fall_penalty = (0.5 - height) * 1.5 * penalty_multiplier
            penalties += min(fall_penalty, 0.8)  

        # 3. Penalidade por movimento lateral excessivo 
        y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
        if y_velocity > 0.3:  
            lateral_penalty = (y_velocity - 0.3) * 1.0 * penalty_multiplier 
            penalties += min(lateral_penalty, 0.5)  

        # 4. Penalidade por inclinação excessiva 
        roll = abs(getattr(sim, "robot_roll", 0))
        if roll > 0.7:  
            roll_penalty = (roll - 0.7) * 0.8 * penalty_multiplier
            penalties += min(roll_penalty, 0.4)  

        pitch = abs(getattr(sim, "robot_pitch", 0))
        if pitch > 0.7:  
            pitch_penalty = (pitch - 0.7) * 0.6 * penalty_multiplier
            penalties += min(pitch_penalty, 0.3)  

        return penalties

    def calculate_marcha_sequence_bonus(self, sim, action) -> float:
        """BÔNUS MASSIVO para a sequência correta da marcha"""
        bonus = 1.0

        # 1. Verificar inclinação do tronco para frente
        pitch = getattr(sim, "robot_pitch", 0)
        if pitch > 0.05:  
            bonus += 50.0

        # 2. Verificar movimento alternado das pernas
        left_hip = getattr(sim, "robot_left_hip_angle", 0)
        right_hip = getattr(sim, "robot_right_hip_angle", 0)
        hip_difference = abs(left_hip - right_hip)

        if hip_difference > 0.4:  
            bonus += 100.0  
        elif hip_difference > 0.2:
            bonus += 40.0

        # 3. Bônus por velocidade positiva consistente
        velocity = getattr(sim, "robot_x_velocity", 0)
        if velocity > 0.1:
            bonus += velocity * 50.0

        # 4. PENALIDADE POR PERNAS MUITO ABERTAS (equilíbrio estático)
        left_hip_lateral = abs(getattr(sim, "robot_left_hip_lateral_angle", 0))
        right_hip_lateral = abs(getattr(sim, "robot_right_hip_lateral_angle", 0))

        if left_hip_lateral > 0.6 or right_hip_lateral > 0.6:
            bonus -= 50.0  

        # 5. BÔNUS por FASE DE VOO (marcha dinâmica)
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)

        if not left_contact and not right_contact:
            bonus += 25.0 

        # 6. BÔNUS POR FLEXÃO DOS JOELHOS PARA CIMA (DURANTE BALANÇO)
        left_knee = getattr(sim, "robot_left_knee_angle", 0)
        right_knee = getattr(sim, "robot_right_knee_angle", 0)

        # JOELHO ESQUERDO FLEXIONADO PARA CIMA durante balanço
        if not left_contact:
            if left_knee > 1.2:  
                bonus += 60.0
            elif left_knee > 0.9:  
                bonus += 40.0
            elif left_knee > 0.6:  
                bonus += 20.0
            elif left_knee > 0.3:  
                bonus += 10.0

        # JOELHO DIREITO FLEXIONADO PARA CIMA durante balanço  
        if not right_contact:
            if right_knee > 1.2:  
                bonus += 60.0
            elif right_knee > 0.9:  
                bonus += 40.0
            elif right_knee > 0.6:  
                bonus += 20.0
            elif right_knee > 0.3:  
                bonus += 10.0

        # 7. BÔNUS POR CLEARANCE ADEQUADO DOS PÉS 
        left_foot_height = getattr(sim, "robot_left_foot_height", 0)
        right_foot_height = getattr(sim, "robot_right_foot_height", 0)

        # Pé esquerdo com clearance adequado durante balanço
        if not left_contact and left_foot_height > 0.08:  
            bonus += 25.0
        elif not left_contact and left_foot_height > 0.05:  
            bonus += 12.0

        # Pé direito com clearance adequado durante balanço
        if not right_contact and right_foot_height > 0.08:  
            bonus += 25.0
        elif not right_contact and right_foot_height > 0.05:  
            bonus += 12.0

        # 8. BÔNUS COMBINADO: Joelho erguido + Clearance (VOLTEI!)
        if ((not left_contact and left_knee > 0.7 and left_foot_height > 0.06) or
            (not right_contact and right_knee > 0.7 and right_foot_height > 0.06)):
            bonus += 35.0  # Bônus extra por coordenação completa

        # 9. BÔNUS EXTRA POR COORDENAÇÃO: Joelho flexionado + Quadril estendido
        if not left_contact and left_knee > 0.8 and left_hip < -0.1:  # Joelho flexionado + quadril para trás
            bonus += 30.0
        if not right_contact and right_knee > 0.8 and right_hip < -0.1:  # Joelho flexionado + quadril para trás
            bonus += 30.0

        # 10. BÔNUS POR PADRÃO ALTERNADO DE FLEXÃO DOS JOELHOS
        if (not left_contact and left_knee > 0.7 and 
            right_contact and right_knee < 0.3):  # Esquerdo flexionado, direito estendido
            bonus += 25.0
        if (not right_contact and right_knee > 0.7 and 
            left_contact and left_knee < 0.3):  # Direito flexionado, esquerdo estendido
            bonus += 25.0

        return bonus

    def _calculate_stability_reward(self, sim, phase_info) -> float:
        try:
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            roll_penalty = min(roll / 0.5, 1.0)  
            pitch_penalty = min(pitch / 0.4, 1.0) 
            angular_stability = 1.0 - (roll_penalty * 0.6 + pitch_penalty * 0.4)
            com_height = getattr(sim, "robot_z_position", 0.8)
            target_com_height = 0.8  
            com_height_stability = 1.0 - min(abs(com_height - target_com_height) / 0.3, 1.0)
            com_vertical_vel = abs(getattr(sim, "robot_z_velocity", 0))
            vertical_velocity_stability = 1.0 - min(com_vertical_vel / 0.5, 1.0)
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            base_stability = 0.0
            if left_contact and right_contact: 
                base_stability = 0.8
            elif left_contact or right_contact:  
                base_stability = 0.5
            else:  
                base_stability = 0.2
            roll_vel = abs(getattr(sim, "robot_roll_vel", 0))
            pitch_vel = abs(getattr(sim, "robot_pitch_vel", 0))
            angular_vel_stability = 1.0 - min((roll_vel + pitch_vel) / 2.0, 1.0)
            stability_components = {
                'angular': angular_stability * 0.35,           
                'com_height': com_height_stability * 0.20,     
                'com_velocity': vertical_velocity_stability * 0.15,  
                'base': base_stability * 0.20,                 
                'angular_vel': angular_vel_stability * 0.10    
            }
            total_stability = sum(stability_components.values())
            movement_bonus = 0.0
            forward_velocity = getattr(sim, "robot_x_velocity", 0)
            if forward_velocity > 0.1 and total_stability > 0.7:
                movement_bonus = 0.1  
            final_stability = min(total_stability + movement_bonus, 1.0)

            return max(final_stability, 0.0)

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Erro no cálculo de estabilidade: {e}")
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            return 1.0 - min((roll + pitch) / 1.0, 1.0)
    
    def _calculate_basic_progress_reward(self, sim, phase_info) -> float:
        base_reward = 0.0

        # 1. COMPONENTE PRINCIPAL: DISTÂNCIA (100 pontos por metro)
        distance = getattr(sim, "episode_distance", 0)
        if distance > 0:
            distance_reward = distance * 100.0  # 100 pontos por metro
            base_reward += distance_reward

        # 2. BÔNUS POR VELOCIDADE
        velocity = getattr(sim, "robot_x_velocity", 0)
        if velocity > 0.1:
            velocity_reward = velocity * 10.0  # 10 pontos por m/s
            base_reward += velocity_reward

        # 3. BÔNUS POR SOBREVIVÊNCIA (se não terminou e se moveu)
        if not getattr(sim, "episode_terminated", True) and distance > 0.1:
            base_reward += 10.0

        return base_reward
    
    def _calculate_posture_reward(self, sim, phase_info) -> float:
        try:
            pitch = getattr(sim, "robot_pitch", 0)
            ideal_pitch = 0.15
            pitch_tolerance = 0.25
            pitch_diff = abs(pitch - ideal_pitch)
            if pitch_diff <= pitch_tolerance:
                pitch_score = 1.0 - (pitch_diff / pitch_tolerance) * 0.3  
            else:
                pitch_score = 0.7 - min((pitch_diff - pitch_tolerance) * 0.5, 0.6)  
            left_hip_lateral = abs(getattr(sim, "robot_left_hip_lateral_angle", 0))
            right_hip_lateral = abs(getattr(sim, "robot_right_hip_lateral_angle", 0))
            max_leg_spread = max(left_hip_lateral, right_hip_lateral)
            if max_leg_spread <= 0.4:    
                leg_spread_score = 1.0
            elif max_leg_spread <= 0.8:  
                leg_spread_score = 0.7
            else:                       
                leg_spread_score = 0.3
            com_height = getattr(sim, "robot_z_position", 0.8)
            height_diff = abs(com_height - 0.8)
            if height_diff <= 0.15:      
                height_score = 1.0
            elif height_diff <= 0.25:     
                height_score = 0.6
            else:                        
                height_score = 0.2
            total_score = (
                pitch_score * 0.5 +      
                leg_spread_score * 0.3 + 
                height_score * 0.2       
            )
            if abs(pitch) > 0.7 or max_leg_spread > 1.0:
                total_score *= 0.5  

            return max(total_score, 0.0)

        except Exception:
            pitch = abs(getattr(sim, "robot_pitch", 0))
            return 0.8 if pitch < 0.3 else 0.3
    
    def _calculate_velocity_reward(self, sim, phase_info) -> float:
        """Recompensa por velocidade consistente e eficiente"""
        current_velocity = getattr(sim, "robot_x_velocity", 0)
        target_velocity = 2.0  # ~7 km/h - objetivo realista
        if current_velocity < 0:
            return 0.0
        # Recompensa velocidade próxima do alvo com baixa variação
        velocity_ratio = current_velocity / target_velocity
        if 0.8 <= velocity_ratio <= 1.2:  # ±20% do alvo
            return 1.0
        elif 0.5 <= velocity_ratio < 0.8 or 1.2 < velocity_ratio <= 1.5:
            return 0.6
        else:
            return 0.1
    
    def _calculate_gait_pattern_reward(self, sim, phase_info) -> float:
        """Recompensa por padrão de marcha natural"""
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)
        
        # Marcha alternada ideal
        if left_contact != right_contact:
            base_score = 0.8
        else:
            base_score = 0.3
        
        # Bônus por ritmo consistente
        step_consistency = 1.0 - min(getattr(sim, "gait_variability", 0.3), 1.0)
        
        return base_score * 0.7 + step_consistency * 0.3
    
    def _calculate_clearance_reward(self, sim, phase_info) -> float:
        """Recompensa por clearance automático dos pés"""
        left_height = getattr(sim, "robot_left_foot_height", 0)
        right_height = getattr(sim, "robot_right_foot_height", 0)
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)
        
        clearance_score = 0.0
        # Recompensa pés altos quando não estão em contato
        if not left_contact and left_height > 0.05:
            clearance_score += 0.5
        if not right_contact and right_height > 0.05:
            clearance_score += 0.5
            
        return min(clearance_score, 1.0)
    
    def _calculate_dynamic_balance_reward(self, sim, phase_info) -> float:
        """Recompensa por estabilidade dinâmica durante movimento"""
        roll_vel = abs(getattr(sim, "robot_roll_vel", 0))
        pitch_vel = abs(getattr(sim, "robot_pitch_vel", 0))
        
        # Estabilidade angular durante movimento
        angular_stability = 1.0 - min((roll_vel + pitch_vel) / 3.0, 1.0)
        
        # Consistência da altura do COM
        com_height = getattr(sim, "robot_z_position", 0.8)
        height_consistency = 1.0 - min(abs(com_height - 0.8) / 0.2, 1.0)
        
        return (angular_stability * 0.6 + height_consistency * 0.4)
    
    def _calculate_smoothness_reward(self, sim, phase_info) -> float:
        """Recompensa por movimentos suaves e fluidos"""
        acceleration = abs(getattr(sim, "robot_x_acceleration", 0))
        jerk = abs(getattr(sim, "robot_jerk", 0))
        
        acceleration_smoothness = 1.0 - min(acceleration / 5.0, 1.0)
        jerk_smoothness = 1.0 - min(jerk / 20.0, 1.0)
        
        return (acceleration_smoothness * 0.7 + jerk_smoothness * 0.3)
    
    def _calculate_rhythm_reward(self, sim, phase_info) -> float:
        """Recompensa por ritmo consistente na marcha"""
        try:
            step_period = getattr(sim, "gait_step_period", 0.5)
            # Ritmo ideal ~0.5-0.6s por passo
            rhythm_quality = 1.0 - min(abs(step_period - 0.55) / 0.3, 1.0)
            
            # Consistência do comprimento da passada
            step_length_var = getattr(sim, "step_length_variability", 0.2)
            length_consistency = 1.0 - min(step_length_var / 0.3, 1.0)
            
            return (rhythm_quality * 0.6 + length_consistency * 0.4)
        except:
            return 0.5
    
    def _calculate_biomechanics_reward(self, sim, phase_info) -> float:
        """Recompensa por eficiência biomecânica"""
        distance = max(getattr(sim, "episode_distance", 0), 0)
        energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
        
        # Eficiência energética
        energy_efficiency = min(distance / energy, 2.0) / 2.0
        
        # Eficiência por passada
        steps = max(getattr(sim, "episode_steps", 1), 1)
        stride_efficiency = min(distance / steps, 0.1) / 0.1
        
        return (energy_efficiency * 0.6 + stride_efficiency * 0.4)
    
    def _calculate_robustness_reward(self, sim, phase_info) -> float:
        """Recompensa por robustez da marcha"""
        recovery_events = getattr(sim, "recovery_success_count", 0)
        total_perturbations = max(getattr(sim, "total_perturbations", 1), 1)
        
        recovery_rate = recovery_events / total_perturbations
        return min(recovery_rate, 1.0)
    
    def _calculate_adaptation_reward(self, sim, phase_info) -> float:
        """Recompensa por adaptação a diferentes condições"""
        speed_adaptation = getattr(sim, "speed_adaptation_score", 0.5)
        terrain_handling = getattr(sim, "terrain_handling_score", 0.5)
        
        return (speed_adaptation * 0.6 + terrain_handling * 0.4)
    
    def _calculate_recovery_reward(self, sim, phase_info) -> float:
        """Recompensa por recuperação eficiente de perturbações"""
        recovery_time = getattr(sim, "recovery_time", 2.0)
        recovery_efficiency = 1.0 - min(recovery_time / 5.0, 1.0)
        
        return recovery_efficiency

    def _calculate_phase_angles_reward(self, sim, phase_info) -> float:
        try:
            left_knee = getattr(sim, "robot_left_knee_angle", 0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0)
            ideal_extension = 0.2 
            ideal_swing = 1.2     
            left_score = 0.0
            right_score = 0.0
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)

            if left_contact:
                left_score = np.exp(-3.0 * (left_knee - ideal_extension)**2)
            else:  
                left_score = np.exp(-2.0 * (left_knee - ideal_swing)**2)

            if right_contact:
                right_score = np.exp(-3.0 * (right_knee - ideal_extension)**2)
            else:
                right_score = np.exp(-2.0 * (right_knee - ideal_swing)**2)

            coordination_bonus = 0.0
            if left_contact != right_contact:  
                coordination_bonus = 0.3

            overflex_penalty = 0.0
            if left_knee > 1.5 or right_knee > 1.5:
                overflex_penalty = 0.2

            extension_bonus = 0.0
            if not left_contact and left_knee < 0.8:
                extension_bonus += 0.2
            if not right_contact and right_knee < 0.8:
                extension_bonus += 0.2

            base_score = (left_score + right_score) / 2.0
            final_score = base_score + coordination_bonus + (extension_bonus * 2) - overflex_penalty

            return min(max(final_score, 0.0), 1.0)

        except Exception as e:
            self.logger.warning(f"Erro no cálculo de recompensa de ângulos: {e}")
            return 0.3  
    
    def _calculate_movement_priority_reward(self, sim, phase_info) -> float:
        """RECOMPENSA - PENALIDADE NUCLEAR POR MOVIMENTO NEGATIVO"""
        distance = getattr(sim, "episode_distance", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)

        # PENALIDADE NUCLEAR POR MOVIMENTO NEGATIVO
        if distance < 0:
            nuclear_penalty = -10000.0 
            return nuclear_penalty

        base_reward = 0.0

        if distance > 0:
            # ESCALA por metro!
            base_reward += distance * 1000.0

            # BÔNUS por marcos - QUALQUER progresso
            if distance > 2.0: base_reward += 3000.0
            elif distance > 1.5: base_reward += 2500.0
            elif distance > 1.0: base_reward += 2000.0
            elif distance > 0.7: base_reward += 1500.0
            elif distance > 0.5: base_reward += 1000.0
            elif distance > 0.3: base_reward += 500.0
            elif distance > 0.2: base_reward += 200.0
            elif distance > 0.1: base_reward += 100.0
            elif distance > 0.05: base_reward += 50.0
            elif distance > 0.02: base_reward += 20.0
            elif distance > 0.01: base_reward += 10.0

        # RECOMPENSA por velocidade positiva
        if velocity > 0:
            base_reward += velocity * 500.0 
        elif velocity < 0:
            # PENALIDADE por velocidade negativa
            base_reward -= 500.0  
        # BÔNUS por sobrevivência com movimento positivo
        if not getattr(sim, "episode_terminated", True) and distance > 0.01:
            base_reward += 200.0

        return base_reward

    def _calculate_propulsion_reward(self, sim, phase_info) -> float:
        pitch = getattr(sim, "robot_pitch", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        if velocity < 0:
            return 0.0
        if pitch < -0.05 and velocity > 0.1:
            return min(abs(pitch) * velocity * 4.0, 1.0)
        return 0.0
    
    def _calculate_coordination_reward(self, sim, phase_info) -> float:
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            if left_contact != right_contact:
                base_coordination = 0.8 
            elif not left_contact and not right_contact:
                base_coordination = 0.6  
            else:
                base_coordination = 0.2 
            current_time = getattr(sim, "episode_steps", 0) * getattr(sim, "time_step_s", 0.033)
            time_since_last_transition = current_time % 1.0  
            rhythm_quality = 0.7 + 0.3 * (1.0 - min(abs(time_since_last_transition - 0.5) * 2.0, 1.0))
            com_velocity_y = abs(getattr(sim, "robot_y_velocity", 0))
            stability_during_transition = 1.0 - min(com_velocity_y / 0.3, 1.0)
            coordination_score = (
                base_coordination * 0.60 +           
                rhythm_quality * 0.25 +              
                stability_during_transition * 0.15   
            )
            forward_velocity = getattr(sim, "robot_x_velocity", 0)
            if forward_velocity > 0.3 and coordination_score > 0.7:
                coordination_score = min(coordination_score + 0.1, 1.0)

            return max(coordination_score, 0.0)

        except Exception:
            try:
                left_contact = getattr(sim, "robot_left_foot_contact", False)
                right_contact = getattr(sim, "robot_right_foot_contact", False)
                if left_contact != right_contact:
                    return 0.8
                elif not left_contact and not right_contact:
                    return 0.5
                else:
                    return 0.2
            except:
                return 0.3
    
    def _calculate_efficiency_reward(self, sim, phase_info) -> float:
        try:
            distance = getattr(sim, "episode_distance", 0)
            steps = max(getattr(sim, "episode_steps", 1), 1)
            energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
            steps_efficiency = distance / steps
            normalized_steps_eff = min(steps_efficiency / 0.08, 1.0)
            energy_efficiency = distance / energy
            normalized_energy_eff = min(energy_efficiency / 2.0, 1.0)
            current_velocity = abs(getattr(sim, "robot_x_velocity", 0))
            target_velocity = phase_info.get('target_speed', 1.0)
            velocity_efficiency = 1.0 - min(abs(current_velocity - target_velocity) / target_velocity, 1.0)
            com_velocity_y = abs(getattr(sim, "robot_y_velocity", 0))
            lateral_efficiency = 1.0 - min(com_velocity_y / 0.2, 1.0)
            combined_efficiency = (
                normalized_steps_eff * 0.40 +    
                normalized_energy_eff * 0.30 +   
                velocity_efficiency * 0.20 +    
                lateral_efficiency * 0.10       
            )
            if distance > 1.0 and combined_efficiency > 0.6:
                combined_efficiency = min(combined_efficiency + 0.1, 1.0)

            return min(combined_efficiency, 1.0)

        except Exception:
            try:
                distance = getattr(sim, "episode_distance", 0)
                steps = max(getattr(sim, "episode_steps", 1), 1)
                energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
                steps_efficiency = distance / steps
                energy_efficiency = distance / energy
                return min((steps_efficiency * 0.6 + energy_efficiency * 0.4) * 2.0, 1.0)
            except:
                return 0.3
    
    def _calculate_success_bonus(self, sim, phase_info) -> float:
        success = getattr(sim, "episode_success", False)
        return 1.0 if success else 0.0
    
    def _calculate_effort_penalty(self, sim, phase_info) -> float:
        try:
            joint_velocities = getattr(sim, "joint_velocities", [0])
            effort = sum(v**2 for v in joint_velocities) / len(joint_velocities) if joint_velocities else 0
            return min(effort * 0.1, 0.5)
        except:
            return 0.0
    
    def get_reward_status(self) -> Dict:
        """Retorna status do sistema de recompensa"""
        active_components = [name for name, comp in self.components.items() if comp.enabled]
        status = {
            "active_components": active_components,
            "total_components": len(self.components),
            "component_weights": {name: comp.weight for name, comp in self.components.items() if comp.enabled}
        }
        
        return status
    

class IntelligentCache:
    def __init__(self, max_size=1000, default_ttl=100):
        self._cache = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._access_pattern = {}  
    
    def get(self, key: str) -> Any:
        if key in self._cache:
            value, timestamp, ttl = self._cache[key]
            if time.time() - timestamp < ttl:
                self._hits += 1
                self._access_pattern[key] = self._access_pattern.get(key, 0) + 1
                return value
            else:
                del self._cache[key]
                del self._access_pattern[key]
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
            
        self._cache[key] = (value, time.time(), ttl or self._default_ttl)
        self._access_pattern[key] = self._access_pattern.get(key, 0) + 1
    
    def _evict_oldest(self):
        """Remove entradas menos acessadas primeiro"""
        if not self._cache:
            return
            
        # Combinar idade + frequência de acesso
        def eviction_score(k):
            value, timestamp, ttl = self._cache[k]
            age = time.time() - timestamp
            access_count = self._access_pattern.get(k, 0)
            return age / (access_count + 1)  # +1 para evitar divisão por zero
            
        oldest_key = min(self._cache.keys(), key=eviction_score)
        del self._cache[oldest_key]
        del self._access_pattern[oldest_key]
    
    def get_hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.get_hit_rate(),
            "size": len(self._cache),
            "max_size": self._max_size
        }
    

class CachedRewardCalculator(RewardCalculator):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.cache = IntelligentCache(max_size=500, default_ttl=50)
        self._last_sim_state = None
        self._last_reward = 0.0
        self._component_priority_cache = {}
        self._last_gait_phase = "unknown"
        self._gait_phase_cache = {}
        
    def calculate(self, sim, action, phase_info: Dict) -> float:
        base_reward = 0.0
        distance = max(getattr(sim, "episode_distance", 0), 0)

        # 1. BÔNUS PRIMÁRIO: PADRÃO DE MARCHA (60% da recompensa)
        marcha_bonus = self.calculate_marcha_sequence_bonus(sim, action)
        base_reward += marcha_bonus

        # 2. BÔNUS SECUNDÁRIO: DISTÂNCIA (30% da recompensa)
        if distance > 0:
            distance_reward = distance * 100.0  
            base_reward += distance_reward

            # BÔNUS progressivo por marcos de DISTÂNCIA COM MARCHA
            if distance > 1.5: base_reward += 200.0
            elif distance > 1.0: base_reward += 100.0
            elif distance > 0.7: base_reward += 50.0
            elif distance > 0.5: base_reward += 20.0
            elif distance > 0.3: base_reward += 10.0

        # 3. BÔNUS TERCIÁRIO: COORDENAÇÃO (10% da recompensa)
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)

        if left_contact != right_contact:  
            base_reward += 15.0

        # 4. PENALIDADES LEVES por instabilidade 
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))

        if roll > 0.5: base_reward -= roll * 10.0  
        if pitch > 0.5: base_reward -= pitch * 8.0  

        return max(base_reward, 0.0)
    
    def _get_sim_state_fingerprint(self, sim) -> str:
        """Fingerprint rápido do estado da simulação"""
        try:
            # Métricas essenciais para fingerprint
            return f"{getattr(sim, 'robot_x_velocity', 0):.2f}_" \
                   f"{getattr(sim, 'robot_roll', 0):.2f}_" \
                   f"{getattr(sim, 'robot_pitch', 0):.2f}_" \
                   f"{getattr(sim, 'robot_z_position', 0):.2f}_" \
                   f"{getattr(sim, 'robot_left_foot_contact', False)}_" \
                   f"{getattr(sim, 'robot_right_foot_contact', False)}"
        except Exception as e:
            self.logger.warning(f"Erro ao criar fingerprint: {e}")
            return "unknown"
    
    def _sim_states_similar(self, state1: str, state2: str, threshold: float = 0.05) -> bool:
        """Comparação MAIS PERMISSIVA para cache"""
        if state1 == "unknown" or state2 == "unknown":
            return False

        try:
            # Verifica se é o mesmo episódio
            parts1 = state1.split('_')
            parts2 = state2.split('_')

            # Se tem menos de 3 partes, não é estado completo
            if len(parts1) < 3 or len(parts2) < 3:
                return False

            # Compara os primeiros elementos (velocidade básica)
            vel1 = float(parts1[0]) if parts1[0].replace('.', '').isdigit() else 0
            vel2 = float(parts2[0]) if parts2[0].replace('.', '').isdigit() else 0

            if abs(vel1 - vel2) > 0.5:  
                return False

            return True

        except Exception as e:
            return False
    
    def _get_gait_phase_from_state(self, state_str: str) -> str:
        """Detecta fase da marcha baseado no estado"""
        try:
            parts = state_str.split('_')
            if len(parts) < 6:
                return "unknown"
                
            left_contact = parts[4].lower() == "true"
            right_contact = parts[5].lower() == "true"
            
            if not left_contact and not right_contact:
                return "flight"
            elif left_contact and not right_contact:
                return "left_stance"
            elif not left_contact and right_contact:
                return "right_stance" 
            else:
                return "double_support"
                
        except Exception:
            return "unknown"
    
    def _prioritize_components(self, sim, enabled_components: List[str]) -> List[str]:
        """Priorização MAIS EFETIVA baseada em pesquisa real"""
        movement_components = ["basic_progress", "velocity", "propulsion"]
        stability_components = ["stability", "posture", "dynamic_balance"]

        # PRIMEIRO: Movimento (60%)
        prioritized = [c for c in movement_components if c in enabled_components]

        # SEGUNDO: Estabilidade (30%)
        prioritized.extend([c for c in stability_components if c in enabled_components])

        # TERCEIRO: Outros (10%)
        other_components = [c for c in enabled_components 
                           if c not in movement_components + stability_components]
        prioritized.extend(other_components)

        return prioritized[:6]
    
    def _generate_cache_key(self, sim, phase_info: Dict) -> str:
        """Gera chave de cache única"""
        base_key = self._get_sim_state_fingerprint(sim)
        components_key = "_".join(sorted(phase_info.get('enabled_components', [])))
        group_key = str(phase_info.get('group_level', 1))
        distance = getattr(sim, "episode_distance", 0)
        return f"{base_key}_{components_key}_{group_key}_{distance:.2f}"
    
    def _calculate_prioritized_reward(self, sim, action, phase_info: Dict) -> float:
        """Cálculo com priorização inteligente de componentes"""
        total_reward = 0.0
        enabled_components = phase_info['enabled_components']
        prioritized_components = self._prioritize_components(sim, enabled_components)
        max_components = min(8, len(prioritized_components))
        calculated_components = 0
        
        for component_name in prioritized_components:
            if component_name in self.components and calculated_components < max_components:
                component = self.components[component_name]
                if component.enabled:
                    component_reward = component.calculator(sim, phase_info)
                    
                    weight = self._get_component_weight(component_name, phase_info)
                    weighted_reward = weight * component_reward
                    total_reward += weighted_reward
                    calculated_components += 1

        penalties = self._calculate_essential_penalties(sim, action)
        total_reward -= penalties

        return max(total_reward, -0.5)
        
    def _get_component_weight(self, component_name: str, phase_info: Dict) -> float:
        """Obtém peso do componente de forma otimizada"""
        valence_weights = phase_info.get('valence_weights', {})
        irl_weights = phase_info.get('irl_weights', {})
        component = self.components[component_name]
        
        if irl_weights and component_name in irl_weights:
            return irl_weights[component_name]
        elif valence_weights and component_name in valence_weights:
            return valence_weights[component_name]
        else:
            return component.weight * component.adaptive_weight
    
    def _calculate_essential_penalties(self, sim, action) -> float:
        """Calcula apenas penalidades essenciais"""
        penalties = 0.0
        
        # 1. Penalidade de ação extrema 
        if hasattr(action, '__len__'):
            action_magnitude = np.sqrt(np.sum(np.square(action)))
            penalties += min(action_magnitude * 0.005, 0.3)
        
        # 2. Penalidade por queda iminente 
        height = getattr(sim, "robot_z_position", 0.8)
        if height < 0.5:
            penalties += min((0.5 - height) * 1.5, 0.8)
            
        # 3. Penalidade por instabilidade extrema
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))
        if roll > 0.8 or pitch > 0.8:
            penalties += 0.3
        
        return penalties
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache para monitoramento"""
        if hasattr(self.cache, 'get_stats'):
            return self.cache.get_stats()
        return {"cache_status": "active", "cache_size": "unknown"}