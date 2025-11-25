# dpg_reward.py
import time
import numpy as np
from typing import Any, Dict, List, Callable, Tuple
from dataclasses import dataclass
from dpg_buffer import Cache

@dataclass
class RewardComponent:
    name: str
    weight: float
    calculator: Callable
    enabled: bool = True
    adaptive_weight: float = 1.0

class RewardCalculator:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        
        # Apenas 6 componentes macro para o crítico ajustar
        self.macro_components = {
            "progresso": 1.5,
            "coordenacao": 1.0, 
            "estabilidade": 1.2,
            "eficiencia": 0.8,
            "valencia_bonus": 0.5,
            "penalidades": 1.0,
            "sparse_success": 3.0
        }
        
        self.base_macro_weights = self.macro_components.copy()
        self.weight_adjustment_rate = 0.002  
        self.max_weight_change = 0.4  
        
        # Sistema de cache unificado
        self.cache = Cache(max_size=2000)
        self._last_sim_state = None
        self._last_reward = 0.0
        
        # Estatísticas
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0

        self.macro_components["sparse_success"] = 0.0  
        self.base_macro_weights["sparse_success"] = 0.0

        self.terrain_parameters = {
            "normal": {
                "speed_weight": 40.0,
                "clearance_min": 0.03,
                "stability_weight": 0.5,
                "coordination_bonus": 15.0,
                "height_target": 0.75,  
                "pitch_target": 0.05
            },
            "ramp_up": {
                "speed_weight": 10.0,  # Mais importante em subida
                "clearance_min": 0.10, # Clearance maior necessário
                "stability_weight": 1.8, # Estabilidade crítica
                "coordination_bonus": 30.0,
                "pitch_target": 0.25,   # Pitch ideal para subida
                "uphill_bonus": 2.0,    # Bônus extra para subida
                "stance_knee_lock_bonus": 1.0, # bônus para travar joelho *durante stance* em subida
                "push_phase_bonus": 1.0        # bônus por impulso ativo (ver abaixo)
            },
            "ramp_down": {
                "speed_weight": 8.0,   # Velocidade controlada em descida
                "clearance_min": 0.10,
                "stability_weight": 1.6, # Máxima prioridade para estabilidade
                "coordination_bonus": 18.0,
                "pitch_target": -0.15, # Pitch ideal para descida
                "braking_bonus": 2.0   # Bônus por frenagem controlada
            },
            "uneven": {
                "speed_weight": 10.0,
                "clearance_min": 0.08,
                "stability_weight": 1.4,
                "coordination_bonus": 25.0, # Coordenação muito importante
                "adaptation_bonus": 1.3
            },
            "low_friction": {
                "speed_weight": 8.0,
                "clearance_min": 0.07,
                "stability_weight": 1.5,
                "coordination_bonus": 20.0,
                "slip_penalty_multiplier": 2.0
            },
            "complex": {
                "speed_weight": 12.0,
                "clearance_min": 0.09,
                "stability_weight": 1.4,
                "coordination_bonus": 30.0, 
                "robustness_bonus": 1.4
            }
        }
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
        current_terrain = phase_info.get('current_terrain', 'normal')
        terrain_params = self.terrain_parameters.get(current_terrain, self.terrain_parameters["normal"])
        self._total_calculations += 1

        # Cache otimizado
        cache_key = self._generate_essential_cache_key(sim, phase_info)
        cached_reward = self.cache.get(cache_key)
        if cached_reward is not None:
            self._cache_hits += 1
            return cached_reward

        self._cache_misses += 1

        # CÁLCULO POR CATEGORIAS MACRO
        component_values = {
            "progresso": self._calculate_progresso_component(sim, phase_info, terrain_params),
            "coordenacao": self._calculate_coordenacao_component(sim, phase_info, terrain_params),
            "estabilidade": self._calculate_estabilidade_component(sim, phase_info, terrain_params),
            "eficiencia": self._calculate_eficiencia_component(sim, phase_info, terrain_params),
            "valencia_bonus": self._calculate_valencia_bonus_component(sim, phase_info),
            "penalidades": self._calculate_penalidades_component(sim, phase_info, terrain_params),
            "sparse_success": self._calculate_sparse_success_component(sim, phase_info),
        }

        # SOMA PONDERADA COM OS PESOS MACRO
        total_reward = 0.0
        for component, value in component_values.items():
            total_reward += value * self.macro_components[component]

        # Cache
        self.cache.set(cache_key, total_reward)

        return max(total_reward, 0.0)  

    # =========================================================================
    # COMPONENTES MACRO (cada um agrega múltiplas funções específicas)
    # =========================================================================
    
    def _calculate_progresso_component(self, sim, phase_info, terrain_params) -> float:
        """PROGRESSO: recompensa distância *apenas* se marcha for funcional e estável."""
        reward = 0.0
        try:
            dist = max(getattr(sim, "episode_distance", 0.0), 0.0)
            vel_x = getattr(sim, "robot_x_velocity", 0.0)
            current_terrain = phase_info.get('current_terrain', 'normal')

            # USA PARÂMETROS DO TERRENO CORRETAMENTE
            speed_weight = terrain_params["speed_weight"]
            clearance_min = terrain_params["clearance_min"]
            push_phase_bonus = terrain_params.get("push_phase_bonus", 1.0)
            uphill_bonus = terrain_params.get("uphill_bonus", 1.0)

            # Estabilidade e funcionalidade mínimas
            roll = abs(getattr(sim, "robot_roll", 0.0))
            pitch = getattr(sim, "robot_pitch", 0.0)

            # PITCH TARGET ESPECÍFICO POR TERRENO
            pitch_target = terrain_params.get("pitch_target", 0.0)
            forward_momentum = vel_x * min(1.0, 1.0 - abs(pitch - pitch_target) / 0.3)
            stable = (roll < 0.25 and abs(pitch - pitch_target) < 0.3)

            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            alternating = (left_contact != right_contact)

            left_knee = getattr(sim, "robot_left_knee_angle", 0.0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0.0)
            left_h = getattr(sim, "robot_left_foot_height", 0.0)
            right_h = getattr(sim, "robot_right_foot_height", 0.0)
            left_xv = getattr(sim, "robot_left_foot_x_velocity", 0.0)
            right_xv = getattr(sim, "robot_right_foot_x_velocity", 0.0)
            robot_xv = getattr(sim, "robot_velocity", [0,0,0])[0]
            
            # COMPORTAMENTO ESPECÍFICO POR TERRENO
            if current_terrain == "ramp_up":
                # Bônus extra para progresso em subida
                if forward_momentum > 0.1 and dist > 0.2:
                    reward += uphill_bonus * dist * 20.0
                
                # Bônus por inclinação funcional (pitch > 0.15) + impulso
                pitch_ok = pitch > 0.15 and pitch < 0.35
                left_push = left_contact and left_xv < robot_xv - 0.05
                right_push = right_contact and right_xv < robot_xv - 0.05
                if pitch_ok and (left_push or right_push):
                    reward += 8.0 * push_phase_bonus

            elif current_terrain == "ramp_down":
                # Progresso controlado em descida (não muito rápido)
                if forward_momentum < 0.5 and dist > 0.1:
                    reward += dist * 25.0

            elif current_terrain == "uneven":
                # Progresso constante é mais valioso que rápido
                if dist > 0.1 and forward_momentum < 0.8:
                    reward += dist * 30.0

            # USA CLEARANCE_MIN DO TERRENO (não valor fixo)
            knee_th = 0.8
            functional = (
                ((not left_contact) and left_h > clearance_min and left_knee > knee_th) or
                ((not right_contact) and right_h > clearance_min and right_knee > knee_th)
            )

            # Só recompensa se estável E funcional
            if stable and functional and alternating:
                reward += dist * 40.0  
                reward += forward_momentum * speed_weight  
                if dist >= 3.0:
                    reward += 300.0  
                elif dist >= 1.0:
                    reward += 150.0  
                elif dist >= 0.5:
                    reward += 70.0
            elif stable and alternating:
                # Progresso básico (sem padrão ideal)
                reward += dist * 15.0 + forward_momentum * (speed_weight * 0.3)  
            else:
                # Progresso instável → quase nada
                reward += dist * 2.0 + forward_momentum * 0.5

            # Sobrevivência só conta se avançou >5 cm
            if not getattr(sim, "episode_terminated", True) and dist > 0.05:
                reward += 20.0

            # BÔNUS PROGRESSIVO ESPECIAL PARA LONGAS DISTÂNCIAS
            if dist > 3.0:
                long_distance_bonus = (dist - 3.0) * 50.0  
                reward += long_distance_bonus

            # MARCOS PROGRESSIVOS
            if dist >= 5.0:
                reward += 300
            if dist >= 7.0:
                reward += 500
            if dist >= 9.0:
                reward += 1000

        except Exception as e:
            self.logger.warning(f"Erro em progresso (terrain-corrected): {e}")
        return reward

    def _calculate_coordenacao_component(self, sim, phase_info, terrain_params) -> float:
        """COORDENAÇÃO: força padrão de marcha funcional."""
        reward = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            coordination_bonus = terrain_params["coordination_bonus"]
            clearance_min = terrain_params["clearance_min"]
            stance_knee_lock_bonus = terrain_params.get("stance_knee_lock_bonus", 1.0)
            adaptation_bonus = terrain_params.get("adaptation_bonus", 1.0)
            robustness_bonus = terrain_params.get("robustness_bonus", 1.0)

            # Dados essenciais
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            alternating = (left_contact != right_contact)
            consecutive = getattr(sim, "consecutive_alternating_steps", 0)

            left_knee = getattr(sim, "robot_left_knee_angle", 0.0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0.0)
            left_hip_f = getattr(sim, "robot_left_hip_frontal_angle", 0.0)
            right_hip_f = getattr(sim, "robot_right_hip_frontal_angle", 0.0)
            left_hip_l = getattr(sim, "robot_left_hip_lateral_angle", 0.0)
            right_hip_l = getattr(sim, "robot_right_hip_lateral_angle", 0.0)

            left_h = getattr(sim, "robot_left_foot_height", 0.0)
            right_h = getattr(sim, "robot_right_foot_height", 0.0)
            left_xv = getattr(sim, "robot_left_foot_x_velocity", 0.0)
            right_xv = getattr(sim, "robot_right_foot_x_velocity", 0.0)
            robot_xv = getattr(sim, "robot_velocity", [0,0,0])[0]

            if current_terrain == "ramp_up":
                # Bônus por travamento funcional do joelho em stance
                left_knee_lock_ok = left_contact and abs(left_knee) < 0.2  # joelho estendido
                right_knee_lock_ok = right_contact and abs(right_knee) < 0.2
                if left_knee_lock_ok or right_knee_lock_ok:
                    reward += 6.0 * stance_knee_lock_bonus
                if left_knee_lock_ok and right_knee_lock_ok:
                    reward += 4.0
        
            # --- Base: alternância estrita ---
            if alternating:
                reward += coordination_bonus * 0.5  
                if consecutive >= 5:
                    reward += min(consecutive * 2.0, coordination_bonus * 0.5)  
            else:
                reward -= 12.0

            # USA CLEARANCE_MIN DO TERRENO (não valor fixo)
            left_clear = (not left_contact) and (left_h > clearance_min)
            right_clear = (not right_contact) and (right_h > clearance_min)
            if left_clear or right_clear:
                reward += 6.0
            if left_clear and right_clear:
                reward += 4.0

            # --- Flexão funcional no swing (joelho + quadril frontal) ---
            knee_th = 0.9
            hip_f_th = 0.6
            left_knee_ok = (not left_contact) and (left_knee > knee_th)
            right_knee_ok = (not right_contact) and (right_knee > knee_th)
            left_hip_ok = (not left_contact) and (left_hip_f > hip_f_th)
            right_hip_ok = (not right_contact) and (right_hip_f > hip_f_th)

            if left_knee_ok or right_knee_ok:
                reward += 6.0
            if left_hip_ok or right_hip_ok:
                reward += 5.0
            if (left_knee_ok and left_hip_ok) or (right_knee_ok and right_hip_ok):
                reward += 7.0  

            # --- Penalização: abertura excessiva de pernas ---
            abd_pen = 0.0
            for hip_l in [left_hip_l, right_hip_l]:
                if abs(hip_l) > 0.2:
                    abd_pen += (abs(hip_l) - 0.2) * 20.0
            reward -= abd_pen

            # --- Penalização: arrasto de pé ---
            drag_th = 0.2
            left_drag = (not left_contact) and abs(left_xv - robot_xv) < drag_th
            right_drag = (not right_contact) and abs(right_xv - robot_xv) < drag_th
            if left_drag:
                reward -= 10.0
            if right_drag:
                reward -= 10.0

            # BÔNUS ESPECÍFICOS
            clearance_ok = (left_h > clearance_min) or (right_h > clearance_min)

            if current_terrain == "uneven" and alternating and consecutive >= 3:
                reward += adaptation_bonus * 10.0

            if current_terrain == "complex" and alternating and clearance_ok:
                reward += robustness_bonus * 15.0

        except Exception as e:
            self.logger.warning(f"Erro em coordenação (terrain-corrected): {e}")

        return reward

    def _calculate_estabilidade_component(self, sim, phase_info, terrain_params) -> float:
        """ESTABILIDADE: reforça postura, penaliza abertura de pernas."""
        reward = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            stability_weight = terrain_params["stability_weight"]

            # ALVOS ESPECÍFICOS POR TERRENO
            pitch_target = terrain_params.get("pitch_target", 0.0)
            pitch = getattr(sim, "robot_pitch", 0.0)
            pitch_error = abs(pitch - pitch_target)

            roll = abs(getattr(sim, "robot_roll", 0.0))
            com_z = getattr(sim, "robot_z_position", 0.8)
            y_vel = abs(getattr(sim, "robot_y_velocity", 0.0))
            left_hip_l = getattr(sim, "robot_left_hip_lateral_angle", 0.0)
            right_hip_l = getattr(sim, "robot_right_hip_lateral_angle", 0.0)
            left_roll = getattr(sim, "robot_left_foot_roll", 0.0)
            right_roll = getattr(sim, "robot_right_foot_roll", 0.0)

            # --- Estabilidade angular ajustada para terreno ---
            if current_terrain == "ramp_up":
                if 0.15 <= pitch <= 0.35:
                    pitch_stab = 1.0
                else:
                    pitch_stab = max(0.0, 1.0 - abs(pitch - 0.25) / 0.2)
            else:
                pitch_stab = max(0.0, 1.0 - pitch_error / 0.4)            

            roll_stab = max(0.0, 1.0 - roll / 0.25)
            angular_stab = (pitch_stab + roll_stab) / 2.0
            reward += angular_stab * 20.0

            # --- Altura do COM (ajustada por terreno) ---
            height_target = terrain_params.get("height_target", 0.8)
            height_error = abs(com_z - height_target)
            height_stab = max(0.0, 1.0 - height_error / 0.2)
            reward += height_stab * 10.0

            # --- Estabilidade lateral ---
            lateral_penalty = y_vel * 20.0
            for hip_l in [left_hip_l, right_hip_l]:
                if abs(hip_l) > 0.15:
                    lateral_penalty += (abs(hip_l) - 0.15) * 15.0
            reward -= lateral_penalty

            # --- Alinhamento dos pés ---
            foot_roll_error = abs(left_roll) + abs(right_roll)
            if foot_roll_error > 0.3:
                reward -= foot_roll_error * 10.0

            # APLICA WEIGHT DO TERRENO NO FINAL
            reward *= stability_weight

        except Exception as e:
            self.logger.warning(f"Erro em estabilidade (terrain-corrected): {e}")
        return reward

    def _calculate_eficiencia_component(self, sim, phase_info, terrain_params) -> float:
        """EFICIÊNCIA: prioriza flexão ativa, penaliza esforço em abdução."""
        reward = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            efficiency_weight = terrain_params.get("efficiency_weight", 1.0)

            # Esforço articular 
            effort = 0.0
            if hasattr(sim, 'joint_velocities'):
                effort = sum(v**2 for v in sim.joint_velocities) * 0.001
            effort_eff = max(0.0, 1.0 - effort / 15.0)
            reward += effort_eff * 8.0

            # --- Eficiência de propulsão ---
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            left_xv = getattr(sim, "robot_left_foot_x_velocity", 0.0)
            right_xv = getattr(sim, "robot_right_foot_x_velocity", 0.0)
            robot_xv = getattr(sim, "robot_velocity", [0,0,0])[0]

            push_left = left_contact and (left_xv < robot_xv - 0.1)
            push_right = right_contact and (right_xv < robot_xv - 0.1)
            if push_left or push_right:
                reward += 6.0
            if push_left and push_right:
                reward += 4.0

            # --- Penalização: esforço em abdução ---
            left_hip_l = getattr(sim, "robot_left_hip_lateral_angle", 0.0)
            right_hip_l = getattr(sim, "robot_right_hip_lateral_angle", 0.0)
            abd_effort = 0.0
            for hip_l in [left_hip_l, right_hip_l]:
                if abs(hip_l) > 0.15:
                    abd_effort += abs(hip_l) * 3.0
            reward -= abd_effort

            # --- Bônus: simetria de esforço ---
            left_knee = getattr(sim, "robot_left_knee_angle", 0.0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0.0)
            symmetry = 1.0 - min(abs(left_knee - right_knee) / 1.0, 1.0)
            reward += symmetry * 3.0

            # APLICA WEIGHT DE EFICIÊNCIA DO TERRENO
            reward *= efficiency_weight

        except Exception as e:
            self.logger.warning(f"Erro em eficiência (terrain-corrected): {e}")
        return max(reward, -20.0)
    
    def _calculate_valencia_bonus_component(self, sim, phase_info) -> float:
        """Componente de BÔNUS DE VALÊNCIAS: recompensas adaptativas"""
        bonus = 0.0
        
        try:
            valence_status = phase_info.get('valence_status', {})
            active_valences = valence_status.get('active_valences', [])
            
            for valence in active_valences:
                if valence == "estabilidade_postural":
                    roll = abs(getattr(sim, "robot_roll", 0))
                    pitch = abs(getattr(sim, "robot_pitch", 0))
                    stability = 1.0 - min((roll + pitch) / 1.0, 1.0)
                    bonus += stability * 25.0
                    
                elif valence == "propulsao_basica":
                    velocity = getattr(sim, "robot_x_velocity", 0)
                    if velocity > 0:
                        bonus += velocity * 18.0
                        
                elif valence == "coordenacao_fundamental":
                    left_contact = getattr(sim, "robot_left_foot_contact", False)
                    right_contact = getattr(sim, "robot_right_foot_contact", False)
                    if left_contact != right_contact:
                        bonus += 30.0

                elif valence == "coordenacao_avancada":
                    left_knee = getattr(sim, "robot_left_knee_angle", 0.0)
                    right_knee = getattr(sim, "robot_right_knee_angle", 0.0)
                    left_contact = getattr(sim, "robot_left_foot_contact", False)
                    right_contact = getattr(sim, "robot_right_foot_contact", False)
                    alternating = (left_contact != right_contact)

                    stable_stance = ((left_contact and abs(left_knee) < 0.2) or 
                                     (right_contact and abs(right_knee) < 0.2))
                    if stable_stance and alternating:
                        bonus += 25.0
                        
        except Exception as e:
            self.logger.warning(f"Erro em bônus de valência: {e}")
            
        return bonus
    
    def _calculate_sparse_success_component(self, sim, phase_info) -> float:
        if not phase_info.get('sparse_success_enabled', False):
            return 0.0

        try:
            distance = max(getattr(sim, "episode_distance", 0), 0)
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            terminated = getattr(sim, "episode_terminated", True)
            steps = getattr(sim, "episode_steps", 500)

            stable = (roll < 0.3 and pitch < 0.3)
            if not stable or terminated:
                return 0.0

            if distance <= 0.5:
                return 0.0

            # SISTEMA DE RECOMPENSA PROGRESSIVA PARA 9M
            base_reward = 0.0
            if distance < 3.0:
                base_reward = (distance / 3.0) * 300  # Até 300 pontos
            elif distance < 5.0:
                base_reward = 300 + ((distance - 3.0) / 2.0) * 450  # Até 750 pontos
            elif distance < 7.0:
                base_reward = 750 + ((distance - 5.0) / 2.0) * 700  # Até 1450 pontos
            elif distance < 9.0:
                base_reward = 1450 + ((distance - 7.0) / 2.0) * 1550  # Até 3000 pontos
            else:  # 9m ou mais
                base_reward = 3000 + (distance - 9.0) * 200  # Bônus adicional

            # Bônus por eficiência mantido, mas aumentado
            efficiency_bonus = 0.0
            if distance > 1.0:
                target_steps = distance * 150  # Mais eficiente
                actual_steps = steps
                if actual_steps <= target_steps:
                    efficiency_bonus = 100  # Bônus fixo maior
                else:
                    efficiency_ratio = max(0.0, 1.0 - (actual_steps - target_steps) / target_steps)
                    efficiency_bonus = efficiency_ratio * 100

            # Bônus por estabilidade aumentado
            stability_bonus = 0.0
            if roll < 0.15 and pitch < 0.15:
                stability_bonus = 100
            elif roll < 0.20 and pitch < 0.20:
                stability_bonus = 50

            total_reward = base_reward + efficiency_bonus + stability_bonus
            return total_reward

        except Exception as e:
            self.logger.warning(f"Erro no sparse_success_component progressivo: {e}")
            return 0.0

    def _calculate_penalidades_component(self, sim, phase_info, terrain_params) -> float:
        """Componente de PENALIDADES: rigor físico + detecção de padrões viciados (abertura, jerk, lock incoerente)."""
        penalties = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            distance = getattr(sim, "episode_distance", 0.0)
            
            # --- Eficiência de propulsão ---
            left_hip_l = getattr(sim, "robot_left_hip_lateral_angle", 0.0)
            right_hip_l = getattr(sim, "robot_right_hip_lateral_angle", 0.0)
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            left_xv = getattr(sim, "robot_left_foot_x_velocity", 0.0)
            right_xv = getattr(sim, "robot_right_foot_x_velocity", 0.0)
            robot_xv = getattr(sim, "robot_velocity", [0,0,0])[0]
            vel_x = getattr(sim, "robot_x_velocity", 0.0)

            push_left = left_contact and (left_xv < robot_xv - 0.1)
            push_right = right_contact and (right_xv < robot_xv - 0.1)
        
            roll = abs(getattr(sim, "robot_roll", 0.0))
            pitch = abs(getattr(sim, "robot_pitch", 0.0))
            yaw_vel = abs(getattr(sim, "robot_orientation_velocity", [0,0,0])[2])
            ramp_up = current_terrain in ["ramp_up"]

            learning_progress = phase_info.get('learning_progress', 0)
            
            # Reduzir todas as penalidades pela metade nos estágios iniciais
            if learning_progress < 0.4:
                penalties *= 0.5
            elif learning_progress < 0.6:
                penalties *= 0.75

            # PENALIDADES ESPECÍFICAS
            if current_terrain == "low_friction":
                # Penaliza mais deslizamentos em baixo atrito
                lateral_velocity = abs(getattr(sim, "robot_y_velocity", 0.0))
                if lateral_velocity > 0.1:
                    penalties -= lateral_velocity * terrain_params.get("slip_penalty_multiplier", 2.0) * 30.0

            elif current_terrain == "ramp_down":
                # Penaliza velocidade excessiva em descida
                velocity = abs(getattr(sim, "robot_x_velocity", 0.0))
                if velocity > 0.8:
                    penalties -= (velocity - 0.8) * 40.0
            
            elif current_terrain == "ramp_up":
                abduction = max(abs(left_hip_l), abs(right_hip_l))
                if abduction > 0.2 and roll < 0.1:  
                    penalties -= (abduction - 0.2) * 40.0
                no_push = not (push_left or push_right)
                if no_push and vel_x > 0.05:  
                    penalties -= 15.0

            # === 1. Instabilidade angular AGRESSIVA (mantido, mas ajustado para rampa) ===
            pitch_target = 0.2 if ramp_up else 0.0
            pitch_err = abs(pitch - pitch_target)
            roll_penalty = max(0.0, roll - 0.15) * 40.0
            pitch_penalty = max(0.0, pitch_err - 0.15) * 30.0  
            yaw_penalty = yaw_vel * 25.0
            penalties -= (roll_penalty + pitch_penalty + yaw_penalty)

            # === 2. Movimento estagnado ou para trás ===
            distance = getattr(sim, "episode_distance", 0.0)
            episode_time = getattr(sim, "episode_time", 0.0)
            if distance < 0:
                penalties -= abs(distance) * 60.0
            elif distance < 0.03 and episode_time > 1.5:
                penalties -= 120.0

            # === 3. Abdução excessiva + incoerência postural (PRINCIPAL vício) ===
            left_hip_l = getattr(sim, "robot_left_hip_lateral_angle", 0.0)
            right_hip_l = getattr(sim, "robot_right_hip_lateral_angle", 0.0)
            y_vel = abs(getattr(sim, "robot_y_velocity", 0.0))
            # Combinação perigosa: abdução + movimento lateral
            abduction = max(abs(left_hip_l), abs(right_hip_l))
            if abduction > 0.2:
                abd_pen = (abduction - 0.2) * 25.0
                lateral_pen = y_vel * 20.0
                penalties -= (abd_pen + lateral_pen)

            # === 4. Jerk (aceleração angular súbita) — detecta spiking ===
            ang_vel = getattr(sim, "robot_orientation_velocity", [0,0,0])
            ang_speed = np.linalg.norm(ang_vel[:3])  
            if ang_speed > 2.0:  
                penalties -= (ang_speed - 2.0) * 15.0

            # === 5. Travamento articular incoerente (lock sem propósito) ===
            lock_timers = getattr(sim, "joint_lock_timers", [])
            if lock_timers:
                long_locks = sum(1 for t in lock_timers if t > 0.3)
                if long_locks > 2:  
                    penalties -= long_locks * 8.0

            # === 6. Ações extremas ou oscilatórias (spiking) ===
            action = phase_info.get('action')
            if action is not None:
                action = np.asarray(action)
                mag = np.linalg.norm(action)
                if mag > 2.5:
                    penalties -= (mag - 2.5) * 12.0
                # Detecta oscilação: diferença absoluta entre ações consecutivas
                if hasattr(self, '_prev_action') and self._prev_action is not None:
                    diff = np.linalg.norm(action - self._prev_action)
                    if diff > 3.0:  
                        penalties -= diff * 2.0
                self._prev_action = action.copy()

            # === 7. Padrão de “pernas abertas + tronco torto” (gambá/bailarino) ===
            left_foot_roll = abs(getattr(sim, "robot_left_foot_roll", 0.0))
            right_foot_roll = abs(getattr(sim, "robot_right_foot_roll", 0.0))
            foot_roll = left_foot_roll + right_foot_roll
            if abduction > 0.25 and roll > 0.2 and foot_roll > 0.4:
                penalties -= 40.0  

            # === 8. Penalidade por esforço (mantida, mas normalizada) ===
            effort = self._calculate_effort_penalty(sim, phase_info)
            penalties += effort

            # === 9. Queda / término crítico (mantido) ===
            termination = getattr(sim, "episode_termination", "")
            if termination == "fell":
                penalties -= 300.0
            elif termination == "yaw_deviated":
                penalties -= 200.0

        except Exception as e:
            self.logger.warning(f"Erro em penalidades (anti-vício): {e}")
        
        # Reduzir impacto das penalidades em longas distâncias
        penalty_reduction_factor = 1.0
        if distance > 3.0:
            penalty_reduction_factor = max(0.3, 1.0 - (distance - 3.0) / 6.0)  # Reduz gradualmente até 9m

        # Aplicar fator de redução no final
        total_penalties = penalties * penalty_reduction_factor
        return total_penalties

    def _calculate_effort_penalty(self, sim, phase_info) -> float:
        """Penalidade por esforço (auxiliar para eficiência)"""
        try:
            if hasattr(sim, 'joint_velocities'):
                effort = sum(v**2 for v in sim.joint_velocities)
                return effort * -0.001
            return 0.0
        except:
            return 0.0
    
    def get_cache_stats(self) -> Dict:
        """Estatísticas de cache """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0
        }
    
    def _generate_essential_cache_key(self, sim, phase_info: Dict) -> str:
        """Gera chave de cache estável baseada em métricas essenciais"""
        try:
            # Métricas essenciais para cache 
            essential_metrics = {}

            # Distância (arredondada)
            distance = getattr(sim, "episode_distance", 0)
            if distance > 1.0: 
                return f"high_progress_{int(distance * 10)}_{hash(str(phase_info))}"
            essential_metrics["dist"] = round(distance, 2)  

            # Velocidade (arredondada)
            velocity = getattr(sim, "robot_x_velocity", 0)
            essential_metrics["vel"] = round(velocity, 2)

            # Estabilidade (roll e pitch)
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            essential_metrics["stab"] = round(roll + pitch, 2)

            # Contatos dos pés
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            essential_metrics["contacts"] = f"{left_contact}_{right_contact}"

            # Info da fase (simplificada)
            phase_hash = hash(str(phase_info.get('group_level', 1)))

            # Combinar tudo em uma chave única mas estável
            cache_key = f"reward_{essential_metrics['dist']}_{essential_metrics['vel']}_{essential_metrics['stab']}_{essential_metrics['contacts']}_{phase_hash}"

            return cache_key

        except Exception as e:
            # Fallback em caso de erro
            self.logger.warning(f"Erro ao gerar chave de cache: {e}")
            return f"reward_fallback_{hash(str(sim))}_{time.time()}"