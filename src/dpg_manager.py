# dpg_manager.py
import numpy as np
from collections import deque
import random
from typing import List, Dict, Any
import time
from dataclasses import dataclass

class Experience:
    def __init__(self, state, action, reward, next_state, done, metrics):
        self.state = np.array(state, dtype=np.float32)
        self.action = np.array(action, dtype=np.float32)
        self.reward = float(reward)
        self.next_state = np.array(next_state, dtype=np.float32)
        self.done = done
        self.quality = self._calculate_quality(metrics)
        
    def _calculate_quality(self, metrics):
        """Qualidade simples baseada em distância e estabilidade"""
        distance = max(metrics.get("distance", 0), 0)
        stability = 1.0 - min((abs(metrics.get("roll", 0)) + abs(metrics.get("pitch", 0))) / 1.0, 1.0)
        
        if distance <= 0:
            return 0.0
            
        return min(distance / 2.0, 0.6) + stability * 0.4
        

class DPGManager:
    def __init__(self, logger, robot, reward_system):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.enabled = True
        self.learning_enabled = True
        
        # Sistemas principais
        self.buffer = SimpleBuffer(capacity=3000)
        self.reward_calculator = RewardCalculator(logger, {})  # ADICIONADO: Inicializar RewardCalculator
        
        # Estado simples
        self.episode_count = 0
        self.performance_history = deque(maxlen=50)
        self.learning_progress = 0.0
        self.current_terrain = "normal"  # ADICIONADO: Definir terreno atual

        # Controle de treinamento
        self.training_interval = 10  # Treinar a cada 10 episódios
        self.min_buffer_size = 100   # Tamanho mínimo do buffer para treinar
        self._last_training_episode = 0

    def calculate_reward(self, sim, action) -> float:
        if not self.enabled:
            return 0.0
            
        # Info básica para recompensa
        phase_info = {
            'current_terrain': self.current_terrain,  # CORRIGIDO: usar self.current_terrain
            'learning_progress': self.learning_progress
        }
        
        reward = self.reward_calculator.calculate(sim, action, phase_info)
        self.performance_history.append(reward)
        
        return max(reward, 0.0)

    def set_current_terrain(self, terrain):  # ADICIONADO: Método para definir terreno
        """Define o terreno atual para cálculos de recompensa"""
        self.current_terrain = terrain

    def store_experience(self, state, action, reward, next_state, done, episode_results):
        if not self.enabled:
            return False

        try:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metrics=episode_results
            )
            
            success = self.buffer.add(experience)
            return success
            
        except Exception as e:
            self.logger.error(f"Erro ao armazenar experiência: {e}")
            return False

    def update_phase_progression(self, episode_results):
        """Atualiza progresso baseado em desempenho recente"""
        self.episode_count += 1
        
        # Progresso baseado em distância e estabilidade
        distance = episode_results.get("distance", 0)
        stability = 1.0 - min((abs(episode_results.get("roll", 0)) + 
                              abs(episode_results.get("pitch", 0))) / 1.0, 1.0)
        
        # Progresso simples: 50% distância, 50% estabilidade
        distance_progress = min(distance / 5.0, 1.0) 
        stability_progress = stability
        
        self.learning_progress = (distance_progress * 0.5 + stability_progress * 0.5)
        
        # Limpeza periódica do buffer
        if self.episode_count % 100 == 0 and len(self.buffer) > 2000:
            self._cleanup_buffer()

    def _cleanup_buffer(self):
        """Remove experiências de baixa qualidade periodicamente"""
        if len(self.buffer.buffer) > 2000:
            # Mantém apenas as melhores 2000 experiências
            sorted_experiences = sorted(self.buffer.buffer, key=lambda x: x.quality, reverse=True)
            self.buffer.buffer = deque(sorted_experiences[:2000], maxlen=self.buffer.capacity)

    def should_train(self, current_episode: int) -> bool:
        """Decide se deve treinar neste episódio"""
        if not self.learning_enabled:
            return False
            
        buffer_ready = len(self.buffer) >= self.min_buffer_size
        interval_ok = (current_episode - self._last_training_episode) >= self.training_interval
        
        return buffer_ready and interval_ok

    def on_training_completed(self, episode: int):
        """Callback quando o treinamento é completado"""
        self._last_training_episode = episode

    def get_training_batch(self, batch_size=32) -> List[Experience]:
        return self.buffer.sample(batch_size)

    def get_integrated_status(self):
        return {
            "enabled": self.enabled,
            "episode_count": self.episode_count,
            "buffer_size": len(self.buffer),
            "learning_progress": self.learning_progress,
            "avg_recent_reward": np.mean(list(self.performance_history)) if self.performance_history else 0
        }


@dataclass
class TerrainParams:
    speed_weight: float = 25.0
    stability_weight: float = 1.2
    coordination_bonus: float = 15.0
    clearance_min: float = 0.05
    push_phase_bonus: float = 1.0
    uphill_bonus: float = 1.0
    stance_knee_lock_bonus: float = 1.0
    adaptation_bonus: float = 1.0
    robustness_bonus: float = 1.0
    pitch_target: float = 0.0
    height_target: float = 0.8
    efficiency_weight: float = 1.0


class RewardCalculator:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        
        # Componentes principais simplificados
        self.components = {
            "progresso": 1.5,
            "coordenacao": 1.2, 
            "estabilidade": 1.3,
            "eficiencia": 0.8,
            "penalidades": -1.0 
        }
        
        # Parâmetros por terreno (mantido para adaptação)
        self.terrain_params = {
            "normal": TerrainParams(),
            "ramp_up": TerrainParams(speed_weight=15.0, stability_weight=1.5, clearance_min=0.08, pitch_target=0.2),
            "ramp_down": TerrainParams(speed_weight=12.0, stability_weight=1.6, clearance_min=0.08, pitch_target=-0.1),
            "uneven": TerrainParams(stability_weight=1.4, coordination_bonus=20.0)
        }

    def calculate(self, sim, action, phase_info: Dict) -> float:
        current_terrain = phase_info.get('current_terrain', 'normal')
        terrain_params = self.terrain_params.get(current_terrain, self.terrain_params["normal"])

        # Cálculo direto sem cache
        component_values = {
            "progresso": self._calculate_progresso(sim, phase_info, terrain_params),
            "coordenacao": self._calculate_coordenacao(sim, phase_info, terrain_params),
            "estabilidade": self._calculate_estabilidade(sim, phase_info, terrain_params),
            "eficiencia": self._calculate_eficiencia(sim, phase_info, terrain_params),
            "penalidades": self._calculate_penalidades_component(sim, phase_info, terrain_params)  
        }

        # Soma ponderada simples
        total_reward = 0.0
        for component, value in component_values.items():
            total_reward += value * self.components[component]

        return max(total_reward, 0.0)

    def _calculate_progresso(self, sim, phase_info, terrain_params) -> float:
        """PROGRESSO: recompensa distância *apenas* se marcha for funcional e estável."""
        reward = 0.0
        try:
            dist = max(getattr(sim, "episode_distance", 0.0), 0.0)
            vel_x = getattr(sim, "robot_x_velocity", 0.0)
            current_terrain = phase_info.get('current_terrain', 'normal')

            # USA PARÂMETROS DO TERRENO CORRETAMENTE
            speed_weight = terrain_params.speed_weight  
            clearance_min = terrain_params.clearance_min
            push_phase_bonus = terrain_params.push_phase_bonus
            uphill_bonus = terrain_params.uphill_bonus

            # Estabilidade e funcionalidade mínimas
            roll = abs(getattr(sim, "robot_roll", 0.0))
            pitch = getattr(sim, "robot_pitch", 0.0)

            # PITCH TARGET ESPECÍFICO POR TERRENO
            pitch_target = terrain_params.pitch_target
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
            robot_xv = getattr(sim, "robot_x_velocity", 0.0)  
            
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

            # Bônus para sequências de terrenos consecutivos
            terrain_sequence = phase_info.get('terrain_sequence', [])
            if len(terrain_sequence) >= 2:
                # Bônus crescente por cada terreno completado
                sequence_bonus = min(len(terrain_sequence) * 25.0, 150.0)
                reward += sequence_bonus
    
            # USA CLEARANCE_MIN DO TERRENO 
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
                long_distance_bonus = (dist - 3.0) * 80.0  
                reward += long_distance_bonus
                # Marcos significativos
                if dist >= 5.0:
                    reward += 200
                if dist >= 7.0:
                    reward += 400
                if dist >= 8.5:
                    reward += 800

        except Exception as e:
            self.logger.warning(f"Erro em progresso (terrain-corrected): {e}")
        return reward

    def _calculate_coordenacao(self, sim, phase_info, terrain_params) -> float:
        """COORDENAÇÃO: força padrão de marcha funcional."""
        reward = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            coordination_bonus = terrain_params.coordination_bonus
            clearance_min = terrain_params.clearance_min
            stance_knee_lock_bonus = terrain_params.stance_knee_lock_bonus
            adaptation_bonus = terrain_params.adaptation_bonus
            robustness_bonus = terrain_params.robustness_bonus

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
            robot_xv = getattr(sim, "robot_x_velocity", 0.0)

            if current_terrain == "ramp_up":
                # Bônus por travamento funcional do joelho em stance
                left_knee_lock_ok = left_contact and abs(left_knee) < 0.2  
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

    def _calculate_estabilidade(self, sim, phase_info, terrain_params) -> float:
        """ESTABILIDADE: reforça postura, penaliza abertura de pernas."""
        reward = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            stability_weight = terrain_params.stability_weight

            # ALVOS ESPECÍFICOS POR TERRENO
            pitch_target = terrain_params.pitch_target
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
            height_target = terrain_params.height_target
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

    def _calculate_eficiencia(self, sim, phase_info, terrain_params) -> float:
        """EFICIÊNCIA: prioriza flexão ativa, penaliza esforço em abdução."""
        reward = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            efficiency_weight = terrain_params.efficiency_weight

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
            robot_xv = getattr(sim, "robot_x_velocity", 0.0)

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

    def _calculate_penalidades_component(self, sim, phase_info, terrain_params) -> float:
        """Componente de PENALIDADES: rigor físico + detecção de padrões viciados"""
        penalties = 0.0
        try:
            current_terrain = phase_info.get('current_terrain', 'normal')
            distance = getattr(sim, "episode_distance", 0.0)
            
            # Dados necessários para cálculos
            left_hip_l = getattr(sim, "robot_left_hip_lateral_angle", 0.0)
            right_hip_l = getattr(sim, "robot_right_hip_lateral_angle", 0.0)
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            left_xv = getattr(sim, "robot_left_foot_x_velocity", 0.0)
            right_xv = getattr(sim, "robot_right_foot_x_velocity", 0.0)
            robot_xv = getattr(sim, "robot_x_velocity", 0.0)
            vel_x = getattr(sim, "robot_x_velocity", 0.0)

            push_left = left_contact and (left_xv < robot_xv - 0.1)
            push_right = right_contact and (right_xv < robot_xv - 0.1)
        
            roll = abs(getattr(sim, "robot_roll", 0.0))
            pitch = getattr(sim, "robot_pitch", 0.0)
            orientation_velocity = getattr(sim, "robot_orientation_velocity", [0,0,0])
            yaw_vel = abs(orientation_velocity[2]) if len(orientation_velocity) > 2 else 0.0
            ramp_up = current_terrain in ["ramp_up"]

            learning_progress = phase_info.get('learning_progress', 0)

            # === 1. Instabilidade angular AGRESSIVA ===
            pitch_target = 0.2 if ramp_up else 0.0
            pitch_err = abs(pitch - pitch_target)
            roll_penalty = max(0.0, roll - 0.15) * 40.0
            pitch_penalty = max(0.0, pitch_err - 0.15) * 30.0  
            yaw_penalty = yaw_vel * 25.0
            penalties -= (roll_penalty + pitch_penalty + yaw_penalty)

            # === 2. Movimento estagnado ou para trás ===
            episode_steps = getattr(sim, "episode_steps", 0)
            time_step_s = getattr(sim, "time_step_s", 0.033)
            episode_time = episode_steps * time_step_s
            
            if distance < 0:
                penalties -= abs(distance) * 60.0
            elif distance < 0.03 and episode_time > 1.5:
                penalties -= 120.0

            # === 3. Abdução excessiva + incoerência postural ===
            abduction = max(abs(left_hip_l), abs(right_hip_l))
            y_vel = abs(getattr(sim, "robot_y_velocity", 0.0))
            
            if abduction > 0.2:
                abd_pen = (abduction - 0.2) * 25.0
                lateral_pen = y_vel * 20.0
                penalties -= (abd_pen + lateral_pen)

            # === 4. Jerk (aceleração angular súbita) ===
            ang_vel = getattr(sim, "robot_orientation_velocity", [0,0,0])
            ang_speed = np.linalg.norm(ang_vel[:2])  
            if ang_speed > 2.0:  
                penalties -= (ang_speed - 2.0) * 15.0

            # === 5. Travamento articular incoerente ===
            lock_timers = getattr(sim, "joint_lock_timers", [])
            if lock_timers:
                long_locks = sum(1 for t in lock_timers if t > 10)  
                if long_locks > 2:  
                    penalties -= long_locks * 8.0

            # === 6. Ações extremas ou oscilatórias ===
            action = phase_info.get('action', [])
            if action is not None and len(action) > 0:
                action = np.asarray(action)
                mag = np.linalg.norm(action)
                if mag > 2.5:
                    penalties -= (mag - 2.5) * 12.0
                
                # Detecta oscilação
                if hasattr(self, '_prev_action') and self._prev_action is not None:
                    diff = np.linalg.norm(action - self._prev_action)
                    if diff > 3.0:  
                        penalties -= diff * 2.0
                self._prev_action = action.copy()

            # === 7. Padrão de "pernas abertas + tronco torto" ===
            left_foot_roll = abs(getattr(sim, "robot_left_foot_roll", 0.0))
            right_foot_roll = abs(getattr(sim, "robot_right_foot_roll", 0.0))
            foot_roll = left_foot_roll + right_foot_roll
            if abduction > 0.25 and roll > 0.2 and foot_roll > 0.4:
                penalties -= 40.0  

            # === 8. Penalidade por esforço ===
            if hasattr(sim, 'joint_velocities'):
                effort = sum(v**2 for v in sim.joint_velocities) * 0.01
                penalties -= effort

            # === 9. Queda / término crítico ===
            termination = getattr(sim, "episode_termination", "")
            if termination == "fell":
                penalties -= 300.0
            elif termination == "yaw_deviated":
                penalties -= 200.0

            # Reduzir impacto baseado no progresso de aprendizado
            if learning_progress < 0.4:
                penalties *= 0.5
            elif learning_progress < 0.6:
                penalties *= 0.75
            
            # Reduzir impacto em longas distâncias
            penalty_reduction_factor = 1.0
            if distance > 3.0:
                penalty_reduction_factor = max(0.4, 1.0 - (distance - 3.0) / 6.0)
            elif distance > 1.0:
                penalty_reduction_factor = max(0.7, 1.0 - (distance - 1.0) / 4.0)
            penalties *= penalty_reduction_factor
            
        except Exception as e:
            self.logger.warning(f"Erro em penalidades (anti-vício): {e}")
            penalties = 0.0  
        
        return penalties


class SimpleBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.quality_threshold = 0.2
        
    def add(self, experience):
        """Adiciona experiência se atender qualidade mínima"""
        if experience.quality >= self.quality_threshold:
            self.buffer.append(experience)
            return True
        return False
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Amostragem balanceada: 70% aleatória, 30% de alta qualidade"""
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
            
        # Separa experiências por qualidade
        high_quality = [exp for exp in self.buffer if exp.quality > 0.5]
        regular = [exp for exp in self.buffer if exp.quality <= 0.5]
        
        # Calcula quantidades
        high_quality_count = min(int(batch_size * 0.3), len(high_quality))
        regular_count = batch_size - high_quality_count
        
        # Amostra
        samples = []
        if high_quality_count > 0:
            samples.extend(random.sample(high_quality, high_quality_count))
        if regular_count > 0:
            samples.extend(random.sample(regular, regular_count))
            
        return samples
    
    def __len__(self):
        return len(self.buffer)