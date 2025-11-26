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
        self.reward_calculator = RewardCalculator(logger, {})  
        
        # Estado simples
        self.episode_count = 0
        self.performance_history = deque(maxlen=50)
        self.learning_progress = 0.0
        self.current_terrain = "normal"  

        # Controle de treinamento
        self.training_interval = 5  
        self.min_buffer_size = 200  
        self.learning_rate_factor = 1.0

        # Histórico para adaptação
        self.recent_success_rate = deque(maxlen=100)
        self.consecutive_failures = 0

        # Variávels para resumo
        self._last_training_episode = 0
        self.last_report_episode = 0
        self.episode_metrics_history = deque(maxlen=50)
        self.terrain_performance = {}

    def calculate_reward(self, sim, action) -> float:
        if not self.enabled:
            return 0.0
            
        # Info básica para recompensa
        phase_info = {
            'current_terrain': self.current_terrain,  
            'learning_progress': self.learning_progress
        }
        
        reward = self.reward_calculator.calculate(sim, action, phase_info)
        self.performance_history.append(reward)
        
        return reward

    def set_current_terrain(self, terrain):  
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
        
        # Garantir que a distância não seja negativa
        distance = max(episode_results.get("distance", 0), 0)
        
        # Cálculo de estabilidade
        roll = abs(episode_results.get("roll", 0))
        pitch = abs(episode_results.get("pitch", 0))
        stability = max(0.0, 1.0 - min((roll + pitch) / 1.0, 1.0))
        
        # Progresso simples
        distance_progress = min(distance / 9.0, 1.0) 
        stability_progress = stability
        
        self.learning_progress = max(0.0, (distance_progress * 0.5 + stability_progress * 0.5))  # Garantir não negativo
        
        # Atualizar histórico de métricas
        self.episode_metrics_history.append({
            "reward": episode_results.get("reward", 0),
            "distance": distance,
            "success": distance > 0.5 and stability > 0.6
        })
        
        # Atualizar desempenho por terreno
        terrain = self.current_terrain
        if terrain not in self.terrain_performance:
            self.terrain_performance[terrain] = {
                'episodes': 0,
                'total_distance': 0,
                'total_reward': 0,
                'successes': 0
            }
        
        self.terrain_performance[terrain]['episodes'] += 1
        self.terrain_performance[terrain]['total_distance'] += distance
        self.terrain_performance[terrain]['total_reward'] += episode_results.get("reward", 0)
        if distance > 0.5 and stability > 0.6:
            self.terrain_performance[terrain]['successes'] += 1
        
        # Gerar relatório a cada 500 episódios
        if self.episode_count - self.last_report_episode >= 500:
            self._generate_comprehensive_report()
            self.last_report_episode = self.episode_count
            
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
    
    def _get_buffer_status(self):
        """Status simplificado do buffer"""
        if len(self.buffer) == 0:
            return {
                "total_experiences": 0,
                "avg_quality": 0,
                "buffer_utilization": 0,
                "active_experiences": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0}
            }
        
        buffer_list = list(self.buffer.buffer) if hasattr(self.buffer, 'buffer') else list(self.buffer)
        qualities = [exp.quality for exp in buffer_list]
        
        if not qualities:  
            return {
                "total_experiences": 0,
                "avg_quality": 0,
                "buffer_utilization": 0,
                "active_experiences": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0}
            }
        
        high_quality = sum(1 for q in qualities if q > 0.7)
        medium_quality = sum(1 for q in qualities if 0.3 <= q <= 0.7)
        low_quality = sum(1 for q in qualities if q < 0.3)
        
        return {
            "total_experiences": len(buffer_list),
            "avg_quality": np.mean(qualities),
            "buffer_utilization": len(buffer_list) / self.buffer.capacity,
            "active_experiences": len(buffer_list),
            "quality_distribution": {
                "high": high_quality,
                "medium": medium_quality,
                "low": low_quality
            }
        }

    def _generate_comprehensive_report(self):
        """RELATÓRIO a cada 500 episódios"""
        try:
            # Dados básicos do sistema
            integrated_status = self.get_integrated_status()
            buffer_status = self._get_buffer_status()
            
            # Estatísticas recentes (últimos 50 episódios)
            recent_episodes = list(self.episode_metrics_history)
            if recent_episodes:
                recent_rewards = [ep['reward'] for ep in recent_episodes]
                recent_distances = [ep['distance'] for ep in recent_episodes]
                recent_successes = sum(1 for ep in recent_episodes if ep['success'])
                
                avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_recent_distance = np.mean(recent_distances) if recent_distances else 0
                success_rate = recent_successes / len(recent_episodes) if recent_episodes else 0
            else:
                avg_recent_reward = 0
                avg_recent_distance = 0
                success_rate = 0

            # RELATÓRIO 
            self.logger.info("=" * 60)
            self.logger.info(f"📊 RELATÓRIO - Episódio {self.episode_count}")
            
            # Status principal
            self.logger.info(f"Progresso: {self.learning_progress:.1%} | "
                           f"Recompensa: {avg_recent_reward:.1f} | "
                           f"Sucesso: {success_rate:.1%}")
            
            # Buffer
            self.logger.info(f"Buffer: {len(self.buffer)} exp | "
                           f"Qualidade: {buffer_status['avg_quality']:.3f}")
            
            # Terrenos
            self.logger.info("Terrenos:")
            if hasattr(self, 'terrain_performance') and self.terrain_performance:
                for terrain, stats in self.terrain_performance.items():
                    if stats['episodes'] > 0:
                        avg_dist = stats['total_distance'] / stats['episodes']
                        success_rate_terrain = stats['successes'] / stats['episodes']
                        self.logger.info(f"  {terrain}: {avg_dist:.2f}m, {success_rate_terrain:.1%} sucesso")
            else:
                self.logger.info("  Nenhum dado de terreno disponível")
            
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Erro no relatório: {e}")


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
            "progresso": 2.0,
            "coordenacao": 1.8, 
            "estabilidade": 1.4,
            "eficiencia": 0.5,
            "penalidades": -0.3 
        }
        
        # Parâmetros por terreno (mantido para adaptação)
        self.terrain_params = {
            "normal": TerrainParams(speed_weight=30.0, stability_weight=1.0),
            "ramp_up": TerrainParams(speed_weight=20.0, stability_weight=1.3, clearance_min=0.06),
            "ramp_down": TerrainParams(speed_weight=18.0, stability_weight=1.4, clearance_min=0.06),
            "uneven": TerrainParams(stability_weight=1.2, coordination_bonus=15.0)
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

        return total_reward

    def _calculate_progresso(self, sim, phase_info, terrain_params) -> float:
        """PROGRESSO com recompensas intermediárias"""
        reward = 0.0
        try:
            dist = max(getattr(sim, "episode_distance", 0.0), 0.0)
            vel_x = getattr(sim, "robot_x_velocity", 0.0)
            
            # Recompensas progressivas mais agressivas
            speed_bonus = 0.0
            if vel_x > 0.1:
                speed_bonus = min(vel_x * 20.0, 50.0)

            # Bônus por distância *consistente*
            dist = max(getattr(sim, "episode_distance", 0.0), 0.0)
            consistency_bonus = 0.0
            consecutive = getattr(sim, "consecutive_alternating_steps", 0)
            if dist > 0.5 and consecutive >= 3:
                consistency_bonus = min(dist * 5.0, 70.0)

            # Bônus final por chegar perto do alvo 
            near_target_bonus = 0.0
            if dist >= 7.0:
                near_target_bonus = (dist - 7.0) * 100.0 

            # Bônus de sobrevivência com progresso
            if not getattr(sim, "episode_terminated", True) and dist > 0.1:
                survival_bonus = min(dist * 10.0, 50.0)

            reward = speed_bonus + consistency_bonus + near_target_bonus + survival_bonus

                
        except Exception as e:
            self.logger.warning(f"Erro em progresso: {e}")
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
        return reward

    def _calculate_penalidades_component(self, sim, phase_info, terrain_params) -> float:
        """PENALIDADES otimizadas menos agressivas no início"""
        penalties = 0.0
        try:
            learning_progress = phase_info.get('learning_progress', 0)
            distance = getattr(sim, "episode_distance", 0.0)
            
            # Penalidades progressivas baseadas no aprendizado
            penalty_multiplier = max(0.1, learning_progress)
            
            # Reduzir penalidades no início do aprendizado
            if learning_progress < 0.2:
                # Só penaliza se for grave
                if roll > 0.5:   
                    penalties -= (roll - 0.5) * 15.0 * penalty_multiplier
                if abs(pitch) > 0.6:  
                    penalties -= (abs(pitch) - 0.6) * 12.0 * penalty_multiplier
            else:
                # Comportamento padrão 
                if roll > 0.3:
                    penalties -= (roll - 0.3) * 15.0 * penalty_multiplier
                if abs(pitch) > 0.4:
                    penalties -= (abs(pitch) - 0.4) * 12.0 * penalty_multiplier
            
            # Penalidades principais 
            roll = abs(getattr(sim, "robot_roll", 0.0))
            pitch = abs(getattr(sim, "robot_pitch", 0.0))
            
            # Instabilidade angular 
            if roll > 0.3:
                penalties -= (roll - 0.3) * 15.0 * penalty_multiplier
            if abs(pitch) > 0.4:
                penalties -= (abs(pitch) - 0.4) * 12.0 * penalty_multiplier
            
            # Movimento para trás 
            if distance < -0.1:
                penalties -= abs(distance) * 20.0 * penalty_multiplier
            elif distance < 0.05 and getattr(sim, "episode_steps", 0) > 20:
                penalties -= 30.0 * penalty_multiplier  
                
            # Queda 
            termination = getattr(sim, "episode_termination", "")
            if termination == "fell":
                penalties -= 200.0
            elif termination == "yaw_deviated":
                penalties -= 150.0
                
        except Exception as e:
            self.logger.warning(f"Erro em penalidades: {e}")
            
        return penalties


class SimpleBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.quality_threshold = 0.3
        
    def add(self, experience):
        """Adiciona experiência se atender qualidade mínima"""
        if experience.quality >= self.quality_threshold:
            self.buffer.append(experience)
            return True
        return False
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Amostragem balanceada com foco em experiências educativas"""
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
            
        # Estratificação por qualidade
        high_quality = [exp for exp in self.buffer if exp.quality > 0.7]
        medium_quality = [exp for exp in self.buffer if 0.3 <= exp.quality <= 0.7]
        low_quality = [exp for exp in self.buffer if exp.quality < 0.3]
        
        # Proporções otimizadas
        high_count = min(int(batch_size * 0.4), len(high_quality)) 
        medium_count = min(int(batch_size * 0.4), len(medium_quality))
        low_count = batch_size - high_count - medium_count
        
        samples = []
        if high_count > 0:
            samples.extend(random.sample(high_quality, high_count))
        if medium_count > 0:
            samples.extend(random.sample(medium_quality, medium_count))
        if low_count > 0 and len(low_quality) > 0:
            samples.extend(random.sample(low_quality, min(low_count, len(low_quality))))
            
        # Preencher com amostras aleatórias se necessário
        if len(samples) < batch_size:
            additional = random.sample(self.buffer, batch_size - len(samples))
            samples.extend(additional)
            
        return samples
    
    def __len__(self):
        return len(self.buffer)