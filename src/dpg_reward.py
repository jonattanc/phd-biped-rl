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
            "progresso": 1.0,
            "coordenacao": 1.0, 
            "estabilidade": 1.0,
            "eficiencia": 1.0,
            "valencia_bonus": 1.0,
            "penalidades": 1.0
        }
        
        self.base_macro_weights = self.macro_components.copy()
        self.weight_adjustment_rate = 0.002  
        self.max_weight_change = 0.4  
        
        # Sistema de cache unificado
        self.cache = Cache(max_size=500)
        self._last_sim_state = None
        self._last_reward = 0.0
        
        # Estatísticas
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0

        self.macro_components["sparse_success"] = 0.0  
        self.base_macro_weights["sparse_success"] = 0.0
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
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
            "progresso": 0.0,
            "coordenacao": 0.0,
            "estabilidade": 0.0, 
            "eficiencia": 0.0,
            "valencia_bonus": 0.0,
            "penalidades": 0.0
        }

        # 1. PROGRESSO (movimento, velocidade, distância)
        component_values["progresso"] = self._calculate_progresso_component(sim, phase_info)
        
        # 2. COORDENAÇÃO (marcha alternada, padrão de gait)
        component_values["coordenacao"] = self._calculate_coordenacao_component(sim, phase_info)
        
        # 3. ESTABILIDADE (roll, pitch, altura COM, equilíbrio)
        component_values["estabilidade"] = self._calculate_estabilidade_component(sim, phase_info)
        
        # 4. EFICIÊNCIA (energia, biomecânica, suavidade)
        component_values["eficiencia"] = self._calculate_eficiencia_component(sim, phase_info)
        
        # 5. BÔNUS DE VALÊNCIAS
        component_values["valencia_bonus"] = self._calculate_valencia_bonus_component(sim, phase_info)
        
        # 6. PENALIDADES (queda, desvios, esforço excessivo)
        component_values["penalidades"] = self._calculate_penalidades_component(sim, phase_info)

        # 7. SPARSE SUCCESS 
        component_values["sparse_success"] = self._calculate_sparse_success_component(sim, phase_info)
        
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
    
    def _calculate_progresso_component(self, sim, phase_info) -> float:
        """Componente de PROGRESSO: só recompensa movimento *estável e efetivo*"""
        reward = 0.0
        try:
            distance = max(getattr(sim, "episode_distance", 0), 0)
            velocity = getattr(sim, "robot_x_velocity", 0)
            episode_time = getattr(sim, "episode_time", 0)
            
            # Bônus por distância inicial
            if 0.1 <= distance < 0.3:
                reward += 20 * (distance / 0.3)
            
            # Bônus por distância real 
            if distance > 0:
                reward += distance * 30.0  

            # Marcos de distância 
            milestones = {
                3.0: 180.0, 2.0: 120.0, 1.5: 70.0,
                1.0: 50.0, 0.7: 35.0, 0.5: 18.0,
                0.3: 8.0, 0.1: 2.0, 0.05: 1.0
            }
            for d, b in milestones.items():
                if distance >= d:
                    reward += b
                    break

            # Só recompensa velocidade se houver estabilidade E progresso
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            stability = 1.0 - min((roll + pitch) / 0.3, 1.0)  

            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            alternating = (left_contact != right_contact)

            effective_velocity_bonus = 0.0
            terrain = getattr(sim, "terrain_type", "normal")
            vel_mult = 18.0 if terrain == "ramp" else 12.0

            if velocity > 0 and stability > 0.6 and alternating:
                effective_velocity_bonus = velocity * vel_mult
            elif velocity > 0 and stability > 0.6:
                effective_velocity_bonus = velocity * 4.0
            elif velocity > 0:
                effective_velocity_bonus = velocity * 1.0

            reward += effective_velocity_bonus

            # Bônus de sobrevivência só se avançou > 2 cm
            if not getattr(sim, "episode_terminated", True) and distance > 0.02:
                reward += 30.0  

        except Exception as e:
            self.logger.warning(f"Erro no cálculo de progresso: {e}")
        return reward
    
    def _calculate_coordenacao_component(self, sim, phase_info) -> float:
        """Componente de COORDENAÇÃO: só recompensa padrão de marcha *funcional*"""
        reward = 0.0
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            alternating = (left_contact != right_contact)

            consecutive_alternating = getattr(sim, "consecutive_alternating_steps", 0)
            coord_duration_bonus = min(consecutive_alternating / 10.0, 1.0)

            left_knee = getattr(sim, "robot_left_knee_angle", 0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0)

            left_foot_h = getattr(sim, "robot_left_foot_height", 0)
            right_foot_h = getattr(sim, "robot_right_foot_height", 0)

            gait_phase = getattr(sim, "gait_phase", 0.5)
            left_swing = not left_contact and gait_phase < 0.5
            right_swing = not right_contact and gait_phase > 0.5

            clearance_ok = (
                (not left_contact and left_foot_h > 0.06) or
                (not right_contact and right_foot_h > 0.06)
            )
            knee_flex_ok = (
                (not left_contact and left_knee > 0.7) or
                (not right_contact and right_knee > 0.7)
            )

            # Base: alternância correta
            if alternating:
                reward += 12.0
            else:
                reward -= 8.0  

            if alternating and consecutive_alternating >= 5:  
                reward += 15.0 * coord_duration_bonus

            # Bônus por clearance funcional
            if clearance_ok:
                reward += 6.0

            # Bônus por flexão de joelho no swing
            if knee_flex_ok:
                reward += 5.0

            # BÔNUS ESTRUTURAL: só se todos os 3 estiverem OK
            if (left_swing or right_swing) and knee_flex_ok > 0.7 and clearance_ok > 0.06:
                reward += 12.0 * (1.0 - abs(gait_phase - 0.5))
    
            # Gait pattern score (se disponível)
            gait_score = getattr(sim, "robot_gait_pattern_score", 0.0)
            if gait_score > 0.5:
                reward += gait_score * 20.0

        except Exception as e:
            self.logger.warning(f"Erro no cálculo de coordenação: {e}")
        return reward
    
    def _calculate_estabilidade_component(self, sim, phase_info) -> float:
        """Componente de ESTABILIDADE: equilíbrio, postura, controle"""
        reward = 0.0
        
        try:
            # Estabilidade angular
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            stability = 1.0 - min((roll + pitch) / 1.0, 1.0)
            reward += stability * 20.0
            
            # Altura adequada do COM
            com_height = getattr(sim, "robot_z_position", 0.8)
            if 0.7 <= com_height <= 0.9:
                reward += 15.0
            elif 0.6 <= com_height < 0.7 or 0.9 < com_height <= 1.0:
                reward += 8.0
                
            # Estabilidade lateral
            y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
            lateral_stability = 1.0 - min(y_velocity / 0.3, 1.0)
            reward += lateral_stability * 10.0
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de estabilidade: {e}")
            
        return reward
    
    def _calculate_eficiencia_component(self, sim, phase_info) -> float:
        """Componente de EFICIÊNCIA: energia, biomecânica, suavidade"""
        reward = 0.0
        
        try:
            # Eficiência de propulsão
            propulsion_efficiency = getattr(sim, "robot_propulsion_efficiency", 0.5)
            reward += propulsion_efficiency * 12.0
            
            # Eficiência energética (inversa do esforço)
            effort_penalty = abs(self._calculate_effort_penalty(sim, phase_info))
            energy_efficiency = 1.0 - min(effort_penalty / 10.0, 1.0)  
            reward += energy_efficiency * 15.0
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de eficiência: {e}")
            
        return reward
    
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
                        
        except Exception as e:
            self.logger.warning(f"Erro em bônus de valência: {e}")
            
        return bonus
    
    def _calculate_sparse_success_component(self, sim, phase_info) -> float:
        """Recompensa esparsa progressiva"""
        if not phase_info.get('sparse_success_enabled', False):
            return 0.0

        try:
            distance = max(getattr(sim, "episode_distance", 0), 0)
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            terminated = getattr(sim, "episode_terminated", True)
            steps = getattr(sim, "episode_steps", 500)

            # CRITÉRIOS DE QUALIDADE (mantidos rigorosos)
            stable = (roll < 0.3 and pitch < 0.3)  
            efficient = (steps <= int(2.0 * distance * 200)) if distance > 0 else False  
            if not stable or terminated:
                return 0.0

            # APPROACH PROGRESSIVO
            if distance <= 0.1:
                return 0.0  

            base_reward = 0.0
            if distance < 1.0:
                base_reward = (distance / 1.0) * 100  
            elif distance < 2.0:
                base_reward = 100 + ((distance - 1.0) / 1.0) * 150  
            elif distance < 3.0:
                base_reward = 250 + ((distance - 2.0) / 1.0) * 200  
            else:
                base_reward = 450 + min((distance - 3.0) * 50, 50) 

            # BÔNUS POR EFICIÊNCIA 
            efficiency_bonus = 0.0
            if distance > 0.5: 
                target_steps = distance * 180  
                actual_steps = steps
                if actual_steps <= target_steps:
                    efficiency_ratio = 1.0
                else:
                    efficiency_ratio = max(0.0, 1.0 - (actual_steps - target_steps) / target_steps)

                efficiency_bonus = efficiency_ratio * 50 

            # BÔNUS POR ESTABILIDADE AVANÇADA
            stability_bonus = 0.0
            if roll < 0.15 and pitch < 0.15:  
                stability_bonus = 30
            elif roll < 0.25 and pitch < 0.25:  
                stability_bonus = 15

            total_reward = base_reward + efficiency_bonus + stability_bonus

            # Limite superior 
            return min(total_reward, 550.0)  

        except Exception as e:
            self.logger.warning(f"Erro no sparse_success_component progressivo: {e}")
            return 0.0

    def _calculate_penalidades_component(self, sim, phase_info) -> float:
        """Componente de PENALIDADES: rigor físico realista"""
        penalties = 0.0
        try:
            # --- 1. Instabilidade angular AGRESSIVA ---
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))

            # Penaliza a partir de 8.5° (0.15 rad), não 17°
            roll_penalty = max(0.0, (roll - 0.15)) * 40.0
            pitch_penalty = max(0.0, (pitch - 0.15)) * 40.0
            penalties -= (roll_penalty + pitch_penalty)

            # --- 2. Movimento para trás ou estagnado ---
            distance = getattr(sim, "episode_distance", 0)
            episode_time = getattr(sim, "episode_time", 0)

            if distance < 0:
                # Andar para trás é inaceitável
                penalties -= abs(distance) * 50.0
            elif abs(distance) < 0.05 and episode_time > 2.0:
                # Congelamento após 2s → punição dura
                penalties -= 100.0

            # --- 3. Desvio lateral excessivo ---
            y_vel = abs(getattr(sim, "robot_y_velocity", 0))
            if y_vel > 0.3:
                penalties -= (y_vel - 0.3) * 100.0

            # --- 4. Altura do COM fora da faixa segura ---
            com_height = getattr(sim, "robot_z_position", 0.8)
            if com_height < 0.6:
                penalties -= (0.6 - com_height) * 50.0
            elif com_height > 1.0:
                penalties -= (com_height - 1.0) * 30.0

            # --- 5. Ações extremas (evitar spiking torque) ---
            action = phase_info.get('action')
            if action is not None:
                action_mag = np.linalg.norm(action)
                if action_mag > 2.5:
                    penalties -= (action_mag - 2.5) * 10.0

            # --- 6. Penalidades críticas (queda, yaw excessivo) ---
            termination = getattr(sim, "episode_termination", "")
            if termination == "fell":
                penalties -= 300.0
            elif termination == "yaw_deviated":
                penalties -= 200.0

            # --- 7. Esforço biomecânico (já existente) ---
            penalties += self._calculate_effort_penalty(sim, phase_info)

        except Exception as e:
            self.logger.warning(f"Erro no cálculo de penalidades: {e}")
        return penalties

    def _calculate_effort_penalty(self, sim, phase_info) -> float:
        """Penalidade por esforço (auxiliar para eficiência)"""
        try:
            if hasattr(sim, 'joint_velocities'):
                effort = sum(v**2 for v in sim.joint_velocities)
                return effort * -0.001
            return 0.0
        except:
            return 0.0

    # Ajuste dos pesos macro
    def adjust_macro_weights(self, weight_adjustments: Dict[str, float]):
        """Ajusta pesos das 6 categorias macro"""
        for component, adjustment in weight_adjustments.items():
            if component in self.macro_components:
                current_weight = self.macro_components[component]
                new_weight = current_weight + (adjustment * self.weight_adjustment_rate)
                
                # Limitar variação
                base_weight = self.base_macro_weights[component]
                max_weight = base_weight * (1 + self.max_weight_change)
                min_weight = base_weight * (1 - self.max_weight_change)
                
                self.macro_components[component] = np.clip(new_weight, min_weight, max_weight)

    def get_macro_weight_status(self) -> Dict:
        """Status dos pesos macro"""
        weight_changes = {}
        for component, current_weight in self.macro_components.items():
            base_weight = self.base_macro_weights[component]
            change_pct = ((current_weight - base_weight) / base_weight) * 100
            weight_changes[component] = {
                'current': current_weight,
                'base': base_weight,
                'change_percent': change_pct
            }
        
        return {
            "macro_weights": weight_changes,
            "adjustment_rate": self.weight_adjustment_rate
        }

    def get_reward_status(self) -> Dict:
        """Status do sistema de recompensas"""
        cache_stats = self.get_cache_stats()
        return {
        "active_components": list(self.components.keys()),
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "total_calculations": self._total_calculations,
            "cache_efficiency": cache_stats["hit_rate"]
        }
    
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
    
    def get_weight_status(self) -> Dict:
        """Status dos pesos para relatórios"""
        weight_changes = {}
        for component, current_weight in self.macro_components.items():
            base_weight = self.base_macro_weights[component]
            change_pct = ((current_weight - base_weight) / base_weight) * 100
            weight_changes[component] = {
                'current': current_weight,
                'base': base_weight,
                'change_percent': change_pct
            }

        return {
            "adaptive_weights": weight_changes,
            "adjustment_rate": self.weight_adjustment_rate
        }
    
    def _generate_essential_cache_key(self, sim, phase_info: Dict) -> str:
        """Gera chave de cache estável baseada em métricas essenciais"""
        try:
            # Métricas essenciais para cache 
            essential_metrics = {}

            # Distância (arredondada)
            distance = getattr(sim, "episode_distance", 0)
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