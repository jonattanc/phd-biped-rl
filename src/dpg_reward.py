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
        
        # NOVO: Apenas 6 componentes macro para o crítico ajustar
        self.macro_components = {
            "progresso": 1.0,
            "coordenacao": 1.0, 
            "estabilidade": 1.0,
            "eficiencia": 1.0,
            "valencia_bonus": 1.0,
            "penalidades": 1.0
        }
        
        self.base_macro_weights = self.macro_components.copy()
        self.weight_adjustment_rate = 0.002  # Ainda mais lento
        self.max_weight_change = 0.4  # 40% de variação máxima
        
        # Sistema de cache unificado
        self.cache = Cache(max_size=500, default_ttl=50)
        self._last_sim_state = None
        self._last_reward = 0.0
        
        # Estatísticas
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0
    
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

        # SOMA PONDERADA COM OS PESOS MACRO
        total_reward = 0.0
        for component, value in component_values.items():
            total_reward += value * self.macro_components[component]

        # Cache
        self.cache.set(cache_key, total_reward, ttl=80)

        return max(total_reward, 0.0)  # Garantir não negativo

    # =========================================================================
    # COMPONENTES MACRO (cada um agrega múltiplas funções específicas)
    # =========================================================================
    
    def _calculate_progresso_component(self, sim, phase_info) -> float:
        """Componente de PROGRESSO: movimento, velocidade, distância"""
        reward = 0.0
        
        # Distância percorrida
        distance = max(getattr(sim, "episode_distance", 0), 0)
        if distance > 0:
            reward += distance * 25.0
            
            # Bônus progressivo por marcos
            if distance > 3.0: reward += 150.0
            elif distance > 2.0: reward += 80.0
            elif distance > 1.5: reward += 50.0
            elif distance > 1.0: reward += 30.0
            elif distance > 0.7: reward += 20.0
            elif distance > 0.5: reward += 12.0
            elif distance > 0.3: reward += 8.0
            elif distance > 0.1: reward += 4.0
            elif distance > 0.05: reward += 2.0
            elif distance > 0.01: reward += 1.0
        
        # Velocidade positiva
        velocity = getattr(sim, "robot_x_velocity", 0)
        if velocity > 0:
            reward += velocity * 6.0
            
        # Bônus de sobrevivência com movimento
        if not getattr(sim, "episode_terminated", True) and distance > 0.01:
            reward += 60.0
            
        return reward
    
    def _calculate_coordenacao_component(self, sim, phase_info) -> float:
        """Componente de COORDENAÇÃO: padrão de marcha, alternância, ritmo"""
        reward = 0.0
        
        try:
            # Padrão alternado
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            
            if left_contact != right_contact:
                reward += 15.0
                
            # Flexão de joelhos durante balanço
            left_knee = getattr(sim, "robot_left_knee_angle", 0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0)
            
            if not left_contact and left_knee > 1.2: reward += 6.0
            if not right_contact and right_knee > 1.2: reward += 6.0
            
            # Clearance dos pés
            left_foot_height = getattr(sim, "robot_left_foot_height", 0)
            right_foot_height = getattr(sim, "robot_right_foot_height", 0)
            
            if not left_contact and left_foot_height > 0.08: reward += 3.0
            if not right_contact and right_foot_height > 0.08: reward += 3.0
            
            # Coordenação completa
            if ((not left_contact and left_knee > 0.7 and left_foot_height > 0.06) or
                (not right_contact and right_knee > 0.7 and right_foot_height > 0.06)):
                reward += 5.0
                
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
            energy_efficiency = 1.0 - min(effort_penalty / 10.0, 1.0)  # Normalizar
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
    
    def _calculate_penalidades_component(self, sim, phase_info) -> float:
        """Componente de PENALIDADES: todas as penalidades consolidadas"""
        penalties = 0.0
        
        try:
            # Penalidades de instabilidade
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            
            if roll > 0.3:  
                penalties -= (roll - 0.3) * 8.0
            if pitch > 0.3:  
                penalties -= (pitch - 0.3) * 10.0
                
            # Penalidade por altura inadequada do COM
            com_height = getattr(sim, "robot_z_position", 0.8)
            if com_height < 0.6:
                penalties -= (0.6 - com_height) * 40.0
            elif com_height > 1.0:
                penalties -= (com_height - 1.0) * 25.0
                
            # Penalidade por desvio lateral
            y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
            if y_velocity > 0.2:  
                penalties -= (y_velocity - 0.2) * 12.0
                
            # Penalidade por ações extremas
            if hasattr(phase_info, 'action') and phase_info.action is not None:
                action_magnitude = np.sqrt(np.sum(np.square(phase_info.action)))
                if action_magnitude > 2.0:
                    penalties -= (action_magnitude - 2.0) * 8.0
                    
            # Penalidades críticas
            if getattr(sim, "episode_termination", "") == "yaw_deviated":
                penalties -= 200.0
            if getattr(sim, "episode_termination", "") == "fell":
                penalties -= 300.0
                
            # Penalidade por esforço
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
            # Métricas essenciais para cache (evitar variações mínimas)
            essential_metrics = {}
            
            # Distância (arredondada)
            distance = getattr(sim, "episode_distance", 0)
            essential_metrics["dist"] = round(distance, 2)  # 2 casas decimais
            
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