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
    """SISTEMA DE RECOMPENSAS UNIFICADO E OTIMIZADO"""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.components = self._initialize_components()
        
        # Sistema de cache unificado
        self.cache = Cache(max_size=500, default_ttl=50)
        self._last_sim_state = None
        self._last_reward = 0.0
        
        # Estatísticas
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0
    
    def _initialize_components(self) -> Dict[str, RewardComponent]:
        """Componentes ESSENCIAIS que devem SEMPRE ser calculados"""
        return {
            "movement_priority": RewardComponent("movement_priority", 10.0, self._calculate_movement_priority_reward),
            "gait_system": RewardComponent("gait_system", 8.0, self._calculate_gait_system_reward),
            "global_penalties": RewardComponent("global_penalties", -5.0, self._calculate_global_penalties),
            "progress_reward": RewardComponent("progress_reward", 6.0, self._calculate_progress_reward),
            "stability": RewardComponent("stability", 3.0, self._calculate_stability_reward),
            "coordination": RewardComponent("coordination", 2.0, self._calculate_coordination_reward),
        }
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
        """Sistema de recompensas UNIFICADO com cache inteligente"""
        self._total_calculations += 1
        
        # Gerar chave de cache baseada no estado essencial
        cache_key = self._generate_essential_cache_key(sim, phase_info)
        
        # Verificar cache apenas para estados similares
        cached_reward = self.cache.get(cache_key)
        if cached_reward is not None:
            self._cache_hits += 1
            return cached_reward
        
        self._cache_misses += 1
        
        # CALCULAR COMPONENTES ESSENCIAIS (sempre calculados)
        total_reward = 0.0
        
        # 1. MOVIMENTO POSITIVO (CRÍTICO - sempre calculado)
        movement_reward = self._calculate_movement_priority_reward(sim, phase_info)
        total_reward += movement_reward
        
        # 2. SISTEMA DE MARCHA (CRÍTICO - sempre calculado)
        gait_reward = self._calculate_gait_system_reward(sim, phase_info)
        total_reward += gait_reward
        
        # 3. PROGRESSO (CRÍTICO - sempre calculado)
        progress_reward = self._calculate_progress_reward(sim, phase_info)
        total_reward += progress_reward
        
        # 4. PENALIDADES GLOBAIS (CRÍTICO - sempre calculado)
        penalties = self._calculate_global_penalties(sim, phase_info)
        total_reward += penalties
        
        # 5. COMPONENTES ADICIONAIS baseados nas valências ativas
        valence_components = self._calculate_valence_based_components(sim, phase_info)
        total_reward += valence_components
        
        # Garantir recompensa não-negativa para movimento positivo
        final_reward = max(total_reward, 0.0) if movement_reward > 0 else total_reward
        
        # Armazenar no cache apenas se for um estado representativo
        if self._should_cache_state(sim):
            self.cache.set(cache_key, final_reward, ttl=30)
        
        return final_reward
    
    def _generate_essential_cache_key(self, sim, phase_info: Dict) -> str:
        """Gera chave de cache baseada em métricas ESSENCIAIS"""
        try:
            # Métricas fundamentais para detecção de estado similar
            distance = max(getattr(sim, "episode_distance", 0), 0)
            velocity = getattr(sim, "robot_x_velocity", 0)
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            
            # Estado de contato dos pés (importante para marcha)
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            
            # Grupo atual e componentes ativos
            group = phase_info.get('group_level', 1)
            active_valences = phase_info.get('valence_status', {}).get('active_valences', [])
            
            # Criar fingerprint do estado
            state_fingerprint = (
                f"d{distance:.2f}_v{velocity:.2f}_"
                f"r{roll:.2f}_p{pitch:.2f}_"
                f"lc{left_contact}_rc{right_contact}_"
                f"g{group}_v{len(active_valences)}"
            )
            
            return state_fingerprint
            
        except Exception as e:
            self.logger.warning(f"Erro ao gerar chave de cache: {e}")
            return f"error_{self._total_calculations}"
    
    def _should_cache_state(self, sim) -> bool:
        """Determina se o estado atual deve ser cacheado"""
        # Cachear apenas estados com movimento significativo ou estabilidade
        distance = max(getattr(sim, "episode_distance", 0), 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        
        # Cachear se: movimento positivo OU velocidade estável OU baixa instabilidade
        return (distance > 0.1 or 
                abs(velocity) > 0.05 or 
                self._is_stable_state(sim))
    
    def _is_stable_state(self, sim) -> bool:
        """Verifica se é um estado estável representativo"""
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))
        return (roll < 0.3 and pitch < 0.3)
    
    def _calculate_movement_priority_reward(self, sim, phase_info) -> float:
        """RECOMPENSA CRÍTICA - Movimento positivo (SEMPRE calculada)"""
        distance = max(getattr(sim, "episode_distance", 0), 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        
        # PENALIDADE NUCLEAR por movimento negativo
        if distance < 0:
            return -10000.0
        
        base_reward = 0.0
        
        # RECOMPENSA PROGRESSIVA por movimento positivo
        if distance > 0:
            # Escala linear base + bônus progressivo
            base_reward += distance * 800.0  # Base aumentada
            
            # Bônus por marcos de distância
            if distance > 3.0: base_reward += 5000.0
            elif distance > 2.0: base_reward += 3000.0
            elif distance > 1.5: base_reward += 2000.0
            elif distance > 1.0: base_reward += 1000.0
            elif distance > 0.7: base_reward += 500.0
            elif distance > 0.5: base_reward += 200.0
            elif distance > 0.3: base_reward += 100.0
            elif distance > 0.1: base_reward += 50.0
            elif distance > 0.05: base_reward += 20.0
            elif distance > 0.01: base_reward += 10.0
        
        # Recompensa por velocidade positiva
        if velocity > 0:
            base_reward += velocity * 300.0
        elif velocity < -0.01:  # Pequena penalidade por velocidade negativa
            base_reward -= 100.0
            
        # Bônus de sobrevivência com movimento
        if not getattr(sim, "episode_terminated", True) and distance > 0.01:
            base_reward += 200.0
            
        return base_reward
    
    def _calculate_gait_system_reward(self, sim, phase_info) -> float:
        """SISTEMA DE MARCHA COMPLETO (SEMPRE calculado)"""
        bonus = 0.0
        
        try:
            # 1. PADRÃO ALTERNADO (fundamental)
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            
            if left_contact != right_contact:
                bonus += 150.0  # Bônus massivo por padrão alternado
            elif not left_contact and not right_contact:
                bonus += 80.0   # Fase de voo
            else:
                bonus -= 50.0   # Penalidade por apoio duplo prolongado
            
            # 2. FLEXÃO DE JOELHOS (clearance)
            left_knee = getattr(sim, "robot_left_knee_angle", 0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0)
            
            # Joelho esquerdo durante balanço
            if not left_contact:
                if left_knee > 1.2: bonus += 120.0
                elif left_knee > 0.9: bonus += 80.0
                elif left_knee > 0.6: bonus += 40.0
            
            # Joelho direito durante balanço
            if not right_contact:
                if right_knee > 1.2: bonus += 120.0
                elif right_knee > 0.9: bonus += 80.0
                elif right_knee > 0.6: bonus += 40.0
            
            # 3. CLEARANCE DOS PÉS
            left_foot_height = getattr(sim, "robot_left_foot_height", 0)
            right_foot_height = getattr(sim, "robot_right_foot_height", 0)
            
            if not left_contact and left_foot_height > 0.08:
                bonus += 60.0
            if not right_contact and right_foot_height > 0.08:
                bonus += 60.0
                
            # 4. COORDENAÇÃO COMPLETA
            if ((not left_contact and left_knee > 0.7 and left_foot_height > 0.06) or
                (not right_contact and right_knee > 0.7 and right_foot_height > 0.06)):
                bonus += 100.0
                
            # 5. ADAPTAÇÃO A INCLINAÇÕES
            pitch = getattr(sim, "robot_pitch", 0)
            if abs(pitch) > 0.15:  # Em rampas
                # Bônus adicional por manter padrão alternado em rampas
                if left_contact != right_contact:
                    bonus += 100.0
                    
                # Flexão adaptativa para rampas
                if pitch > 0:  # Subida
                    if (not left_contact and left_knee > 0.8) or (not right_contact and right_knee > 0.8):
                        bonus += 80.0
                        
            # 6. FLEXÃO PLANTAR (tração)
            left_foot_pitch = getattr(sim, "robot_left_foot_pitch", 0)
            right_foot_pitch = getattr(sim, "robot_right_foot_pitch", 0)
            
            if left_contact and left_foot_pitch > 0.08:
                bonus += 70.0
            if right_contact and right_foot_pitch > 0.08:
                bonus += 70.0
                
        except Exception as e:
            self.logger.warning(f"Erro no cálculo do sistema de marcha: {e}")
            
        return bonus
    
    def _calculate_progress_reward(self, sim, phase_info) -> float:
        """RECOMPENSA de PROGRESSO (SEMPRE calculada)"""
        base_reward = 0.0
        distance = max(getattr(sim, "episode_distance", 0), 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        
        # Componente principal: distância percorrida
        if distance > 0:
            distance_reward = distance * 200.0
            base_reward += distance_reward
            
            # Bônus progressivo por marcos
            if distance > 2.0: base_reward += 1500.0
            elif distance > 1.5: base_reward += 800.0
            elif distance > 1.0: base_reward += 400.0
            elif distance > 0.7: base_reward += 200.0
            elif distance > 0.5: base_reward += 100.0
            elif distance > 0.3: base_reward += 50.0
            elif distance > 0.1: base_reward += 20.0
        
        # Componente secundário: velocidade consistente
        if velocity > 0.1:
            base_reward += velocity * 80.0
            
        # Bônus de sobrevivência
        if not getattr(sim, "episode_terminated", True) and distance > 0.05:
            base_reward += 150.0
            
        return base_reward
    
    def _calculate_global_penalties(self, sim, phase_info) -> float:
        """PENALIDADES GLOBAIS (SEMPRE calculadas)"""
        penalties = 0.0
        
        try:
            # 1. Penalidade por instabilidade angular
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            
            if roll > 0.4:
                penalties -= roll * 150.0
            if pitch > 0.4:
                penalties -= pitch * 120.0
                
            # 2. Penalidade por altura inadequada do COM
            com_height = getattr(sim, "robot_z_position", 0.8)
            if com_height < 0.6:
                penalties -= (0.6 - com_height) * 300.0
            elif com_height > 1.0:
                penalties -= (com_height - 1.0) * 200.0
                
            # 3. Penalidade por movimento lateral excessivo
            y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
            if y_velocity > 0.3:
                penalties -= (y_velocity - 0.3) * 100.0
                
            # 4. Penalidade por ações extremas
            if hasattr(phase_info, 'action') and phase_info.action is not None:
                action_magnitude = np.sqrt(np.sum(np.square(phase_info.action)))
                if action_magnitude > 2.0:
                    penalties -= (action_magnitude - 2.0) * 50.0
                    
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de penalidades globais: {e}")
            
        return penalties
    
    def _calculate_valence_based_components(self, sim, phase_info) -> float:
        """Componentes ADAPTATIVOS baseados nas valências ativas"""
        additional_reward = 0.0
        
        try:
            valence_status = phase_info.get('valence_status', {})
            active_valences = valence_status.get('active_valences', [])
            
            # Recompensas específicas por valência ativa
            for valence in active_valences:
                if valence == "estabilidade_postural":
                    # Bônus por estabilidade quando esta valência está ativa
                    roll = abs(getattr(sim, "robot_roll", 0))
                    pitch = abs(getattr(sim, "robot_pitch", 0))
                    stability = 1.0 - min((roll + pitch) / 1.0, 1.0)
                    additional_reward += stability * 80.0
                    
                elif valence == "propulsao_basica":
                    # Bônus por velocidade positiva
                    velocity = getattr(sim, "robot_x_velocity", 0)
                    if velocity > 0:
                        additional_reward += velocity * 60.0
                        
                elif valence == "coordenacao_fundamental":
                    # Bônus por padrão alternado
                    left_contact = getattr(sim, "robot_left_foot_contact", False)
                    right_contact = getattr(sim, "robot_right_foot_contact", False)
                    if left_contact != right_contact:
                        additional_reward += 100.0
                        
        except Exception as e:
            self.logger.warning(f"Erro em componentes baseados em valência: {e}")
            
        return additional_reward
    
    def _calculate_stability_reward(self, sim, phase_info) -> float:
        """Recompensa por estabilidade (componente opcional)"""
        try:
            roll = abs(getattr(sim, "robot_roll", 0))
            pitch = abs(getattr(sim, "robot_pitch", 0))
            stability = 1.0 - min((roll + pitch) / 1.0, 1.0)
            return stability * 50.0
        except:
            return 0.0
    
    def _calculate_coordination_reward(self, sim, phase_info) -> float:
        """Recompensa por coordenação (componente opcional)"""
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            
            if left_contact != right_contact:
                return 80.0
            elif not left_contact and not right_contact:
                return 40.0
            else:
                return -30.0
        except:
            return 0.0
    
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
        """Estatísticas de cache CORRIGIDAS"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0
        }