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
            "velocity": RewardComponent("velocity", 1.5, self._calculate_velocity_reward),
            "coordination": RewardComponent("coordination", 1.2, self._calculate_coordination_reward),
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

        # 1. Verificar inclinação do tronco para frente (adaptativa para rampas)
        pitch = getattr(sim, "robot_pitch", 0)
        # Em rampas, a inclinação ideal muda - ser mais permissivo
        if pitch > 0.03:  # Limite mais baixo para acomodar rampas
            bonus += 40.0
        elif pitch < -0.08:  # Penalizar inclinação excessiva para trás (especialmente em rampas descendentes)
            bonus -= 30.0

        # 2. Verificar movimento alternado das pernas (CRUCIAL em todos os terrenos)
        left_hip = getattr(sim, "robot_left_hip_angle", 0)
        right_hip = getattr(sim, "robot_right_hip_angle", 0)
        hip_difference = abs(left_hip - right_hip)

        if hip_difference > 0.4:  
            bonus += 100.0  
        elif hip_difference > 0.2:
            bonus += 40.0

        # 3. Bônus por velocidade positiva consistente (adaptado para terrenos difíceis)
        velocity = getattr(sim, "robot_x_velocity", 0)
        if velocity > 0.05:  # Limite mais baixo para terrenos difíceis
            bonus += velocity * 40.0  # Reduzido para não penalizar muito em terrenos difíceis

        # 4. PENALIDADE POR PERNAS MUITO ABERTAS (mas menos severa para terrenos instáveis)
        left_hip_lateral = abs(getattr(sim, "robot_left_hip_lateral_angle", 0))
        right_hip_lateral = abs(getattr(sim, "robot_right_hip_lateral_angle", 0))

        # Em terrenos instáveis (granulado), pernas mais abertas podem ser necessárias
        if left_hip_lateral > 0.7 or right_hip_lateral > 0.7:  # Threshold aumentado
            bonus -= 30.0  # Penalidade reduzida

        # 5. BÔNUS por FASE DE VOO (marcha dinâmica) - crucial para todos os terrenos
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)

        if not left_contact and not right_contact:
            bonus += 25.0 

        # 6. BÔNUS POR FLEXÃO DOS JOELHOS PARA CIMA (ESSENCIAL para clearance em terrenos irregulares)
        left_knee = getattr(sim, "robot_left_knee_angle", 0)
        right_knee = getattr(sim, "robot_right_knee_angle", 0)

        # JOELHO ESQUERDO FLEXIONADO PARA CIMA durante balanço
        if not left_contact:
            if left_knee > 1.2:  # FLEXÃO ALTA (joelho bem erguido)
                bonus += 70.0  # Aumentado para terrenos irregulares
            elif left_knee > 0.9:  # FLEXÃO BOA
                bonus += 50.0  # Aumentado
            elif left_knee > 0.6:  # FLEXÃO MÍNIMA
                bonus += 25.0
            elif left_knee > 0.3:  # PEQUENA FLEXÃO
                bonus += 12.0

        # JOELHO DIREITO FLEXIONADO PARA CIMA durante balanço  
        if not right_contact:
            if right_knee > 1.2:  # FLEXÃO ALTA (joelho bem erguido)
                bonus += 70.0  # Aumentado para terrenos irregulares
            elif right_knee > 0.9:  # FLEXÃO BOA
                bonus += 50.0  # Aumentado
            elif right_knee > 0.6:  # FLEXÃO MÍNIMA
                bonus += 25.0
            elif right_knee > 0.3:  # PEQUENA FLEXÃO
                bonus += 12.0

        # 7. BÔNUS POR CLEARANCE ADEQUADO DOS PÉS (CRÍTICO para terrenos irregulares)
        left_foot_height = getattr(sim, "robot_left_foot_height", 0)
        right_foot_height = getattr(sim, "robot_right_foot_height", 0)

        # Clearance MAIS IMPORTANTE em terrenos irregulares
        if not left_contact:
            if left_foot_height > 0.10:  # 10cm de clearance (aumentado para terrenos irregulares)
                bonus += 35.0
            elif left_foot_height > 0.07:  # 7cm de clearance
                bonus += 20.0
            elif left_foot_height > 0.04:  # 4cm de clearance mínimo
                bonus += 8.0

        if not right_contact:
            if right_foot_height > 0.10:  # 10cm de clearance (aumentado para terrenos irregulares)
                bonus += 35.0
            elif right_foot_height > 0.07:  # 7cm de clearance
                bonus += 20.0
            elif right_foot_height > 0.04:  # 4cm de clearance mínimo
                bonus += 8.0

        # 8. BÔNUS COMBINADO: Joelho erguido + Clearance (MAIS IMPORTANTE em terrenos irregulares)
        if ((not left_contact and left_knee > 0.7 and left_foot_height > 0.06) or
            (not right_contact and right_knee > 0.7 and right_foot_height > 0.06)):
            bonus += 45.0  # Bônus extra aumentado para coordenação completa

        # 9. BÔNUS POR ADAPTAÇÃO A BLOQUEIOS ARTICULARES
        # Detectar movimento suave mesmo com possíveis bloqueios
        joint_velocities = getattr(sim, "joint_velocities", [])
        if joint_velocities:
            avg_joint_velocity = np.mean(np.abs(joint_velocities))
            # Movimento suave é recompensado (indica adaptação a bloqueios)
            if avg_joint_velocity > 0.1 and avg_joint_velocity < 0.8:  # Nem muito lento, nem muito rápido
                bonus += 20.0

        # 10. BÔNUS POR ESTABILIDADE EM SUPERFÍCIES DE BAIXO ATRITO
        # Verificar se está mantendo trajetória estável mesmo com baixo atrito
        y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
        if y_velocity < 0.2:  # Baixa velocidade lateral indica boa estabilidade
            bonus += 15.0

        # 11. BÔNUS POR RECUPERAÇÃO DE EQUILÍBRIO (importante para terrenos irregulares)
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch_abs = abs(pitch)

        # Recompensar por manter equilíbrio apesar de perturbações
        if roll < 0.3 and pitch_abs < 0.4:  # Boa estabilidade angular
            bonus += 20.0

        # 12. BÔNUS POR PADRÃO ALTERNADO DE FLEXÃO DOS JOELHOS
        if (not left_contact and left_knee > 0.7 and 
            right_contact and right_knee < 0.3):  # Esquerdo flexionado, direito estendido
            bonus += 25.0
        if (not right_contact and right_knee > 0.7 and 
            left_contact and left_knee < 0.3):  # Direito flexionado, esquerdo estendido
            bonus += 25.0

        # 13. BÔNUS POR FLEXÃO DOS PÉS (NOVO - CRÍTICO PARA TRABALHO EM RAMPAS)
        left_foot_pitch = getattr(sim, "robot_left_foot_pitch", 0)
        right_foot_pitch = getattr(sim, "robot_right_foot_pitch", 0)

        # FLEXÃO PLANTAR (ponta do pé para baixo) durante apoio - MELHOR TRAÇÃO
        if left_contact and left_foot_pitch > 0.1:  # Pé esquerdo com flexão plantar
            bonus += 35.0
        elif left_contact and left_foot_pitch > 0.05:
            bonus += 20.0

        if right_contact and right_foot_pitch > 0.1:  # Pé direito com flexão plantar
            bonus += 35.0
        elif right_contact and right_foot_pitch > 0.05:
            bonus += 20.0

        # 14. BÔNUS POR PRONAÇÃO/SUPINAÇÃO DOS PÉS (ESTABILIDADE EM RAMPAS)
        left_foot_roll = getattr(sim, "robot_left_foot_roll", 0)
        right_foot_roll = getattr(sim, "robot_right_foot_roll", 0)

        # Pés com inclinação lateral controlada (nem muita pronação nem supinação)
        left_foot_stability = 1.0 - min(abs(left_foot_roll) / 0.3, 1.0)  # Ideal: roll próximo de 0
        right_foot_stability = 1.0 - min(abs(right_foot_roll) / 0.3, 1.0)

        bonus += left_foot_stability * 20.0  # Até +20 por pé estável
        bonus += right_foot_stability * 20.0

        # 15. BÔNUS POR CONTATO FIRME COM O SOLO
        # Pés com orientação adequada para máximo contato
        if left_contact and abs(left_foot_pitch - 0.15) < 0.1:  # Ângulo ideal para rampas
            bonus += 25.0
        if right_contact and abs(right_foot_pitch - 0.15) < 0.1:
            bonus += 25.0

        # 16. BÔNUS COMBINADO: Joelho flexionado + Pé com flexão plantar
        if ((not left_contact and left_knee > 0.8) and 
            (right_contact and right_foot_pitch > 0.08)):
            bonus += 40.0  # Coordenação perfeita para rampas

        if ((not right_contact and right_knee > 0.8) and 
            (left_contact and left_foot_pitch > 0.08)):
            bonus += 40.0

        # 17. BÔNUS POR PADRÃO ALTERNADO COM ADAPTAÇÃO À RAMPA
        if (not left_contact and left_knee > 0.7 and 
            right_contact and right_foot_pitch > 0.06):  # Esquerdo balanço, direito tração
            bonus += 30.0
        if (not right_contact and right_knee > 0.7 and 
            left_contact and left_foot_pitch > 0.06):  # Direito balanço, esquerdo tração
            bonus += 30.0

        # 18. BÔNUS ESPECÍFICO PARA COORDENAÇÃO EM RAMPAS
        pitch = getattr(sim, "robot_pitch", 0)
        if abs(pitch) > 0.15:  
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)

            # BÔNUS MASSIVO por padrão alternado em rampas
            if left_contact != right_contact:
                bonus += 80.0  

            # BÔNUS por adaptação à inclinação
            if pitch > 0:  
                # Flexão adequada de joelhos para subida
                left_knee = getattr(sim, "robot_left_knee_angle", 0)
                right_knee = getattr(sim, "robot_right_knee_angle", 0)
                if (not left_contact and left_knee > 0.8) or (not right_contact and right_knee > 0.8):
                    bonus += 40.0

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
        if hasattr(self, 'cache') and hasattr(self.cache, 'get_stats'):
            return self.cache.get_stats()
        return {"hit_rate": 0.0, "hits": 0, "misses": 0, "size": 0}