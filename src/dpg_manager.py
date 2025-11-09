# dpg_manager.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from dpg_valence import OptimizedValenceManager, ValenceState
from dpg_reward import CachedRewardCalculator
from dpg_buffer import OptimizedBufferManager

@dataclass
class CriticWeights:
    """Pesos do critic baseados em valências"""
    stability: float = 0.25
    propulsion: float = 0.25  
    coordination: float = 0.25
    efficiency: float = 0.25
    irl_influence: float = 0.0

class ValenceAwareCritic:
    """Critic funcional baseado em valências sem PyTorch"""
    
    def __init__(self, logger, valence_count):
        self.logger = logger
        self.valence_count = valence_count
        self.weights = CriticWeights()
        self.performance_history = []
        self.learning_rate = 0.01
        
    def predict_value(self, state, action, valence_levels, irl_weights, episode_metrics):
        """Prediz valor Q baseado em valências e IRL"""
        if len(valence_levels) != self.valence_count:
            return 0.0
            
        # Calcular scores individuais
        stability_score = self._calculate_stability_score(episode_metrics, valence_levels)
        propulsion_score = self._calculate_propulsion_score(episode_metrics, valence_levels)
        coordination_score = self._calculate_coordination_score(episode_metrics, valence_levels)
        efficiency_score = self._calculate_efficiency_score(episode_metrics, valence_levels)
        
        # Combinar scores com pesos atuais
        base_value = (
            stability_score * self.weights.stability +
            propulsion_score * self.weights.propulsion + 
            coordination_score * self.weights.coordination +
            efficiency_score * self.weights.efficiency
        )
        
        # Aplicar influência IRL se disponível
        if irl_weights and self.weights.irl_influence > 0:
            irl_value = self._calculate_irl_value(episode_metrics, irl_weights)
            final_value = (
                base_value * (1 - self.weights.irl_influence) + 
                irl_value * self.weights.irl_influence
            )
        else:
            final_value = base_value
            
        # Armazenar para aprendizado futuro
        self.performance_history.append({
            'valence_levels': valence_levels,
            'base_value': base_value,
            'final_value': final_value,
            'episode_metrics': episode_metrics
        })
        
        # Limitar histórico
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
            
        return final_value
    
    def _calculate_stability_score(self, metrics, valence_levels):
        """Calcula score de estabilidade"""
        roll = abs(metrics.get('roll', 0))
        pitch = abs(metrics.get('pitch', 0))
        stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
        stability_valence = valence_levels[0] if len(valence_levels) > 0 else 0.0
        return stability * (0.7 + stability_valence * 0.3)
    
    def _calculate_propulsion_score(self, metrics, valence_levels):
        """Calcula score de propulsão"""
        distance = max(metrics.get("distance", 0), 0)
        velocity = metrics.get('speed', 0)
        distance_score = min(distance / 3.0, 1.0)
        velocity_score = min(abs(velocity) / 1.5, 1.0) if velocity > 0 else 0.0
        propulsion_valence = valence_levels[1] if len(valence_levels) > 1 else 0.0
        propulsion_bonus = 0.5 + propulsion_valence * 0.5
        
        return (distance_score * 0.6 + velocity_score * 0.4) * propulsion_bonus
    
    def _calculate_coordination_score(self, metrics, valence_levels):
        """Calcula score de coordenação"""
        alternating = metrics.get('alternating', False)
        gait_score = metrics.get('gait_pattern_score', 0.5)
        coordination_score = gait_score
        if alternating:
            coordination_score += 0.3
        coordination_valence = valence_levels[2] if len(valence_levels) > 2 else 0.0
        
        return min(coordination_score * (0.8 + coordination_valence * 0.4), 1.0)
    
    def _calculate_efficiency_score(self, metrics, valence_levels):
        """Calcula score de eficiência"""
        efficiency = metrics.get('propulsion_efficiency', 0.5)
        energy_used = metrics.get('energy_used', 1.0)
        efficiency_score = efficiency
        if energy_used < 0.8:  
            efficiency_score += 0.2
        efficiency_valence = valence_levels[3] if len(valence_levels) > 3 else 0.0
        
        return min(efficiency_score * (0.7 + efficiency_valence * 0.5), 1.0)
    
    def _calculate_irl_value(self, metrics, irl_weights):
        """Calcula valor baseado em alinhamento IRL"""
        alignment = 0.0
        if 'progress' in irl_weights:
            distance = max(metrics.get("distance", 0), 0)
            alignment += irl_weights['progress'] * min(distance / 2.0, 1.0)
        if 'stability' in irl_weights:
            roll = abs(metrics.get('roll', 0))
            pitch = abs(metrics.get('pitch', 0))
            stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
            alignment += irl_weights['stability'] * stability
        if 'efficiency' in irl_weights:
            efficiency = metrics.get('propulsion_efficiency', 0.5)
            alignment += irl_weights['efficiency'] * efficiency
        if 'coordination' in irl_weights:
            alternating = metrics.get('alternating', False)
            coordination = 0.8 if alternating else 0.3
            alignment += irl_weights['coordination'] * coordination
            
        return min(alignment, 1.0)
    
    def update_weights(self, valence_status):
        """Atualiza pesos do critic baseado no desempenho recente"""
        try:
            learning_valences = [
                name for name, details in valence_status['valence_details'].items()
                if details['state'] == 'learning' and details['current_level'] < 0.7
            ]
            
            regressing_valences = [
                name for name, details in valence_status['valence_details'].items()
                if details['state'] == 'regressing'
            ]
            
            adjustment_made = False
            
            for valence_name in learning_valences + regressing_valences:
                if valence_name == 'propulsao_eficiente' or valence_name == 'propulsao_basica':
                    self.weights.propulsion = min(self.weights.propulsion + 0.05, 0.5)
                    adjustment_made = True
                elif valence_name == 'eficiencia_biomecanica':
                    self.weights.efficiency = min(self.weights.efficiency + 0.05, 0.4)
                    adjustment_made = True
                elif 'estabilidade' in valence_name:
                    self.weights.stability = min(self.weights.stability + 0.05, 0.5)
                    adjustment_made = True
                elif 'coordenacao' in valence_name or 'ritmo' in valence_name:
                    self.weights.coordination = min(self.weights.coordination + 0.05, 0.4)
                    adjustment_made = True
            
            if not regressing_valences and len(learning_valences) < 2:
                self.weights.irl_influence = max(self.weights.irl_influence - 0.02, 0.1)
            elif regressing_valences:
                self.weights.irl_influence = min(self.weights.irl_influence + 0.05, 0.5)
                
        except Exception as e:
            self.logger.warning(f"Erro na atualização dos pesos do critic: {e}")


class DPGManager:
    """DPG Manager com critic funcional"""
    
    def __init__(self, logger, robot, reward_system, state_dim=10, action_dim=6):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.enabled = True
        self.config = type('Config', (), {'enabled': True})
        self.config.valence_system = True

        self.valence_manager = OptimizedValenceManager(logger, {}) 
        self.reward_calculator = CachedRewardCalculator(logger, {}) 
        self.buffer_manager = OptimizedBufferManager(logger, {})
        self.buffer_manager._dpg_manager = self

        valence_count = len(self.valence_manager.valences)
        self.critic = ValenceAwareCritic(logger, valence_count)

        self.learning_progress = 0.0
        self.episode_count = 0
        self.performance_trend = 0.0
        self.mission_bonus_multiplier = 1.0
        self.last_valence_update_episode = 0
        self.valence_update_interval = 5
        self.last_critic_update_episode = 0
        self.critic_update_interval = 50
        self.last_report_episode = 0
        self.report_interval = 200
        self._cached_valence_weights = {}
        self._cached_irl_weights = {}
        self._cache_episode = 0
        self.episode_metrics_history = []
        self.current_group = 1
        self._last_known_group = 1
        self.group_transition_history = []
        self.crutches = {
            "enabled": True,
            "level": 1.0,  
            "progress_thresholds": [0.3, 0.5, 0.7, 0.9], 
            "current_stage": 0,
            "base_reward_boost": 2.0,
            "action_smoothing": 0.8,
            "penalty_reduction": 0.3
        }
        self._emergency_activated = False
        self._emergency_episode_threshold = 100  
        self.activate_propulsion_irl()

    def enable(self, enabled=True):
        """Ativa o sistema completo"""
        self.enabled = enabled
        if enabled:
            self.logger.info("Sistema DPG Adaptável ativado")

    def calculate_reward(self, sim, action) -> float:
        """Sistema SIMPLES com ajuda progressiva"""
        if not self.enabled:
            return 0.0
        self._current_episode_action = action
        valence_status = self.valence_manager.get_valence_status()
        valence_weights = self.valence_manager.get_valence_weights_for_reward()
        irl_weights = self.valence_manager.get_irl_weights()
        enabled_components = self.valence_manager.get_active_reward_components()

        phase_info = {
            'group_level': max(1, int(valence_status['overall_progress'] * 3) + 1),
            'group_name': 'valence_system',
            'valence_weights': valence_weights,
            'irl_weights': irl_weights,
            'enabled_components': enabled_components,
            'target_speed': self._get_target_speed_from_valences(valence_status),
            'learning_progress': valence_status['overall_progress'],
            'valence_status': valence_status
        }

        base_reward = self.reward_calculator.calculate(sim, action, phase_info)
        crutch_level = self.crutches["level"]
        crutch_stage = self.crutches["current_stage"]
        crutch_multipliers = [4.0, 3.0, 2.0, 1.5, 1.0]  
        crutch_multiplier = crutch_multipliers[crutch_stage]
        boosted_reward = base_reward * crutch_multiplier
        if self.episode_count < 500: 
            distance = getattr(sim, "episode_distance", 0)
            if distance > 0.05:  
                distance_bonus = min(distance * 5.0, 8.0) 
                boosted_reward += distance_bonus
            
            if not getattr(sim, "episode_terminated", True):
                survival_bonus = 1.0  
                boosted_reward += survival_bonus

        return max(boosted_reward, 0.0)
    
    def _extract_valence_levels(self, valence_status):
        """Extrai níveis das valências como array"""
        levels = []
        for valence_name in self.valence_manager.valences.keys():
            level = valence_status['valence_details'].get(valence_name, {}).get('current_level', 0.0)
            levels.append(level)
        return levels

    def _get_target_speed_from_valences(self, valence_status):
        """Calcula velocidade alvo baseada no progresso das valências"""
        overall_progress = valence_status['overall_progress']
        base_speed = 0.1 + (overall_progress * 2.4)  # Até 2.5 m/s

        details = valence_status['valence_details']
        if details.get('propulsao_eficiente', {}).get('current_level', 0) > 0.5:
            base_speed *= 1.3  # Bônus de velocidade quando propulsão eficiente
        if details.get('eficiencia_biomecanica', {}).get('current_level', 0) > 0.6:
            base_speed *= 1.1  # Pequeno bônus por eficiência biomecânica

        return min(base_speed, 2.5)
    
    def _extract_state(self, sim):
        """Extrai estado da simulação"""
        return np.array([
            getattr(sim, "robot_x_velocity", 0),
            getattr(sim, "robot_roll", 0),
            getattr(sim, "robot_pitch", 0),
            getattr(sim, "robot_z_position", 0.8),
        ], dtype=np.float32)

    def _prepare_valence_metrics(self, episode_results):
        """Prepara métricas estendidas para o sistema de valências"""
        extended = episode_results.copy()
        try:
            # Métricas básicas
            roll = abs(episode_results.get("roll", 0))
            pitch = abs(episode_results.get("pitch", 0))
            distance = max(episode_results.get('distance', 0), 0)
            velocity = episode_results.get("speed", 0)

            # Métricas para valências aprimoradas
            extended["stability"] = 1.0 - min((roll + pitch) / 2.0, 1.0)
            extended["positive_movement_rate"] = 1.0 if distance > 0.1 else 0.0

            # Métricas de estabilidade dinâmica
            extended["com_height_consistency"] = 0.8 
            extended["lateral_stability"] = 1.0 - min(abs(getattr(self.robot, "y_velocity", 0)) / 0.3, 1.0)
            extended["pitch_velocity"] = abs(getattr(self.robot, "pitch_velocity", 0))

            # Métricas de propulsão eficiente
            extended["velocity_consistency"] = 0.7 
            extended["acceleration_smoothness"] = 0.8 

            # Métricas de ritmo de marcha
            left_contact = episode_results.get("left_contact", False)
            right_contact = episode_results.get("right_contact", False)
            extended["alternating_consistency"] = 1.0 if left_contact != right_contact else 0.3
            extended["step_length_consistency"] = 0.7  
            extended["stance_swing_ratio"] = 0.6  

            # Métricas de eficiência biomecânica
            extended["energy_efficiency"] = episode_results.get("propulsion_efficiency", 0.5)
            extended["stride_efficiency"] = distance / max(episode_results.get("steps", 1), 1)
            extended["clearance_score"] = episode_results.get("clearance_score", 0.5)

            # Métricas de marcha robusta
            extended["gait_robustness"] = 0.7  
            extended["recovery_success"] = 1.0 if episode_results.get("success", False) else 0.0
            extended["speed_adaptation"] = 0.8  
            extended["terrain_handling"] = 0.6  

        except Exception as e:
            self.logger.warning(f"Erro ao preparar métricas de valência: {e}")

        return extended

    def _determine_group_from_valences(self, valence_status):
        """Determina grupo baseado nas valências mastered"""
        mastered_count = sum(
            1 for details in valence_status['valence_details'].values()
            if details['state'] == 'mastered'
        )
        if mastered_count >= 4:
            return 3 
        elif mastered_count >= 2:  
            return 2 
        else:
            return 1

    def _calculate_training_consistency(self) -> Dict:
        """Calcula métricas de consistência do treinamento"""
        if len(self.episode_metrics_history) < 10:
            return {
                'overall_consistency': 0.5,
                'avg_reward': 0.0,
                'avg_distance': 0.0,
                'success_rate': 0.0
            }
        rewards = [m['reward'] for m in self.episode_metrics_history]
        distances = [max(m['distance'], 0) for m in self.episode_metrics_history]
        successes = [m['success'] for m in self.episode_metrics_history]
        avg_reward = np.mean(rewards)
        avg_distance = np.mean(distances)
        success_rate = np.mean(successes)
        reward_std = np.std(rewards)
        max_reward = max(rewards) if rewards else 1.0
        reward_consistency = 1.0 - min(reward_std / max(avg_reward, 0.1), 1.0)
        if len(distances) > 5:
            recent_distances = distances[-5:]
            distance_trend = np.polyfit(range(len(recent_distances)), recent_distances, 1)[0]
            distance_consistency = 1.0 - min(abs(distance_trend) / 0.1, 1.0) if distance_trend >= 0 else 0.3
        else:
            distance_consistency = 0.5

        overall_consistency = (reward_consistency * 0.6 + distance_consistency * 0.4)

        return {
            'overall_consistency': overall_consistency,
            'avg_reward': avg_reward,
            'avg_distance': avg_distance,
            'success_rate': success_rate
        }

    def get_brain_status(self):
        """Retorna status completo do sistema DPG"""
        if not self.enabled:
            return {"status": "disabled"}
        valence_status = self.valence_manager.get_valence_status()
        status = {
            "status": "active",
            "enabled": self.enabled,
            "learning_progress": self.learning_progress,
            "performance_trend": self.performance_trend,
            "episode_count": self.episode_count,
            "valence_system": True,
        }
        status.update({
            "current_valences": len(valence_status['active_valences']),
            "overall_progress": valence_status['overall_progress'],
            "active_missions": len(valence_status['current_missions']),
            "valence_states": {
                name: details['state'] 
                for name, details in valence_status['valence_details'].items()
            }
        })
        if hasattr(self, 'buffer_manager') and self.buffer_manager:
            try:
                buffer_status = self.buffer_manager.get_status()
                status.update({
                    "total_experiences": buffer_status.get("total_experiences", 0),
                    "current_group_experiences": buffer_status.get("current_group_experiences", 0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter status do buffer_manager: {e}")
        
        return status

    def get_advanced_metrics(self):
        """Retorna métricas avançadas para monitoramento"""
        if not self.enabled:
            return {}
        valence_status = self.valence_manager.get_valence_status()
        metrics = {
            "system_health": 1.0,
            "learning_progress": self.learning_progress,
            "performance_trend": self.performance_trend,
            "valence_system_active": True,
            "active_valences_count": len(valence_status['active_valences']),
            "mission_bonus_active": self.mission_bonus_multiplier > 1.0,
            "irl_active": len(self.valence_manager.get_irl_weights()) > 0,
            "critic_active": True,
        }
        valence_details = valence_status['valence_details']
        for valence_name, details in valence_details.items():
            metrics[f"valence_{valence_name}_level"] = details['current_level']
            metrics[f"valence_{valence_name}_state"] = details['state']
            metrics[f"valence_{valence_name}_consistency"] = details.get('consistency', 0)
        metrics.update({
            "critic_irl_influence": self.critic.weights.irl_influence,
            "critic_stability_weight": self.critic.weights.stability,
            "critic_propulsion_weight": self.critic.weights.propulsion,
            "critic_coordination_weight": self.critic.weights.coordination,
            "critic_efficiency_weight": self.critic.weights.efficiency,
            "crutch_level": self.crutches["level"],
            "crutch_stage": self.crutches["current_stage"],
            "crutch_enabled": self.crutches["enabled"],
            "crutch_reward_boost": self.crutches["base_reward_boost"] * self.crutches["level"]
        })
        if hasattr(self, 'buffer_manager') and self.buffer_manager:
            try:
                buffer_metrics = self.buffer_manager.get_metrics()
                metrics.update({
                    "buffer_avg_quality": buffer_metrics.get("buffer_avg_quality", 0),
                    "buffer_avg_reward": buffer_metrics.get("buffer_avg_reward", 0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter métricas do buffer_manager: {e}")
        
        return metrics

    def get_system_status(self):
        """Retorna status completo do sistema"""
        status = {
            "enabled": self.enabled,
            "learning_progress": self.learning_progress,
            "performance_trend": self.performance_trend,
            "valence_system": True,
        }
        status.update(self.valence_manager.get_valence_status())
        if self.reward_calculator:
            status.update(self.reward_calculator.get_reward_status())
        if self.buffer_manager:
            status.update(self.buffer_manager.get_status())
        
        return status
    
    def _extract_episode_metrics(self, sim):
        """Extrai métricas do episódio atual para o critic"""
        try:
            alternating = (
                getattr(sim, 'robot_left_foot_contact', False) != 
                getattr(sim, 'robot_right_foot_contact', False)
            )
        except:
            alternating = False
                
        return {
            'distance': getattr(sim, 'episode_distance', 0),
            'speed': getattr(sim, 'robot_x_velocity', 0),
            'roll': abs(getattr(sim, 'robot_roll', 0)),
            'pitch': abs(getattr(sim, 'robot_pitch', 0)),
            'success': getattr(sim, 'episode_success', False),
            'propulsion_efficiency': getattr(sim, 'robot_propulsion_efficiency', 0.5),
            'energy_used': getattr(sim, 'robot_energy_used', 1.0),
            'alternating': alternating,
            'gait_pattern_score': getattr(sim, 'robot_gait_pattern_score', 0.5),
        }

    def update_phase_progression(self, episode_results):
        """Atualização com critic funcional"""
        if not self.enabled:
            return
        self.episode_count += 1
        distance_raw = episode_results.get('distance', 'N/A')
        
        # FORÇAR conversão para float se necessário
        if isinstance(distance_raw, (int, float)):
            episode_results['distance'] = float(distance_raw)
        else:
            episode_results['distance'] = 0.0
        
        self.update_crutch_system(episode_results)
        should_update_valences = self._should_update_valences(episode_results)
        should_update_critic = self._should_update_critic()

        if should_update_valences:
            self._perform_valence_update(episode_results)

        if should_update_critic:
            self._perform_critic_update()
        self._store_optimized_experience(episode_results)
        self._update_metrics_history(episode_results)
        self._check_irl_activations(episode_results)

        if self.episode_count >= 3000 and self.episode_count% 500 == 0:
            self._perform_periodic_cleanup()

        if (self.episode_count - self.last_report_episode) >= self.report_interval:
            self._generate_comprehensive_report()
            self.last_report_episode = self.episode_count
        
    def _perform_periodic_cleanup(self):
        """Limpeza periódica do buffer"""
        try:
            if hasattr(self, 'buffer_manager') and self.buffer_manager:
                self.buffer_manager.cleanup_low_quality_experiences(min_quality_threshold=0.35)
                   
                # Limpar experiências muito antigas a cada 1000 episódios
                if self.episode_count % 1000 == 0:
                    self.buffer_manager.cleanup_old_experiences(max_age_episodes=1500)
                       
        except Exception as e:
            self.logger.warning(f"Erro na limpeza periódica: {e}")
            
    def _should_update_valences(self, episode_results) -> bool:
        """Verifica se atualização de valências é necessária"""
        if (self.episode_count - self.last_valence_update_episode) < self.valence_update_interval:
            return False
        special_events = (
            episode_results.get('success', False) or
            episode_results.get('distance', 0) > 2.0 or  
            any(metric in episode_results for metric in ['emergency_correction', 'movement_bonus'])
        )

        return special_events or (self.episode_count % self.valence_update_interval == 0)

    def _should_update_critic(self) -> bool:
        """Verifica se atualização do critic é necessária"""
        return (
            (self.episode_count - self.last_critic_update_episode) >= self.critic_update_interval and
            len(self.episode_metrics_history) >= 10  
        )

    def _perform_valence_update(self, episode_results):
        """Atualização de valências"""
        extended_results = self._prepare_valence_metrics_optimized(episode_results)
        valence_weights, mission_bonus = self.valence_manager.update_valences(extended_results)
        self._cached_valence_weights = valence_weights
        self._cached_irl_weights = self.valence_manager.get_irl_weights()
        self.mission_bonus_multiplier = mission_bonus
        valence_status = self.valence_manager.get_valence_status()
        self.learning_progress = valence_status['overall_progress']
        self.last_valence_update_episode = self.episode_count
        new_group = self._determine_group_from_valences(valence_status)
        if new_group != self.current_group:
            self._check_group_transition(valence_status)

    def _prepare_valence_metrics_optimized(self, episode_results):
        """Preparação de métricas COMPLETA"""
        extended = episode_results.copy()

        try:
            # MÉTRICAS ESSENCIAIS PARA TODAS AS VALÊNCIAS
            try:
                raw_distance = episode_results.get("distance", 0)
                if not isinstance(raw_distance, (int, float)):
                    self.logger.error(f"❌ DISTÂNCIA NÃO NUMÉRICA: {raw_distance} (type: {type(raw_distance)})")
                    distance = 0.0
                else:
                    distance = float(raw_distance)

                extended["distance"] = distance

            except Exception as e:
                self.logger.error(f"❌ ERRO CRÍTICO ao processar distância: {e}")
                extended["distance"] = 0.0

            velocity = episode_results.get("speed", 0)
            roll = abs(episode_results.get("roll", 0))
            pitch = abs(episode_results.get("pitch", 0))

            # Métricas básicas sempre disponíveis
            extended.update({
                "speed": float(velocity) if isinstance(velocity, (int, float)) else 0.0,
                "roll": float(roll) if isinstance(roll, (int, float)) else 0.0,
                "pitch": float(pitch) if isinstance(pitch, (int, float)) else 0.0,
                "stability": 1.0 - min((roll + pitch) / 1.0, 1.0),
                "positive_movement_rate": 1.0 if distance > 0.1 else 0.0
            })

            # MÉTRICAS DE ESTABILIDADE
            extended["com_height_consistency"] = 0.8
            extended["lateral_stability"] = 1.0 - min(abs(getattr(self.robot, "y_velocity", 0)) / 0.3, 1.0)

            # MÉTRICAS DE PROPULSÃO
            extended["velocity_consistency"] = 0.7
            extended["acceleration_smoothness"] = 0.8

            # MÉTRICAS DE COORDENAÇÃO
            left_contact = episode_results.get("left_contact", False)
            right_contact = episode_results.get("right_contact", False)
            extended["alternating_consistency"] = 1.0 if left_contact != right_contact else 0.3
            extended["step_length_consistency"] = 0.7
            extended["gait_pattern_score"] = 0.8 if left_contact != right_contact else 0.4

            # MÉTRICAS DE EFICIÊNCIA
            extended["energy_efficiency"] = episode_results.get("propulsion_efficiency", 0.5)
            extended["stride_efficiency"] = distance / max(episode_results.get("steps", 1), 1)

            # MÉTRICAS DE MARCHA ROBUSTA
            extended["gait_robustness"] = 0.7
            extended["recovery_success"] = 1.0 if episode_results.get("success", False) else 0.0
            extended["speed_adaptation"] = 0.8
            extended["terrain_handling"] = 0.6

        except Exception as e:
            self.logger.warning(f"Erro ao preparar métricas: {e}")

        return extended

    def _perform_critic_update(self):
        """Atualização otimizada do critic"""
        if len(self.episode_metrics_history) < 10:
            return

        try:
            valence_status = self.valence_manager.get_valence_status()
            self.critic.update_weights(valence_status)
            self._stabilize_critic_weights_adaptive()
            self.last_critic_update_episode = self.episode_count

        except Exception as e:
            self.logger.warning(f"Erro na atualização do critic: {e}")

    def _stabilize_critic_weights_adaptive(self):
        """Estabilização adaptativa baseada no progresso atual"""
        valence_status = self.valence_manager.get_valence_status()
        overall_progress = valence_status['overall_progress']
        max_irl_influence = 0.3 if overall_progress < 0.3 else 0.2

        if self.critic.weights.propulsion > 0.8:
            # Redistribuir pesos quando propulsão dominar demais
            excess = self.critic.weights.propulsion - 0.6
            self.critic.weights.propulsion = 0.6
            self.critic.weights.coordination += excess * 0.6
            self.critic.weights.stability += excess * 0.4

        # GARANTIR que IRL não domine o critic
        if self.critic.weights.irl_influence > max_irl_influence:
            self.critic.weights.irl_influence = max_irl_influence

        self._normalize_critic_weights()

    def _calculate_stability_factor(self, valence_status) -> float:
        """Calcula fator de estabilidade baseado nas valências"""
        try:
            stability_valences = ['estabilidade_dinamica', 'estabilidade_postural']
            stability_levels = []

            for valence_name in stability_valences:
                if valence_name in valence_status['valence_details']:
                    level = valence_status['valence_details'][valence_name]['current_level']
                    stability_levels.append(level)

            if stability_levels:
                avg_stability = sum(stability_levels) / len(stability_levels)
                return 1.0 + (0.7 - avg_stability) * 0.5
            return 1.0

        except Exception:
            return 1.0

    def _calculate_propulsion_factor(self, valence_status) -> float:
        """Calcula fator de propulsão baseado nas valências"""
        try:
            propulsion_valences = ['propulsao_eficiente', 'propulsao_basica']
            propulsion_levels = []

            for valence_name in propulsion_valences:
                if valence_name in valence_status['valence_details']:
                    level = valence_status['valence_details'][valence_name]['current_level']
                    state = valence_status['valence_details'][valence_name]['state']
                    factor = 0.5 if state == 'regressing' else 1.0
                    propulsion_levels.append(level * factor)

            if propulsion_levels:
                avg_propulsion = sum(propulsion_levels) / len(propulsion_levels)
                return 1.0 + (0.6 - avg_propulsion) * 0.6
            return 1.0

        except Exception:
            return 1.0

    def _normalize_critic_weights(self):
        """Garante que a soma dos pesos do critic seja 1.0"""
        total = (self.critic.weights.stability + 
                 self.critic.weights.propulsion + 
                 self.critic.weights.coordination + 
                 self.critic.weights.efficiency)

        if total > 0:
            self.critic.weights.stability /= total
            self.critic.weights.propulsion /= total
            self.critic.weights.coordination /= total
            self.critic.weights.efficiency /= total

    def _store_optimized_experience(self, episode_results):
        """Armazenamento otimizado de experiência"""
        real_action = getattr(self, '_current_episode_action', None)

        try:
            if hasattr(self, 'buffer_manager') and self.buffer_manager:
                # Criar dados de experiência
                experience_data = {
                    "state": self._extract_state(self.robot).tolist(),  
                    "action": real_action,
                    "reward": episode_results.get('reward', 0),
                    "next_state": self._extract_state(self.robot).tolist(),  
                    "phase_info": {
                        'group_level': self.current_group,
                        'sub_phase': 0,
                        'valence_status': self.valence_manager.get_valence_status()
                    },
                    "metrics": episode_results,  # 
                    "group_level": self.current_group
                }

                # Chamar armazenamento
                self.buffer_manager.store_experience(experience_data)
                self._current_episode_action = None

        except Exception as e:
            self.logger.error(f"❌ ERRO CRÍTICO no armazenamento: {e}")

    def _update_metrics_history(self, episode_results):
        """Atualização otimizada do histórico"""
        self.episode_metrics_history.append({
            'reward': episode_results.get('reward', 0),
            'distance': max(episode_results.get('distance', 0), 0),
            'speed': episode_results.get('speed', 0),
            'success': episode_results.get('success', False)
        })

        if len(self.episode_metrics_history) > 50:
            self.episode_metrics_history.pop(0)
        
    def _check_irl_activations(self, episode_results):
        """Ativação AGRESSIVA de IRL quando movimento é insuficiente"""
        distance = episode_results.get('distance', 0)
        alternating = episode_results.get('alternating', False)
        pitch = abs(episode_results.get('pitch', 0))
        valence_status = self.valence_manager.get_valence_status()
        movimento_level = valence_status['valence_details'].get('movimento_basico', {}).get('current_level', 0)
        coordenacao_level = valence_status['valence_details'].get('coordenacao_fundamental', {}).get('current_level', 0)
    
        # SÓ ativa IRL se critic não estiver já focando no mesmo componente
        current_focus = max([
            ('propulsion', self.critic.weights.propulsion),
            ('coordination', self.critic.weights.coordination),
            ('stability', self.critic.weights.stability)
        ], key=lambda x: x[1])

        if pitch > 0.15 and not alternating:
            self.activate_coordination_focus()
        
        if movimento_level > 0.6 and coordenacao_level < 0.3:
            self.activate_coordination_focus()
        
        # Se critic já está focado em propulsão, NÃO ativa IRL de propulsão
        if current_focus[0] == 'propulsion' and current_focus[1] > 0.7:
            if distance < 0.5:
                self.critic.weights.coordination += 0.1
                self.critic.weights.propulsion -= 0.1
        else:
            if distance < 0.5:  
                self.activate_propulsion_irl()
            elif distance < 1:
                self.activate_stabilization_irl()
            elif distance < 2 and not alternating:
                self.activate_coordination_focus()
            else:
                self.critic.weights.irl_influence = max(0.1, self.critic.weights.irl_influence - 0.01)
        
    def activate_propulsion_irl(self):
        """Ativar IRL ESPECÍFICO para movimento"""
        propulsion_irl_weights = {
            'progress': 0.9,      # FOCO MÁXIMO
            'stability': 0.06,     # Mínimo vital
            'efficiency': 0.02,    # Quase zero
            'coordination': 0.02   # Quase zero
        }

        # FORÇAR pesos do critic
        self.critic.weights.propulsion = 0.9
        self.critic.weights.stability = 0.06
        self.critic.weights.coordination = 0.02
        self.critic.weights.efficiency = 0.02
        self.critic.weights.irl_influence = 0.3

        self.valence_manager.irl_weights = propulsion_irl_weights

    def activate_stabilization_irl(self):
        """Ativa IRL específico para estabilização quando detectada instabilidade"""
        self.valence_manager.get_valence_status()
        irl_weights = {
            'progress': 0.3,     
            'stability': 0.5,     
            'efficiency': 0.1,    
            'coordination': 0.1  
        }
        try:
            if hasattr(self.valence_manager, 'set_irl_weights'):
                self.valence_manager.set_irl_weights(irl_weights)
            elif hasattr(self.valence_manager, 'update_irl_weights'):
                self.valence_manager.update_irl_weights(irl_weights)
        except Exception as e:
            self.logger.warning(f"❌ Erro ao ativar IRL de estabilização: {e}")
            return
        self.critic.weights.irl_influence = min(self.critic.weights.irl_influence + 0.1, 0.4)
   
    def activate_coordination_focus(self):
        """ATIVA FOCO MÁXIMO EM COORDENAÇÃO"""
        coordination_irl_weights = {
            'coordination': 0.60,      
            'propulsion': 0.25,          
            'stability': 0.12,        
            'efficiency': 0.03         
        }

        # FORÇAR pesos do critic para coordenação
        self.critic.weights.coordination = 0.65
        self.critic.weights.propulsion = 0.25
        self.critic.weights.stability = 0.08
        self.critic.weights.efficiency = 0.02
        self.critic.weights.irl_influence = 0.15  

        self.valence_manager.irl_weights = coordination_irl_weights
    
    def _get_adaptive_config(self):
        """Retorna configuração para preservação adaptativa"""
        valence_status = self.valence_manager.get_valence_status()
        overall_progress = valence_status['overall_progress']
        if overall_progress > 0.8:
            return {
                "learning_preservation": "medium",
                "skill_transfer": True,
                "core_preservation": True,
                "preservation_rate": 0.7
            }
        elif overall_progress > 0.5:
            return {
                "learning_preservation": "high", 
                "skill_transfer": True,
                "core_preservation": True,
                "preservation_rate": 0.8
            }
        else:
            return {
                "learning_preservation": "high",
                "skill_transfer": True, 
                "core_preservation": True,
                "preservation_rate": 0.9
            }
    
    def _check_group_transition(self, valence_status):
        """Verifica e executa transição de grupo se necessário"""
        new_group = self._determine_group_from_valences(valence_status)
        if new_group != self.current_group:           
            self.buffer_manager.transition_with_preservation(
                self.current_group, new_group, self._get_adaptive_config()
            )
            self.current_group = new_group
    
    def update_crutch_system(self, episode_results):
        """SISTEMA DE CRUTCH MAIS AGRESSIVO"""
        distance = max(episode_results.get('distance', 0), 0)
        valence_status = self.valence_manager.get_valence_status()

        # BASEADO no progresso REAL, não apenas em episódios
        movimento_level = valence_status['valence_details'].get('movimento_basico', {}).get('current_level', 0)

        # SE não há progresso real, MANTÉM ou AUMENTA muletas
        if movimento_level < 0.3 and distance < 0.3:
            new_level = max(self.crutches["level"], 0.6)
        else:
            # Redução baseada em progresso REAL
            if movimento_level > 0.5:
                reduction_factor = 0.8
            elif distance > 0.5:
                reduction_factor = 0.9
            else:
                reduction_factor = 0.95

            new_level = self.crutches["level"] * reduction_factor

        self.crutches["level"] = max(new_level, 0.05)
        self._update_crutch_stage()

    def _update_crutch_stage(self):
        """ESTÁGIOS MAIS BEM DISTRIBUÍDOS"""
        crutch_level = self.crutches["level"]

        if crutch_level > 0.8:
            self.crutches["current_stage"] = 0  
        elif crutch_level > 0.6:
            self.crutches["current_stage"] = 1    
        elif crutch_level > 0.4:
            self.crutches["current_stage"] = 2  
        elif crutch_level > 0.2:
            self.crutches["current_stage"] = 3  
        else:
            self.crutches["current_stage"] = 4

    def get_training_strategy(self):
        """ESTRATÉGIA REALISTA PARA 10.000 EPISÓDIOS"""
        return {
            "fase_1_movimento_basico": {
                "episodios": "0-1500",
                "objetivo": "Qualquer movimento positivo (> 0.01m)",
                "valencia_principal": "movimento_basico",
                "target_distance": 0.1,
                "crutch_level": 0.95
            },
            "fase_2_estabilidade": {
                "episodios": "1500-3000", 
                "objetivo": "Movimento estável (> 0.3m com estabilidade)",
                "valencia_principal": "estabilidade_postural",
                "target_distance": 0.5,
                "crutch_level": 0.85
            },
            "fase_3_propulsao": {
                "episodios": "3000-5000",
                "objetivo": "Propulsão consistente (> 1.0m)",
                "valencia_principal": "propulsao_basica", 
                "target_distance": 1.5,
                "crutch_level": 0.70
            },
            "fase_4_coordenacao": {
                "episodios": "5000-7000",
                "objetivo": "Marcha coordenada (> 2.0m)",
                "valencia_principal": "coordenacao_fundamental",
                "target_distance": 2.5,
                "crutch_level": 0.50
            },
            "fase_5_marcha_robusta": {
                "episodios": "7000-10000", 
                "objetivo": "Marcha robusta (> 3.0m)",
                "valencia_principal": "marcha_robusta",
                "target_distance": 4.0,
                "crutch_level": 0.30
            }
        }

    def _generate_comprehensive_report(self):
        """RELATÓRIO COMPLETO - CRÍTIC, IRL, MULETAS E MISSÕES"""

        # Coletar dados de todos os sistemas
        valence_status = self.valence_manager.get_valence_status()
        buffer_status = self.buffer_manager.get_status()
        irl_weights = self.valence_manager.get_irl_weights()
        cache_stats = getattr(self.reward_calculator, 'get_cache_stats', lambda: {})()

        # Métricas principais
        real_distance = self.buffer_manager._calculate_avg_distance()
        movement_exps = buffer_status.get('movement_experience_count', 0)
        total_exps = buffer_status.get('total_experiences', 0)
        movement_rate = movement_exps / total_exps if total_exps > 0 else 0

        # Calcular tendência de progresso
        progress_trend = "🟢 SUBINDO" if self.performance_trend > 0.01 else "🔴 CAINDO" if self.performance_trend < -0.01 else "🟡 ESTÁVEL"

        self.logger.info("=" * 70)
        self.logger.info(f"🎯 EPISÓDIO {self.episode_count} | PROGRESSO: {valence_status['overall_progress']:.1%} {progress_trend}")
        self.logger.info(f"📊 Distância média: {real_distance:.3f}m | Movimento: {movement_exps}/{total_exps} ({movement_rate:.1%})")

        # SEÇÃO 1: SISTEMA DE CRÍTIC
        self.logger.info("🧠 SISTEMA CRÍTIC (Avaliação):")
        critic_weights = self.critic.weights
        self.logger.info(f"   Estabilidade: {critic_weights.stability:.3f} | Propulsão: {critic_weights.propulsion:.3f}")
        self.logger.info(f"   Coordenação: {critic_weights.coordination:.3f} | Eficiência: {critic_weights.efficiency:.3f}")

        # SEÇÃO 2: SISTEMA IRL (Aprendizado por Reforço Inverso)
        self.logger.info(f"🔍 SISTEMA IRL com {critic_weights.irl_influence:.1%} de influência (Preferências):")
        if irl_weights:
            irl_total = sum(irl_weights.values())
            if irl_total > 0:
                self.logger.info(f"   Progresso: {irl_weights.get('progress', 0):.3f} | Estabilidade: {irl_weights.get('stability', 0):.3f}")
                self.logger.info(f"   Eficiência: {irl_weights.get('efficiency', 0):.3f} | Coordenação: {irl_weights.get('coordination', 0):.3f}")

        # SEÇÃO 3: SISTEMA DE MULETAS
        crutch_stage_names = ["MÁXIMO", "ALTO", "MÉDIO", "BAIXO", "MÍNIMO"]
        stage_idx = self.crutches["current_stage"]
        self.logger.info(f"🦯 SISTEMA DE MULETAS (Suporte) no estágio {crutch_stage_names[stage_idx]}")
        self.logger.info(f"   Nível: {self.crutches['level']:.3f} | Multiplicador: {self.crutches['base_reward_boost'] * self.crutches['level']:.2f}x")

        # SEÇÃO 4: MISSÕES ATIVAS
        if valence_status["current_missions"]:
            self.logger.info("🎯 MISSÕES ATIVAS:")
            for mission_data in valence_status["current_missions"]:
                valence_name = mission_data.get('valence', 'desconhecida')
                current_level = valence_status['valence_details'].get(valence_name, {}).get('current_level', 0)
                episodes_left = mission_data.get('episodes_remaining', 0)
                progress_icon = "🟢" if current_level > 0.5 else "🟡" if current_level > 0.3 else "🔴"
                self.logger.info(f"   {progress_icon} {valence_name}: {current_level:.1%} ({episodes_left} episódios restantes)")

        # SEÇÃO 5: VALÊNCIAS PRINCIPAIS (apenas as ativas/relevantes)
        self.logger.info("📈 VALÊNCIAS PRINCIPAIS:")
        active_valences = []
        mastered_valences = []
        learning_valences = []
        regressing_valences = []

        for valence_name, details in valence_status["valence_details"].items():
            if details['state'] != 'inactive' or details['current_level'] > 0.1:
                if details['state'] == 'mastered':
                    mastered_valences.append((valence_name, details))
                elif details['state'] == 'learning':
                    learning_valences.append((valence_name, details))
                elif details['state'] == 'regressing':
                    regressing_valences.append((valence_name, details))
                else:
                    active_valences.append((valence_name, details))

        # Ordenar por nível atual (mais alto primeiro)
        for category in [mastered_valences, learning_valences, regressing_valences, active_valences]:
            category.sort(key=lambda x: x[1]['current_level'], reverse=True)

        # Mostrar masterizadas primeiro
        for valence_name, details in mastered_valences:
            self.logger.info(f"   🏆 {valence_name}: {details['current_level']:.1%} (DOMINADA)")

        # Mostrar em aprendizado
        for valence_name, details in learning_valences:
            learning_icon = "📈" if details.get('learning_rate', 0) > 0.01 else "📉"
            self.logger.info(f"   {learning_icon} {valence_name}: {details['current_level']:.1%} (aprendendo)")

        # Mostrar regredindo
        for valence_name, details in regressing_valences:
            self.logger.info(f"   ⚠️  {valence_name}: {details['current_level']:.1%} (REGREDINDO!)")

        # Mostrar outras ativas
        for valence_name, details in active_valences:
            state_icon = '⚪' if details['state'] == 'inactive' else '🔵'
            self.logger.info(f"   {state_icon} {valence_name}: {details['current_level']:.1%} ({details['state']})")

        if not any([mastered_valences, learning_valences, regressing_valences, active_valences]):
            self.logger.info("   ⚠️  Nenhuma valência ativa ainda")

        # SEÇÃO 6: RECOMPENSAS E EFICIÊNCIA
        self.logger.info("💰 SISTEMA DE RECOMPENSAS (média):")
        avg_reward = buffer_status.get('avg_reward', 0)
        avg_quality = buffer_status.get('avg_quality', 0)

        reward_efficiency = "ALTA" if avg_reward > 50 else "MÉDIA" if avg_reward > 20 else "BAIXA"
        quality_efficiency = "ALTA" if avg_quality > 0.7 else "MÉDIA" if avg_quality > 0.4 else "BAIXA"

        self.logger.info(f"   Recompensa: {avg_reward:.1f} ({reward_efficiency})| Qualidade: {avg_quality:.1%} ({quality_efficiency})")

        # Cache performance se disponível
        if cache_stats and 'hit_rate' in cache_stats:
            self.logger.info(f"   Cache: {cache_stats['hit_rate']:.1%} eficiência")

        # SEÇÃO 7: RECOMENDAÇÕES AUTOMÁTICAS
        recommendations = self._generate_automated_recommendations(valence_status, buffer_status)
        if recommendations:
            self.logger.info("💡 RECOMENDAÇÕES:")
            for rec in recommendations[:3]:  # Mostrar apenas as top 3
                self.logger.info(f"   {rec}")

        self.logger.info("=" * 80)

    def _generate_automated_recommendations(self, valence_status, buffer_status):
        """Gera recomendações automáticas baseadas no estado atual"""
        recommendations = []

        # Análise do Critic
        if self.critic.weights.irl_influence > 0.4:
            recommendations.append("🔻 Reduzir influência IRL do critic (muito alta)")

        if self.critic.weights.propulsion < 0.2 and buffer_status.get('avg_distance', 0) < 0.5:
            recommendations.append("🔺 Aumentar peso de propulsão no critic")

        if self.critic.weights.coordination < 0.1 and self.episode_count > 1000:
            recommendations.append("🔺 Aumentar peso de coordenação no critic")

        # Análise de Muletas
        if self.crutches["level"] > 0.6 and self.episode_count > 1500:
            recommendations.append("🔻 Reduzir nível de muletas mais agressivamente")

        if self.crutches["level"] < 0.1 and buffer_status.get('avg_distance', 0) < 0.3:
            recommendations.append("🔺 Aumentar temporariamente muletas (dificuldade alta)")

        # Análise de Valências
        for valence_name, details in valence_status["valence_details"].items():
            if details['state'] == 'regressing':
                recommendations.append(f"🎯 Criar missão para valência {valence_name} (regredindo)")

            if details['learning_rate'] < 0.005 and details['current_level'] < 0.5:
                recommendations.append(f"🔍 Investigar valência {valence_name} (aprendizado lento)")

        # Análise de Movimento
        movement_rate = buffer_status.get('movement_experience_count', 0) / max(buffer_status.get('total_experiences', 1), 1)
        if movement_rate < 0.3:
            recommendations.append("🚨 Foco urgente em movimento positivo (taxa muito baixa)")

        if buffer_status.get('avg_distance', 0) < 0.1 and self.episode_count > 500:
            recommendations.append("🎯 Ativar IRL de propulsão agressivo")

        return recommendations