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
        distance = metrics.get('distance', 0)
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
            distance = metrics.get('distance', 0)
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
                self.weights.irl_influence = min(self.weights.irl_influence + 0.05, 0.8)
                
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

    def enable(self, enabled=True):
        """Ativa o sistema completo"""
        self.enabled = enabled
        if enabled:
            self.logger.info("Sistema DPG Adaptável ativado")

    def calculate_reward(self, sim, action) -> float:
        """Sistema SIMPLES com ajuda progressiva"""
        if not self.enabled:
            return 0.0

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
            distance = episode_results.get("distance", 0)
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
        distances = [m['distance'] for m in self.episode_metrics_history]
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
        try:
            gait_pattern_score = getattr(sim, 'robot_gait_pattern_score', 0.5)
        except:
            gait_pattern_score = 0.5
        try:
            propulsion_efficiency = getattr(sim, 'robot_propulsion_efficiency', 0.5)
        except:
            propulsion_efficiency = 0.5
        try:
            energy_used = getattr(sim, 'robot_energy_used', 1.0)
        except:
            energy_used = 1.0
        return {
            'distance': getattr(sim, 'episode_distance', 0),
            'speed': getattr(sim, 'robot_x_velocity', 0),
            'roll': abs(getattr(sim, 'robot_roll', 0)),
            'pitch': abs(getattr(sim, 'robot_pitch', 0)),
            'success': getattr(sim, 'episode_success', False),
            'propulsion_efficiency': propulsion_efficiency,
            'energy_used': energy_used,
            'alternating': alternating,
            'gait_pattern_score': gait_pattern_score,
        }

    def update_phase_progression(self, episode_results):
        """Atualização com critic funcional"""
        if not self.enabled:
            return
        self.episode_count += 1
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

        if (self.episode_count - self.last_report_episode) >= self.report_interval:
            self._generate_comprehensive_report()
            self.last_report_episode = self.episode_count

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
        """Preparação de métricas - apenas cálculos essenciais"""
        extended = episode_results.copy()
        active_valences = self.valence_manager.active_valences
        if not active_valences:
            return extended
        required_metrics = set()
        for valence_name in active_valences:
            if valence_name in self.valence_manager.valences:
                valence_config = self.valence_manager.valences[valence_name]
                required_metrics.update(valence_config.metrics)
        try:
            if any(m in required_metrics for m in ["stability", "lateral_stability", "com_height_consistency"]):
                roll = abs(episode_results.get("roll", 0))
                pitch = abs(episode_results.get("pitch", 0))
                extended["stability"] = 1.0 - min((roll + pitch) / 2.0, 1.0)
                extended["lateral_stability"] = 1.0 - min(abs(getattr(self.robot, "y_velocity", 0)) / 0.3, 1.0)
                extended["com_height_consistency"] = 0.8  

            if "alternating_consistency" in required_metrics:
                left_contact = episode_results.get("left_contact", False)
                right_contact = episode_results.get("right_contact", False)
                extended["alternating_consistency"] = 1.0 if left_contact != right_contact else 0.3

            if "energy_efficiency" in required_metrics:
                extended["energy_efficiency"] = episode_results.get("propulsion_efficiency", 0.5)

            default_metrics = {
                "velocity_consistency": 0.7,
                "acceleration_smoothness": 0.8,
                "step_length_consistency": 0.7,
                "gait_robustness": 0.7,
                "speed_adaptation": 0.8,
                "terrain_handling": 0.6
            }

            for metric, default_value in default_metrics.items():
                if metric in required_metrics and metric not in extended:
                    extended[metric] = default_value

        except Exception as e:
            self.logger.warning(f"Erro otimizado ao preparar métricas: {e}")

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
        distance = getattr(self, '_last_distance', 0)

        stability_factor = self._calculate_stability_factor(valence_status)
        propulsion_factor = self._calculate_propulsion_factor(valence_status)

        if overall_progress < 0.3:  
            self.critic.weights.propulsion = 0.75 * propulsion_factor
            self.critic.weights.stability = 0.15 * stability_factor
            self.critic.weights.coordination = 0.06
            self.critic.weights.efficiency = 0.04
            self.critic.weights.irl_influence = 0.9

        elif overall_progress < 0.6:  
            self.critic.weights.propulsion = 0.45 * propulsion_factor
            self.critic.weights.stability = 0.30 * stability_factor
            self.critic.weights.coordination = 0.15
            self.critic.weights.efficiency = 0.10
            self.critic.weights.irl_influence = 0.6

        else:  
            self.critic.weights.stability = 0.40 * stability_factor
            self.critic.weights.propulsion = 0.30 * propulsion_factor
            self.critic.weights.coordination = 0.15
            self.critic.weights.efficiency = 0.15
            self.critic.weights.irl_influence = 0.3

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
        try:
            if hasattr(self, 'buffer_manager') and self.buffer_manager:
                experience_data = {
                    "state": np.zeros(10).tolist(), 
                    "action": np.zeros(6).tolist(),  
                    "reward": episode_results.get('reward', 0),
                    "phase_info": {
                        'group_level': self.current_group,
                        'sub_phase': 0
                    },
                    "metrics": episode_results,
                    "group_level": self.current_group
                }
                self.buffer_manager.store_experience(experience_data)
        except Exception as e:
            self.logger.warning(f"Erro otimizado ao armazenar experiência: {e}")

    def _update_metrics_history(self, episode_results):
        """Atualização otimizada do histórico"""
        self.episode_metrics_history.append({
            'reward': episode_results.get('reward', 0),
            'distance': episode_results.get('distance', 0),
            'speed': episode_results.get('speed', 0),
            'success': episode_results.get('success', False)
        })

        if len(self.episode_metrics_history) > 50:
            self.episode_metrics_history.pop(0)
        
    def _check_irl_activations(self, episode_results):
        """Verifica e ativa IRL quando necessário"""
        distance = episode_results.get('distance', 0)
        reward = episode_results.get('reward', 0)
        if distance < 0 and self.episode_count > 100:
            self.activate_propulsion_irl()
            self.critic.weights.irl_influence = 0.9
        
        if (distance < 0.5 and self.episode_count > 200 and 
            not hasattr(self, '_propulsion_irl_activated')):
            self.activate_propulsion_irl()
            self._propulsion_irl_activated = True
        
        valence_status = self.valence_manager.get_valence_status()
        instability = self._calculate_instability(valence_status)
        if instability > 0.6 and self.episode_count > 50:
            self.activate_stabilization_irl()
        
    def activate_propulsion_irl(self):
        """Ativar IRL ESPECÍFICO para movimento"""
        propulsion_irl_weights = {
            'progress': 0.95,      # FOCO MÁXIMO
            'stability': 0.03,     # Mínimo vital
            'efficiency': 0.01,    # Quase zero
            'coordination': 0.01   # Quase zero
        }

        # FORÇAR pesos do critic
        self.critic.weights.propulsion = 0.95
        self.critic.weights.stability = 0.04
        self.critic.weights.coordination = 0.005
        self.critic.weights.efficiency = 0.005
        self.critic.weights.irl_influence = 0.98

        self.valence_manager.irl_weights = propulsion_irl_weights

    def activate_stabilization_irl(self):
        """Ativa IRL específico para estabilização quando detectada instabilidade"""
        valence_status = self.valence_manager.get_valence_status()
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

    def _calculate_instability(self, valence_status) -> float:
        """Calcula nível de instabilidade baseado nas oscilações das valências"""
        instability = 0.0
        regressing_count = 0
        for valence_name, details in valence_status['valence_details'].items():
            if details['state'] == 'regressing':
                regressing_count += 1
                deficit = details['target_level'] - details['current_level']
                instability += min(deficit, 0.5)
        if regressing_count >= 2:
            instability += 0.3

        return min(instability, 1.0)
    
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
        """Atualiza nível de ajuda baseado em performance REAL"""
        distance = episode_results.get('distance', 0)
        valence_status = self.valence_manager.get_valence_status()
        movimento_level = valence_status['valence_details']['movimento_positivo_basico']['current_level']
        
        # CRITÉRIO PRINCIPAL: movimento real
        if distance > 1.0:
            new_level = 0.1  # Mínimo
        elif distance > 0.5:
            new_level = 0.3
        elif distance > 0.2:
            new_level = 0.5
        elif distance > 0.1:
            new_level = 0.7
        else:
            new_level = 0.9
        
        # BÔNUS por valência de movimento
        if movimento_level > 0.5:
            new_level = max(new_level - 0.2, 0.1)
        
        self.crutches["level"] = new_level
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

    def _calculate_adaptive_crutch_level(self, success, distance, reward, success_rate, avg_distance, avg_reward):
        """Calcula nível de crutch baseado em múltiplas métricas"""
        episode_factor = max(0, 1.0 - (self.episode_count / 1000))
        success_factor = 1.0 - min(success_rate * 1.5, 1.0)
        distance_factor = 1.0 - min(avg_distance / 3.0, 1.0)
        reward_factor = 1.0 - min(max(avg_reward, 0) / 5.0, 1.0)
        valence_status = self.valence_manager.get_valence_status()
        stability_factor = 0.0
        if 'estabilidade_postural' in valence_status['valence_details']:
            stability_level = valence_status['valence_details']['estabilidade_postural']['current_level']
            stability_factor = 1.0 - stability_level  
        crutch_level = (
            episode_factor * 0.2 +
            success_factor * 0.3 +
            distance_factor * 0.25 +
            reward_factor * 0.15 +
            stability_factor * 0.1
        )

        return max(0.0, min(crutch_level, 1.0))

    def _generate_comprehensive_report(self):
        """Relatório completo otimizado"""
        try:
            valence_status = self.valence_manager.get_valence_status()
            consistency_metrics = self._calculate_training_consistency()
            current_irl_weights = self.valence_manager.get_irl_weights()

            mastered_count = sum(
                1 for details in valence_status['valence_details'].values()
                if details['state'] == 'mastered'
            )
            learning_count = sum(
                1 for details in valence_status['valence_details'].values()
                if details['state'] == 'learning'
            )
            regressing_count = sum(
                1 for details in valence_status['valence_details'].values()
                if details['state'] == 'regressing'
            )

            self.logger.info("=" * 70)
            self.logger.info(f"📊 RELATÓRIO DPG COMPLETO - Episódio {self.episode_count}")
            self.logger.info(f"🎯 Progresso Geral: {valence_status['overall_progress']:.1%}")
            self.logger.info(f"📈 Consistência: {consistency_metrics['overall_consistency']:.1%}")
            self.logger.info(f"📊 Valências: 🟢{mastered_count} 🟡{learning_count} 🔴{regressing_count}")

            # STATUS DO IRL
            irl_active = len(current_irl_weights) > 0 and any(w > 0 for w in current_irl_weights.values())
            irl_status = "🟢 ATIVO" if irl_active else "⚫ INATIVO"

            self.logger.info(f"   Influência IRL: {self.critic.weights.irl_influence:.1%}")
            self.logger.info(f"   Pesos Critic: S:{self.critic.weights.stability:.2f} P:{self.critic.weights.propulsion:.2f} C:{self.critic.weights.coordination:.2f} E:{self.critic.weights.efficiency:.2f}")

            # SISTEMA DE CRUTCH
            crutch_stage_names = ["Máximo", "Alto", "Médio", "Baixo", "Mínimo"]
            self.logger.info(f"   Crutch System: Nível {self.crutches['level']:.2f} ({crutch_stage_names[self.crutches['current_stage']]})")

            # ESTATÍSTICAS DO BUFFER
            buffer_status = self.buffer_manager.get_status()
            quality_working = buffer_status.get("quality_calculation_working", False)

            self.logger.info("🔧 VERIFICAÇÃO DO BUFFER:")
            self.logger.info(f"    Calculando: {'✅ SIM' if quality_working else '❌ NÃO'} | Experiências: {buffer_status.get('total_calculated', 0)}")
            self.logger.info(f"    Qualidade média: {buffer_status.get('avg_quality', 0):.2f} | Distância média: {buffer_status.get('avg_distance', 0):.2f}m")

            # ALERTA SE HÁ PROBLEMA
            if not quality_working and buffer_status.get('total_calculated', 0) > 10:
                self.logger.info("   🚨 ALERTA: Sistema NÃO está calculando qualidade corretamente!")
                self.logger.info("   💡 AÇÃO: Verificar _calculate_quality e criação de experiências")

            # ESTADO DETALHADO DAS VALÊNCIAS
            self.logger.info("📈 ESTADO DAS VALÊNCIAS:")
            for valence_name, details in valence_status["valence_details"].items():
                state_icon = {
                    "inactive": "⚫", "learning": "🟡", "consolidating": "🟠",
                    "mastered": "🟢", "regressing": "🔴"
                }.get(details["state"], "⚫")

                consistency = details.get('consistency', 0)
                consistency_icon = "🟢" if consistency > 0.7 else "🟡" if consistency > 0.5 else "🔴"

                self.logger.info(
                    f"   {state_icon} {valence_name}: {details['current_level']:.1%} / "
                    f"{details['target_level']:.1%} {consistency_icon}({consistency:.1%})"
                )

            # MISSÕES ATIVAS 
            if valence_status["current_missions"]:
                self.logger.info("🎯 MISSÕES ATIVAS:")
                for mission in valence_status["current_missions"]:
                    try:
                        # Verificar se os campos existem antes de acessar
                        valence_name = mission.get('valence', 'desconhecida')
                        progress = mission.get('progress', '0')
                        episodes_remaining = mission.get('episodes_remaining', 0)
                        target_improvement = mission.get('target_improvement', 0.1)  # Valor padrão

                        # Calcular progresso percentual com segurança
                        if isinstance(progress, (int, float)) and target_improvement > 0:
                            progress_percent = (float(progress) / target_improvement) * 100
                            progress_text = f"{progress_percent:.0f}%"
                        else:
                            progress_text = f"{progress}"

                        self.logger.info(
                            f"    {valence_name}: {progress_text} "
                            f"({episodes_remaining} episódios restantes)"
                        )
                    
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Erro ao processar missão: {e}")
                        self.logger.info(f"   🎯 Missão com dados incompletos")

            # MÉTRICAS DE PERFORMANCE
            self.logger.info("📈 MÉTRICAS DE TREINAMENTO:")
            self.logger.info(f"   📊 Recompensa média: {consistency_metrics['avg_reward']:.2f}")
            self.logger.info(f"   🏃 Distância média: {consistency_metrics['avg_distance']:.2f}m")
            self.logger.info(f"   🎯 Taxa de sucesso: {consistency_metrics['success_rate']:.1%}")
            self.logger.info(f"   🔄 Bônus de missão: {self.mission_bonus_multiplier:.2f}x")

            # MÉTRICAS DE OTIMIZAÇÃO 
            if hasattr(self.valence_manager, 'get_cache_stats'):
                cache_stats = self.valence_manager.get_cache_stats()
                if cache_stats.get('cache_hit_rate', 0) > 0:
                    self.logger.info(f"   💾 Cache Valence: {cache_stats['cache_hit_rate']:.1%}")

            if hasattr(self.reward_calculator, 'get_cache_stats'):
                reward_cache_stats = self.reward_calculator.get_cache_stats()
                if 'hit_rate' in reward_cache_stats:
                    self.logger.info(f"   💾 Cache Reward: {reward_cache_stats['hit_rate']:.1%}")

            # RECOMENDAÇÕES BASEADAS NO ESTADO ATUAL
            self._log_recommendations(valence_status, consistency_metrics)

            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar relatório: {e}")

    def _log_recommendations(self, valence_status, consistency_metrics):
        """Loga recomendações baseadas no estado atual do treinamento"""
        recommendations = []

        # Analisar estado para gerar recomendações
        if consistency_metrics['success_rate'] < 0.1:
            recommendations.append("🎯 Focar em estabilidade e movimento básico")

        if valence_status['overall_progress'] > 0.7 and consistency_metrics['avg_distance'] < 1.0:
            recommendations.append("🚀 Considerar ativar IRL de propulsão")

        regressing_count = sum(1 for details in valence_status['valence_details'].values() 
                              if details['state'] == 'regressing')
        if regressing_count >= 2:
            recommendations.append("🔴 Estabilizar valências em regressão")

        if self.crutches['level'] > 0.8 and self.episode_count > 500:
            recommendations.append("🔧 Reduzir gradualmente ajuda do crutch system")

        # Logar recomendações se houver
        if recommendations:
            self.logger.info("💡 RECOMENDAÇÕES:")
            for i, recommendation in enumerate(recommendations, 1):
                self.logger.info(f"   {i}. {recommendation}")