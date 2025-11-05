# dpg_manager.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from dpg_valence import ValenceManager, ValenceState
from dpg_reward import RewardCalculator
from dpg_buffer import SmartBufferManager

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
        if len(self.performance_history) < 10:
            return
        learning_valences = [
            name for name, details in valence_status['valence_details'].items()
            if details['state'] == 'learning' and details['current_level'] < 0.7
        ]
        if learning_valences:
            for valence_name in learning_valences:
                if valence_name == 'propulsao_basica':
                    self.weights.propulsion = min(self.weights.propulsion + 0.1, 0.4)
                elif valence_name == 'eficiencia_propulsiva':
                    self.weights.efficiency = min(self.weights.efficiency + 0.1, 0.4)

class DPGManager:
    """DPG Manager com critic funcional sem PyTorch"""
    
    def __init__(self, logger, robot, reward_system, state_dim=10, action_dim=6):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.enabled = True
        self.config = type('Config', (), {'enabled': True})
        self.config.valence_system = True

        self.valence_manager = ValenceManager(logger, {})
        self.reward_calculator = RewardCalculator(logger, {})
        self.buffer_manager = SmartBufferManager(logger, {})
        self.buffer_manager._dpg_manager = self
        valence_count = len(self.valence_manager.valences)
        self.critic = ValenceAwareCritic(logger, valence_count)

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

        progress = valence_status['overall_progress']
        if progress < 0.3: 
            boosted_reward = base_reward * 2.0  
        elif progress < 0.6: 
            boosted_reward = base_reward * 1.5  
        else:  
            boosted_reward = base_reward        

        if self.episode_count < 300:
            distance = getattr(sim, "episode_distance", 0)
            if distance > 0.1:
                boosted_reward += min(distance * 1.0, 2.0)

        return boosted_reward
    
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
        base_speed = 0.1 + (overall_progress * 1.9) 
        details = valence_status['valence_details']
        if details.get('propulsao_basica', {}).get('current_level', 0) > 0.5:
            base_speed *= 1.3
        if details.get('eficiencia_propulsiva', {}).get('current_level', 0) > 0.6:
            base_speed *= 1.2
        
        return min(base_speed, 2.0)
    
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
            roll = abs(episode_results.get("roll", 0))
            pitch = abs(episode_results.get("pitch", 0))
            extended["stability"] = 1.0 - min((roll + pitch) / 2.0, 1.0)
            distance = episode_results.get("distance", 0)
            extended["positive_movement_rate"] = 1.0 if distance > 0.1 else 0.0
            left_contact = episode_results.get("left_contact", False)
            right_contact = episode_results.get("right_contact", False)
            extended["alternating_score"] = 1.0 if left_contact != right_contact else 0.3
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
            self.logger.warning(f"Erro ao armazenar experiência: {e}")
        self._last_distance = episode_results.get('distance', 0)
        if (self.episode_count > 50 and 
            len(self.valence_manager.get_irl_weights()) == 0):
            self.activate_irl_guidance()
        if self._last_distance < 2.0 and self.episode_count > 100:
            self.activate_propulsion_irl()
        valence_status = self.valence_manager.get_valence_status()
        if (valence_status['overall_progress'] > 0.8 and 
            self._last_distance < 1.0 and 
            self.episode_count > 500):
            self._emergency_unstick_propulsion(valence_status)
        extended_results = self._prepare_valence_metrics(episode_results)
        valence_weights, mission_bonus = self.valence_manager.update_valences(extended_results)
        regressing_count = sum(1 for details in valence_status['valence_details'].values() 
        if details['state'] == 'regressing')
        if regressing_count >= 2 and self.episode_count % 50 == 0:
            self.emergency_stabilization()
        if self._last_distance < 1.0 and self.episode_count > 300:
            self.activate_propulsion_irl()
        if (self.episode_count > 50 and 
            len(self.valence_manager.get_irl_weights()) == 0 and
            valence_status['overall_progress'] > 0.3):
            self.activate_irl_guidance() 
        if (self.episode_count - self.last_valence_update_episode) >= self.valence_update_interval:
            extended_results = self._prepare_valence_metrics(episode_results)
            valence_weights, mission_bonus = self.valence_manager.update_valences(extended_results)
            self._cached_valence_weights = valence_weights
            self._cached_irl_weights = self.valence_manager.get_irl_weights()
            self._cache_episode = self.episode_count
            self.mission_bonus_multiplier = mission_bonus
            valence_status = self.valence_manager.get_valence_status()
            self.learning_progress = valence_status['overall_progress']
            self.last_valence_update_episode = self.episode_count
            self._check_group_transition(valence_status)
        if (self.episode_count > 100 and 
            self.valence_manager.get_irl_weights() == {} and
            episode_results.get('reward', 0) < -2.0): 
            self.activate_irl_guidance()
        if (self.episode_count - self.last_critic_update_episode) >= self.critic_update_interval:
            valence_status = self.valence_manager.get_valence_status()
            self.critic.update_weights(valence_status)
            self.last_critic_update_episode = self.episode_count
            self._stabilize_critic_weights()
        if (self.episode_count - self.last_report_episode) >= self.report_interval:
            self._generate_comprehensive_report()
            self.last_report_episode = self.episode_count
        self.episode_metrics_history.append({
            'reward': episode_results.get('reward', 0),
            'distance': episode_results.get('distance', 0),
            'speed': episode_results.get('speed', 0),
            'success': episode_results.get('success', False)
        })
        if len(self.episode_metrics_history) > 50:
            self.episode_metrics_history.pop(0)

    def _emergency_propulsion_fix(self, valence_status):
        """Correção de emergência quando propulsão fica zerada"""
        if 'propulsao_basica' in valence_status['valence_details']:
            details = valence_status['valence_details']['propulsao_basica']
            if details['current_level'] == 0.0 and self.episode_count > 100:
                self.valence_manager.valence_performance['propulsao_basica'].current_level = 0.3
                self.valence_manager.valence_performance['propulsao_basica'].state = ValenceState.LEARNING
                self.valence_manager.active_valences.add('propulsao_basica')
            
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
    
    def activate_propulsion_irl(self):
        """Ativar IRL ESPECÍFICO para movimento"""
        propulsion_irl_weights = {
            'progress': 0.8,      
            'stability': 0.2,    
            'efficiency': 0.1,   
            'coordination': 0.1  
        }
        try:
            self.valence_manager.irl_weights = propulsion_irl_weights
            self.critic.weights.irl_influence = 0.9
        except Exception as e:
            self.logger.warning(f"❌ Erro ao ativar IRL de propulsão: {e}")
        
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

    def _stabilize_critic_weights(self):
        """Estabiliza pesos do critic para evitar oscilações - VERSÃO MAIS CONSERVADORA"""
        distance = getattr(self, '_last_distance', 0)
        if distance < 4.0: 
            self.critic.weights.propulsion = 0.80
            self.critic.weights.stability = 0.10
            self.critic.weights.coordination = 0.05
            self.critic.weights.efficiency = 0.04
        else:
            self.critic.weights.stability = 0.60
            self.critic.weights.propulsion = 0.20
            self.critic.weights.coordination = 0.10
            self.critic.weights.efficiency = 0.10
        if distance < 3.0:
            self.critic.weights.irl_influence = 0.9

    def emergency_stabilization(self):
        """Ativação emergencial para estabilizar valências oscilantes"""
        valence_status = self.valence_manager.get_valence_status()
        fundamental_valences = ['estabilidade_postural', 'propulsao_basica']
        for valence_name in fundamental_valences:
            if valence_name in self.valence_manager.valence_performance:
                perf = self.valence_manager.valence_performance[valence_name]
                if perf.state == ValenceState.REGRESSING:
                    perf.state = ValenceState.LEARNING
                    perf.current_level = max(perf.current_level, 0.3) 
        self.critic.weights.stability = 0.40
        self.critic.weights.propulsion = 0.30
        self.critic.weights.coordination = 0.15
        self.critic.weights.efficiency = 0.15
        self.critic.weights.irl_influence = 0.1
    
    def _check_group_transition(self, valence_status):
        """Verifica e executa transição de grupo se necessário"""
        new_group = self._determine_group_from_valences(valence_status)
        if new_group != self.current_group:           
            self.buffer_manager.transition_with_preservation(
                self.current_group, new_group, self._get_adaptive_config()
            )
            self.current_group = new_group

    def activate_irl_guidance(self):
        """Ativa e configura o sistema IRL para fornecer orientação"""       
        manual_irl_weights = {
            'progress': 0.4,     
            'stability': 0.3,     
            'efficiency': 0.2,   
            'coordination': 0.1  
        }
        try:
            if hasattr(self.valence_manager, 'irl_weights'):
                self.valence_manager.irl_weights = manual_irl_weights
            elif hasattr(self.valence_manager.irl_system, 'learned_weights'):
                self.valence_manager.irl_system.learned_weights = manual_irl_weights
            self.valence_manager.irl_system._active = True
            self.critic.weights.irl_influence = 0.3
        except Exception as e:
            self.logger.warning(f"❌ Erro ao ativar IRL: {e}")
            
    def emergency_reward_correction(self):
        """Correção emergencial para recompensas negativas"""
        try:
            valence_status = self.valence_manager.get_valence_status()
            propulsao_details = valence_status['valence_details'].get('propulsao_basica', {})
            if propulsao_details.get('current_level', 0) < 0:
                correction_metrics = {
                    'positive_movement_rate': 0.5,  
                    'distance': 0.5,                
                    'speed': 0.3,                  
                    'emergency_correction': True   
                }
                self.valence_manager.update_valences(correction_metrics)
        except Exception as e:
            self.logger.warning(f"⚠️ Correção automática falhou: {e}")
        self.critic.weights.propulsion = 0.35
        self.critic.weights.stability = 0.30
        self.critic.weights.efficiency = 0.20
        self.critic.weights.coordination = 0.15
        self.mission_bonus_multiplier = 1.5

    def _emergency_unstick_propulsion(self, valence_status):
        """Correção mais eficiente para propulsão travada"""
        if 'propulsao_basica' in valence_status['valence_details']:
            details = valence_status['valence_details']['propulsao_basica']
            if (details['current_level'] < 0.3 and 
                self.episode_count > 100):  
                self.valence_manager.valence_performance['propulsao_basica'].current_level = 0.5
                self.valence_manager.valence_performance['propulsao_basica'].state = ValenceState.LEARNING
                self.valence_manager.active_valences.add('propulsao_basica')
            
    def emergency_stabilization(self):
        """CORREÇÃO para estabilidade"""
        self.critic.weights.stability = 0.35
        self.critic.weights.propulsion = 0.30
        self.critic.weights.coordination = 0.18
        self.critic.weights.efficiency = 0.17
    
    def override_valence_update(self):
        """Intercepta atualizações de valência para prevenir regressão"""
        original_update_valences = self.valence_manager.update_valences
        def protected_update_valences(metrics):
            protected_metrics = metrics.copy()
            if protected_metrics.get('distance', 0) < 0:
                protected_metrics['distance'] = 0.1
            if protected_metrics.get('speed', 0) < 0:
                protected_metrics['speed'] = 0.05
            if protected_metrics.get('distance', 0) > 0:
                protected_metrics['movement_bonus'] = 0.5
            return original_update_valences(protected_metrics)
        self.valence_manager.update_valences = protected_update_valences
    
    def _generate_comprehensive_report(self):
        """Relatório completo com status do critic"""
        valence_status = self.valence_manager.get_valence_status()
        consistency_metrics = self._calculate_training_consistency()
        current_irl_weights = self.valence_manager.get_irl_weights()

        self.logger.info("=" * 70)
        self.logger.info(f"📊 RELATÓRIO DPG - Episódio {self.episode_count}")
        self.logger.info(f"🎯 Progresso Geral: {valence_status['overall_progress']:.1%}")
        self.logger.info(f"📈 Consistência: {consistency_metrics['overall_consistency']:.1%}")
        
        irl_active = len(current_irl_weights) > 0 and any(w > 0 for w in current_irl_weights.values())
        irl_status = "🟢 ATIVO" if irl_active else "⚫ INATIVO"
        
        self.logger.info(f"🧠 IRL: {irl_status} | Critic Funcional: 🟢 ATIVO")
        self.logger.info(f"   Influência IRL: {self.critic.weights.irl_influence:.1%}")
        self.logger.info(f"   Pesos Critic: S:{self.critic.weights.stability:.2f} P:{self.critic.weights.propulsion:.2f} C:{self.critic.weights.coordination:.2f} E:{self.critic.weights.efficiency:.2f}")

        if irl_active:
            self.activate_irl_guidance()

        self.logger.info(f"🔧 Valências Ativas: {len(valence_status['active_valences'])}")
        self.logger.info(f"💾 Experiências: {getattr(self.buffer_manager, 'experience_count', 0)}")

        self.logger.info("📈 ESTADO DAS VALÊNCIAS:")
        for valence_name, details in valence_status["valence_details"].items():
            state_icon = {
                "inactive": "⚫", "learning": "🟡", "consolidating": "🟠",
                "mastered": "🟢", "regressing": "🔴"
            }.get(details["state"], "⚫")

            self.logger.info(
                f"   {state_icon} {valence_name}: {details['current_level']:.1%} / "
                f"{details['target_level']:.1%} ({details['state']}) "
                f"consistência: {details.get('consistency', 0):.1%}"
            )

        # Missões ativas
        if valence_status["current_missions"]:
            self.logger.info("🎯 MISSÕES ATIVAS:")
            for mission in valence_status["current_missions"]:
                self.logger.info(
                    f"   🎯 {mission['valence']}: +{mission['progress']} "
                    f"({mission['episodes_remaining']} episódios restantes)"
                )

        # Métricas de performance
        self.logger.info("📈 MÉTRICAS DE TREINAMENTO:")
        self.logger.info(f"   📊 Recompensa média: {consistency_metrics['avg_reward']:.2f}")
        self.logger.info(f"   🏃 Distância média: {consistency_metrics['avg_distance']:.2f}m")
        self.logger.info(f"   🎯 Taxa de sucesso: {consistency_metrics['success_rate']:.1%}")
        self.logger.info(f"   🔄 Bônus de missão: {self.mission_bonus_multiplier:.2f}x")
        self.logger.info("=" * 70)