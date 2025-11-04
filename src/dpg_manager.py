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
        
        # Bonus se valência de estabilidade está alta
        stability_valence = valence_levels[0] if len(valence_levels) > 0 else 0.0
        return stability * (0.7 + stability_valence * 0.3)
    
    def _calculate_propulsion_score(self, metrics, valence_levels):
        """Calcula score de propulsão"""
        distance = metrics.get('distance', 0)
        velocity = metrics.get('speed', 0)
        
        distance_score = min(distance / 3.0, 1.0)
        velocity_score = min(abs(velocity) / 1.5, 1.0) if velocity > 0 else 0.0
        
        # Bonus se valência de propulsão está alta
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
            
        # Bonus se valência de coordenação está alta
        coordination_valence = valence_levels[2] if len(valence_levels) > 2 else 0.0
        return min(coordination_score * (0.8 + coordination_valence * 0.4), 1.0)
    
    def _calculate_efficiency_score(self, metrics, valence_levels):
        """Calcula score de eficiência"""
        efficiency = metrics.get('propulsion_efficiency', 0.5)
        energy_used = metrics.get('energy_used', 1.0)
        
        efficiency_score = efficiency
        if energy_used < 0.8:  # Baixo consumo de energia
            efficiency_score += 0.2
            
        # Bonus se valência de eficiência está alta
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
        if len(self.performance_history) < 20:
            return

        # Analisar correlação entre valências e desempenho
        recent_performance = self.performance_history[-20:]

        # Calcular quais valências estão correlacionadas com bom desempenho
        valence_correlations = [0.0] * self.valence_count

        for i in range(self.valence_count):
            valence_levels = [p['valence_levels'][i] for p in recent_performance]
            performance_scores = [p['final_value'] for p in recent_performance]

            if (len(valence_levels) > 1 and 
                np.std(valence_levels) > 1e-8 and 
                np.std(performance_scores) > 1e-8):

                correlation = np.corrcoef(valence_levels, performance_scores)[0,1]
                if not np.isnan(correlation):
                    valence_correlations[i] = max(correlation, 0.0)

        # Atualizar pesos baseado nas correlações
        total_correlation = sum(valence_correlations)
        if total_correlation > 0:
            new_weights = [corr / total_correlation for corr in valence_correlations]

            self.weights.stability = 0.9 * self.weights.stability + 0.1 * (new_weights[0] if len(new_weights) > 0 else 0.25)
            self.weights.propulsion = 0.9 * self.weights.propulsion + 0.1 * (new_weights[1] if len(new_weights) > 1 else 0.25)
            self.weights.coordination = 0.9 * self.weights.coordination + 0.1 * (new_weights[2] if len(new_weights) > 2 else 0.25)
            self.weights.efficiency = 0.9 * self.weights.efficiency + 0.1 * (new_weights[3] if len(new_weights) > 3 else 0.25)

        # Atualizar influência IRL baseado na utilidade
        irl_useful = any(p['final_value'] > p['base_value'] for p in recent_performance[-10:])
        if irl_useful:
            self.weights.irl_influence = min(self.weights.irl_influence + 0.02, 0.3)
        else:
            self.weights.irl_influence = max(self.weights.irl_influence - 0.01, 0.0)

class DPGManager:
    """DPG Manager com critic funcional sem PyTorch"""
    
    def __init__(self, logger, robot, reward_system, state_dim=10, action_dim=6):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system

        self.config = type('Config', (), {})()
        self.config.enabled = True
        self.config.valence_system = True
        
        # Componentes especializados
        self.valence_manager = ValenceManager(logger, {})
        self.reward_calculator = RewardCalculator(logger, {})
        self.buffer_manager = SmartBufferManager(logger, {})
        valence_count = len(self.valence_manager.valences)
        self.critic = ValenceAwareCritic(logger, valence_count)
        
        # Estado do sistema
        self.enabled = False
        self.learning_progress = 0.0
        self.performance_trend = 0.0
        self.episode_count = 0
        self.mission_bonus_multiplier = 1.0

        # Controles de frequência
        self.last_valence_update_episode = 0
        self.valence_update_interval = 5
        self.last_critic_update_episode = 0
        self.critic_update_interval = 50
        self.last_report_episode = 0
        self.report_interval = 200

        # Cache para performance
        self._cached_valence_weights = {}
        self._cached_irl_weights = {}
        self._cache_episode = 0
        self.episode_metrics_history = []

    def enable(self, enabled=True):
        """Ativa o sistema completo"""
        self.enabled = enabled
        if enabled:
            self.logger.info("Sistema DPG Adaptável ativado")

    def calculate_reward(self, sim, action) -> float:
        """Calcula recompensa com valências + IRL + Critic funcional"""
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
        
        # Recompensa base do sistema de valências
        base_reward = self.reward_calculator.calculate(sim, action, phase_info)
        state = self._extract_state(sim)
        valence_levels = self._extract_valence_levels(valence_status)
        episode_metrics = self._extract_episode_metrics(sim)
        
        critic_value = self.critic.predict_value(
            state, action, valence_levels, irl_weights, episode_metrics
        )
        
        # Combinar recompensas
        critic_weight = 0.2
        total_reward = base_reward * (1 - critic_weight) + critic_value * critic_weight
        total_reward *= self.mission_bonus_multiplier
        
        experience_data = {
            "state": state,
            "action": action,
            "reward": total_reward,
            "phase_info": phase_info,
            "metrics": episode_metrics,
            "group_level": phase_info['group_level'],
            "valence_data": {
                "active_valences": valence_status['active_valences'],
                "valence_weights": valence_weights,
                "irl_weights": irl_weights,
                "critic_value": critic_value,
                "critic_influence": critic_weight,
                "critic_weights": {
                    "stability": self.critic.weights.stability,
                    "propulsion": self.critic.weights.propulsion,
                    "coordination": self.critic.weights.coordination,
                    "efficiency": self.critic.weights.efficiency,
                    "irl_influence": self.critic.weights.irl_influence
                }
            }
        }
        self.buffer_manager.store_experience(experience_data)
        
        return total_reward

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
        
        # Bônus por valências específicas
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

    def _extract_metrics(self, sim):
        """Extrai métricas básicas da simulação"""
        return {
            "distance": getattr(sim, "episode_distance", 0),
            "speed": getattr(sim, "robot_x_velocity", 0),
            "roll": abs(getattr(sim, "robot_roll", 0)),
            "pitch": abs(getattr(sim, "robot_pitch", 0)),
        }

    def _prepare_valence_metrics(self, episode_results):
        """Prepara métricas estendidas para o sistema de valências"""
        extended = episode_results.copy()
        
        try:
            # Estabilidade composta
            roll = abs(episode_results.get("roll", 0))
            pitch = abs(episode_results.get("pitch", 0))
            extended["stability"] = 1.0 - min((roll + pitch) / 2.0, 1.0)
            
            # Taxa de movimento positivo
            distance = episode_results.get("distance", 0)
            extended["positive_movement_rate"] = 1.0 if distance > 0.1 else 0.0
            
            # Score de alternância
            left_contact = episode_results.get("left_contact", False)
            right_contact = episode_results.get("right_contact", False)
            extended["alternating_score"] = 1.0 if left_contact != right_contact else 0.3
            
        except Exception as e:
            self.logger.warning(f"Erro ao preparar métricas de valência: {e}")
        
        return extended

    def _calculate_training_consistency(self):
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
        
        # Métricas do critic
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
        self.episode_metrics_history.append({
            'reward': episode_results.get('reward', 0),
            'distance': episode_results.get('distance', 0),
            'speed': episode_results.get('speed', 0),
            'success': episode_results.get('success', False)
        })

        if len(self.episode_metrics_history) > 50:
            self.episode_metrics_history.pop(0)
            
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

        # Atualizar critic periodicamente
        if (self.episode_count - self.last_critic_update_episode) >= self.critic_update_interval:
            valence_status = self.valence_manager.get_valence_status()
            self.critic.update_weights(valence_status)
            self.last_critic_update_episode = self.episode_count

        if (self.episode_count - self.last_report_episode) >= self.report_interval:
            self._generate_comprehensive_report()
            self.last_report_episode = self.episode_count

    def _generate_comprehensive_report(self):
        """Relatório completo com status do critic"""
        valence_status = self.valence_manager.get_valence_status()
        consistency_metrics = self._calculate_training_consistency()
        irl_weights = self.valence_manager.get_irl_weights()

        self.logger.info("=" * 70)
        self.logger.info(f"📊 RELATÓRIO DPG - Episódio {self.episode_count}")
        self.logger.info(f"🎯 Progresso Geral: {valence_status['overall_progress']:.1%}")
        self.logger.info(f"📈 Consistência: {consistency_metrics['overall_consistency']:.1%}")
        
        # Status IRL + Critic
        irl_active = len(irl_weights) > 0
        irl_status = "🟢 ATIVO" if irl_active else "⚫ INATIVO"
        
        self.logger.info(f"🧠 IRL: {irl_status} | Critic Funcional: 🟢 ATIVO")
        self.logger.info(f"   Influência IRL: {self.critic.weights.irl_influence:.1%}")
        self.logger.info(f"   Pesos Critic: S:{self.critic.weights.stability:.2f} P:{self.critic.weights.propulsion:.2f} C:{self.critic.weights.coordination:.2f} E:{self.critic.weights.efficiency:.2f}")

        if irl_active:
            self.logger.info(f"   Pesos IRL: {irl_weights}")

        self.logger.info(f"🔧 Valências Ativas: {len(valence_status['active_valences'])}")
        self.logger.info(f"💾 Experiências: {getattr(self.buffer_manager, 'experience_count', 0)}")

        self.logger.info("\n📈 ESTADO DAS VALÊNCIAS:")
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
            self.logger.info("\n🎯 MISSÕES ATIVAS:")
            for mission in valence_status["current_missions"]:
                self.logger.info(
                    f"   🎯 {mission['valence']}: +{mission['progress']} "
                    f"({mission['episodes_remaining']} episódios restantes)"
                )

        # Métricas de performance
        self.logger.info("\n📈 MÉTRICAS DE TREINAMENTO:")
        self.logger.info(f"   📊 Recompensa média: {consistency_metrics['avg_reward']:.2f}")
        self.logger.info(f"   🏃 Distância média: {consistency_metrics['avg_distance']:.2f}m")
        self.logger.info(f"   🎯 Taxa de sucesso: {consistency_metrics['success_rate']:.1%}")
        self.logger.info(f"   🔄 Bônus de missão: {self.mission_bonus_multiplier:.2f}x")
        self.logger.info("=" * 70)

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