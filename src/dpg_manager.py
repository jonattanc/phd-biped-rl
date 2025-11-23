# dpg_manager.py 
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from dpg_valence import ValenceManager, ValenceState
from dpg_reward import RewardCalculator
from dpg_buffer import AdaptiveBufferManager, Experience
from collections import deque
import time

@dataclass
class AdaptiveCriticWeights:
    """Pesos do crítico dinamicamente ajustáveis"""
    propulsion: float = 0.3
    stability: float = 0.3  
    coordination: float = 0.25
    efficiency: float = 0.15
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "propulsion": self.propulsion,
            "stability": self.stability,
            "coordination": self.coordination, 
            "efficiency": self.efficiency
        }

class AdaptiveCritic:
    """Crítico que se adapta às necessidades do buffer e valências"""
    
    def __init__(self, logger):
        self.logger = logger
        self.weights = AdaptiveCriticWeights()
        self.performance_history = deque(maxlen=100)
        self.adaptation_rate = 0.02
        self._last_valence_analysis = {}
        
    def analyze_valence_needs(self, valence_status: Dict, buffer_status: Dict) -> Dict[str, float]:
        """Analisa quais componentes precisam de mais foco baseado em valências e buffer"""
        adjustments = {
            "propulsion": 0.0,
            "stability": 0.0,
            "coordination": 0.0,
            "efficiency": 0.0
        }
        
        try:
            valence_details = valence_status.get('valence_details', {})
            buffer_metrics = buffer_status.get('adaptive_status', {})
            
            # ANÁLISE DE DEFICIÊNCIAS NAS VALÊNCIAS
            low_valences = []
            for name, details in valence_details.items():
                current_level = details.get('current_level', 0)
                target_level = details.get('target_level', 0.5)
                
                if current_level < target_level * 0.6:  
                    low_valences.append((name, current_level, target_level))
            
            # MAPEAMENTO VALÊNCIA
            for valence_name, current, target in low_valences:
                deficit = target - current
                
                if 'movimento' in valence_name or 'propulsao' in valence_name:
                    adjustments["propulsion"] += deficit * 0.4
                elif 'estabilidade' in valence_name:
                    adjustments["stability"] += deficit * 0.5
                elif 'coordenacao' in valence_name:
                    adjustments["coordination"] += deficit * 0.6
                elif 'eficiencia' in valence_name:
                    adjustments["efficiency"] += deficit * 0.3
            
            # ANÁLISE DO BUFFER
            buffer_quality = buffer_metrics.get('avg_quality', 0.5)
            outdated_experiences = buffer_metrics.get('outdated_experiences', 0)
            total_experiences = buffer_metrics.get('total_experiences', 1)
            
            # Se muitas experiências desatualizadas, focar em componentes estáveis
            outdated_ratio = outdated_experiences / total_experiences
            if outdated_ratio > 0.3:
                adjustments["stability"] += 0.2
                adjustments["coordination"] += 0.1
            
            # Se qualidade média baixa, focar em fundamentos
            if buffer_quality < 0.4:
                adjustments["propulsion"] += 0.3
                adjustments["stability"] += 0.2
            
            # DETECÇÃO DE ESTAGNAÇÃO
            if len(self.performance_history) > 10:
                recent_performance = list(self.performance_history)[-5:]
                avg_recent = np.mean([p.get('reward', 0) for p in recent_performance])
                
                if avg_recent < 10:  
                    # Foco agressivo em movimento básico
                    adjustments["propulsion"] += 0.5
                    adjustments["stability"] += 0.3
            
        except Exception as e:
            self.logger.warning(f"Erro na análise do crítico: {e}")
            
        return adjustments
    
    def update_weights(self, adjustments: Dict[str, float], current_group: int):
        """Atualiza pesos com taxa adaptativa baseada no grupo"""
        # Taxa de aprendizado por grupo
        group_rates = {1: 0.03, 2: 0.02, 3: 0.01}
        rate = group_rates.get(current_group, 0.02)
        
        # Aplica ajustes
        self.weights.propulsion = max(0.1, min(0.7, 
            self.weights.propulsion + adjustments["propulsion"] * rate))
        self.weights.stability = max(0.1, min(0.6, 
            self.weights.stability + adjustments["stability"] * rate))
        self.weights.coordination = max(0.1, min(0.6, 
            self.weights.coordination + adjustments["coordination"] * rate))
        self.weights.efficiency = max(0.05, min(0.4, 
            self.weights.efficiency + adjustments["efficiency"] * rate))
        
        # Normaliza
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Garante soma 1.0 mantendo proporções relativas"""
        total = (self.weights.propulsion + self.weights.stability + 
                self.weights.coordination + self.weights.efficiency)
        
        if total > 0:
            self.weights.propulsion /= total
            self.weights.stability /= total
            self.weights.coordination /= total
            self.weights.efficiency /= total
    
    def get_optimization_advice(self) -> Dict:
        """Retorna conselhos para otimização do sistema"""
        max_component = max(self.weights.to_dict().items(), key=lambda x: x[1])
        
        advice = {
            "focus_component": max_component[0],
            "focus_strength": max_component[1],
            "recommended_buffer_strategy": "",
            "valence_priority": []
        }
        
        if max_component[0] == "propulsion":
            advice["recommended_buffer_strategy"] = "movement_focus"
            advice["valence_priority"] = ["movimento_basico", "propulsao_basica"]
        elif max_component[0] == "stability":
            advice["recommended_buffer_strategy"] = "stability_focus" 
            advice["valence_priority"] = ["estabilidade_postural"]
        elif max_component[0] == "coordination":
            advice["recommended_buffer_strategy"] = "coordination_focus"
            advice["valence_priority"] = ["coordenacao_fundamental"]
        else:
            advice["recommended_buffer_strategy"] = "efficiency_focus"
            advice["valence_priority"] = ["eficiencia_biomecanica"]
            
        return advice

class AdaptiveCrutchSystem:
    """Sistema de muletas que se integra com valências e buffer"""
    
    def __init__(self, logger):
        self.logger = logger
        self.crutch_level = 1.0
        self.current_stage = 0
        self.performance_window = deque(maxlen=20)
        self.valence_progress_history = deque(maxlen=50)
        
        # Configurações adaptativas
        self.stage_thresholds = [0.8, 0.6, 0.4, 0.2]
        self.aggression_factors = {1: 0.95, 2: 0.93, 3: 0.90, 4: 0.85}
    
    def update_crutch_level(self, valence_status: Dict, buffer_status: Dict, 
                          current_episode: int) -> float:
        """Atualiza nível de muleta baseado em múltiplos fatores"""
        
        # FATOR 1: Progresso das valências (40% de peso)
        valence_progress = valence_status.get('overall_progress', 0)
        active_valences = len(valence_status.get('active_valences', []))
        valence_factor = (1.0 - valence_progress) * 0.6 + (1.0 - min(active_valences / 5.0, 1.0)) * 0.4
        
        # FATOR 2: Qualidade do buffer (30% de peso)
        buffer_quality = buffer_status.get('adaptive_status', {}).get('avg_quality', 0.5)
        buffer_utilization = buffer_status.get('adaptive_status', {}).get('buffer_utilization', 0.5)
        buffer_factor = (1.0 - buffer_quality) * 0.7 + (1.0 - buffer_utilization) * 0.3
        
        # FATOR 3: Performance recente (30% de peso)
        if self.performance_window:
            avg_recent = np.mean(list(self.performance_window))
            performance_factor = 1.0 - min(avg_recent / 100.0, 1.0)
        else:
            performance_factor = 1.0
        
        # CÁLCULO DO NOVO NÍVEL
        new_level = (
            valence_factor * 0.4 +
            buffer_factor * 0.3 + 
            performance_factor * 0.3
        )
        
        # AGESSIVIDADE POR ESTÁGIO
        aggression = self.aggression_factors.get(self.current_stage, 0.9)
        self.crutch_level = max(0.05, self.crutch_level * aggression)
        
        # PROTEÇÃO CONTRA REDUÇÃO PRECOCE
        if current_episode < 1000 and self.crutch_level < 0.3:
            self.crutch_level = 0.3
        elif current_episode < 500 and self.crutch_level < 0.5:
            self.crutch_level = 0.5
            
        self._update_stage()
        return self.crutch_level
    
    def _update_stage(self):
        """Atualiza estágio das muletas"""
        for i, threshold in enumerate(self.stage_thresholds):
            if self.crutch_level > threshold:
                self.current_stage = i
                return
        self.current_stage = len(self.stage_thresholds)
    
    def get_reward_multiplier(self) -> float:
        """Retorna multiplicador de recompensa baseado no estágio"""
        multipliers = [3.0, 2.0, 1.5, 1.2, 1.0]
        return multipliers[self.current_stage]
    
    def add_performance_sample(self, reward: float):
        """Adiciona amostra de performance para cálculo adaptativo"""
        self.performance_window.append(min(reward, 200.0))  

class DPGManager:
    """DPG Manager com integração total ao buffer adaptativo"""
    
    def __init__(self, logger, robot, reward_system, state_dim=10, action_dim=6):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.enabled = True
        
        # SISTEMAS PRINCIPAIS OTIMIZADOS
        self.valence_manager = ValenceManager(logger, {})
        self.reward_calculator = RewardCalculator(logger, {})
        self.buffer_manager = AdaptiveBufferManager(logger, {}, max_experiences=2000)
        self.adaptive_critic = AdaptiveCritic(logger)
        self.crutch_system = AdaptiveCrutchSystem(logger)
        
        # CONTROLE DE ESTADO
        self.episode_count = 0
        self.current_group = 1
        self.current_terrain_type = "normal"
        self.overall_progress = 0.0
        self.performance_history = deque(maxlen=100)
        
        # CONFIGURAÇÕES ADAPTATIVAS
        self.optimization_intervals = {
            "critic_update": 10,      
            "buffer_reevaluation": 25,  
            "crutch_update": 5,       
            "valence_force_update": 50 
        }
        
        self._last_optimization = {
            "critic": 0,
            "buffer": 0, 
            "crutch": 0,
            "valence": 0
        }

        # CONTROLE DE APRENDIZADO
        self.learning_enabled = True
        self.min_batch_size = 32
        self.training_interval = 10  
        self._last_training_episode = 0

        # CONTROLE DO SPARSE SUCCESS PROGRESSIVO
        self.sparse_success_transition_episode = 2000
        
    def calculate_reward(self, sim, action) -> float:
        """Cálculo de recompensa com integração total dos sistemas"""
        if not self.enabled:
            return 0.0
            
        self._current_episode_action = action
        
        # OBTER ESTADO ATUAL DO SISTEMA
        valence_status = self.valence_manager.get_valence_status()
        buffer_status = self.buffer_manager.get_adaptive_status()
        
        # CALCULAR RECOMPENSA BASE
        phase_info = self._build_phase_info(valence_status, buffer_status)
        base_reward = self.reward_calculator.calculate(sim, action, phase_info)
        
        # APLICAR MULTIPLICADOR ADAPTATIVO DE MULETAS
        crutch_multiplier = self.crutch_system.get_reward_multiplier()
        adapted_reward = base_reward * crutch_multiplier
        
        # BÔNUS DE ATIVAÇÃO PARA VALÊNCIAS NOVAS
        valence_bonus = self._calculate_valence_activation_bonus(valence_status)
        final_reward = max(adapted_reward + valence_bonus, 0.0)
        
        # OBTER DISTÂNCIA DO EPISÓDIO
        try:
            # Tenta obter a distância do sim
            if hasattr(sim, 'episode_distance'):
                distance = sim.episode_distance
            elif hasattr(sim, 'get_episode_distance'):
                distance = sim.get_episode_distance()
            else:
                distance = 0.0
        except:
            distance = 0.0
        
        # REGISTRAR PARA OTIMIZAÇÃO
        self.crutch_system.add_performance_sample(final_reward)
        self.performance_history.append({
            'reward': final_reward,
            'valence_progress': valence_status.get('overall_progress', 0),
            'crutch_level': self.crutch_system.crutch_level,
            'distance': distance
        })
        
        return final_reward

    def _build_phase_info(self, valence_status: Dict, buffer_status: Dict) -> Dict:
        """Constroi informações de fase integradas"""
        critic_advice = self.adaptive_critic.get_optimization_advice()

        return {
            'group_level': self.current_group,
            'group_name': 'adaptive_system',
            'valence_weights': self.valence_manager.get_valence_weights_for_reward(),
            'enabled_components': self.valence_manager.get_active_reward_components(),
            'target_speed': self._get_adaptive_target_speed(valence_status),
            'learning_progress': valence_status.get('overall_progress', 0),
            'valence_status': valence_status,
            'critic_advice': critic_advice,
            'buffer_quality': buffer_status.get('avg_quality', 0.5),
            'crutch_level': self.crutch_system.crutch_level,
            'sparse_success_enabled': valence_status.get('overall_progress', 0) > 0.4
        }

    def update_phase_progression(self, episode_results):
        """INTEGRAÇÃO TOTAL ENTRE SISTEMAS"""
        if not self.enabled:
            return

        self.episode_count += 1

        try:
            # ATUALIZAÇÃO DE VALÊNCIAS (SEMPRE)
            self._update_valence_system(episode_results)
            valence_status = self.valence_manager.get_valence_status()
            self.overall_progress = valence_status.get('overall_progress', 0)

            # ATUALIZAÇÃO DO CRÍTICO (COM INTERVALO ADAPTATIVO)
            if self._should_optimize("critic"):
                self._update_adaptive_critic(valence_status)
                self._last_optimization["critic"] = self.episode_count

            # ATUALIZAÇÃO DO BUFFER (COM INTEGRAÇÃO DO CRÍTICO)
            if self._should_optimize("buffer"):
                self._update_adaptive_buffer(valence_status)
                self._last_optimization["buffer"] = self.episode_count

            # ATUALIZAÇÃO DE MULETAS (RESPONSIVA)
            if self._should_optimize("crutch"):
                self._update_adaptive_crutches(valence_status)
                self._last_optimization["crutch"] = self.episode_count

            # ATIVAÇÃO DE EMERGÊNCIA SE ESTAGNADO
            if self.episode_count % 100 == 0 and self.overall_progress < 0.2:
                self._emergency_optimization(episode_results)

            # DETECÇÃO AUTOMÁTICA DE GRUPO
            self._update_current_group(valence_status)
            
            if self.episode_count % 500 == 0:
                self._generate_comprehensive_report()

        except Exception as e:
            self.logger.error(f"❌ ERRO em update_phase_progression: {e}")

    def _update_valence_system(self, episode_results):
        """Atualização robusta do sistema de valências"""
        extended_metrics = self._prepare_valence_metrics_optimized(episode_results)
        self.valence_manager.update_valences(extended_metrics)
        
        # FORÇAR ATIVAÇÃO SE POUCO PROGRESSO
        if self.episode_count % self.optimization_intervals["valence_force_update"] == 0:
            self._force_valence_activation(episode_results)

    def _update_adaptive_critic(self, valence_status: Dict):
        """Atualização do crítico com análise integrada"""
        buffer_status = self.buffer_manager.get_adaptive_status()
        
        # ANALISAR NECESSIDADES
        adjustments = self.adaptive_critic.analyze_valence_needs(
            valence_status, buffer_status
        )
        
        # ATUALIZAR PESOS
        self.adaptive_critic.update_weights(adjustments, self.current_group)
        
    def _update_adaptive_buffer(self, valence_status: Dict):
        """Atualização do buffer com critérios do crítico"""
        critic_weights = self.adaptive_critic.weights.to_dict()
        
        self.buffer_manager.update_quality_criteria(
            critic_weights=critic_weights,
            current_group=self.current_group,
            current_episode=self.episode_count,
            force_reevaluation=True  
        )
        
        # OTIMIZAR TAMANHO DO BUFFER BASEADO NO PROGRESSO
        buffer_status = self.buffer_manager.get_adaptive_status()
        if buffer_status.get('total_experiences', 0) > 3000 and self.overall_progress > 0.5:
            self.buffer_manager.main_buffer.cleanup_low_quality(0.4)

    def _update_adaptive_crutches(self, valence_status: Dict):
        """Atualização adaptativa do sistema de muletas"""
        buffer_status = self.buffer_manager.get_adaptive_status()
        
        self.crutch_system.update_crutch_level(
            valence_status=valence_status,
            buffer_status=buffer_status,
            current_episode=self.episode_count
        )
        
    def _should_optimize(self, system: str) -> bool:
        """Verifica se deve otimizar um sistema específico"""
        interval = self.optimization_intervals.get(f"{system}_update", 20)
        return (self.episode_count - self._last_optimization[system]) >= interval

    def _update_current_group(self, valence_status: Dict):
        """Atualiza grupo atual baseado em valências mestras"""
        valence_details = valence_status.get('valence_details', {})
        
        mastered_count = sum(
            1 for details in valence_details.values() 
            if details.get('state') == 'mastered'
        )
        
        if mastered_count >= 4 and self.current_group < 3:
            self.current_group = 3
            self.logger.info("🚀 PROMOÇÃO: Grupo 3 ativado")
        elif mastered_count >= 2 and self.current_group < 2:
            self.current_group = 2  
            self.logger.info("📈 PROMOÇÃO: Grupo 2 ativado")
        elif self.current_group > 1 and mastered_count < 1:
            self.current_group = 1
            self.logger.warning("⚠️  REGRESSÃO: Grupo 1 reativado")

    def _emergency_optimization(self, episode_results):
        """Otimização de emergência para casos de estagnação"""
        
        # ESTRATÉGIAS AGRESSIVAS
        self.crutch_system.crutch_level = min(1.0, self.crutch_system.crutch_level + 0.3)
        
        # FOCO MÁXIMO EM MOVIMENTO BÁSICO
        self.adaptive_critic.weights.propulsion = 0.7
        self.adaptive_critic.weights.stability = 0.2
        self.adaptive_critic.weights.coordination = 0.1
        self.adaptive_critic.weights.efficiency = 0.0
        self.adaptive_critic._normalize_weights()
        
        # FORÇAR REAVALIAÇÃO COMPLETA DO BUFFER
        self.buffer_manager.update_quality_criteria(
            critic_weights=self.adaptive_critic.weights.to_dict(),
            current_group=1,  # Forçar grupo 1
            current_episode=self.episode_count,
            force_reevaluation=True
        )

    def should_train(self, current_episode: int) -> bool:
        """Verifica se deve realizar treinamento"""
        if not self.learning_enabled:
            return False
            
        buffer_status = self.buffer_manager.get_adaptive_status()
        total_experiences = buffer_status.get('total_experiences', 0)
        
        return (total_experiences >= self.min_batch_size and 
                current_episode - self._last_training_episode >= self.training_interval)
    
    def get_training_batch(self, batch_size: int = None) -> List[Experience]:
        """Obtém batch para treinamento"""
        if batch_size is None:
            batch_size = self.min_batch_size
            
        return self.buffer_manager.sample(batch_size, self.current_group)
    
    def on_training_completed(self, episode: int):
        """Callback quando treinamento é completado"""
        self._last_training_episode = episode
        
    def enable_learning(self, enable: bool = True):
        """Ativa/desativa aprendizado"""
        self.learning_enabled = enable
        self.logger.info(f"📚 Aprendizado do DPG {'ativado' if enable else 'desativado'}")
        
    def _calculate_valence_activation_bonus(self, valence_status: Dict) -> float:
        """Calcula bônus por ativação de novas valências"""
        bonus = 0.0
        valence_details = valence_status.get('valence_details', {})
        
        for name, details in valence_details.items():
            if details.get('state') == ValenceState.LEARNING:
                episodes_active = details.get('episodes_active', 0)
                if episodes_active < 10:  
                    bonus += (10 - episodes_active) * 2.0
                    
        return bonus

    def _get_adaptive_target_speed(self, valence_status: Dict) -> float:
        """Velocidade alvo adaptativa baseada em múltiplos fatores"""
        base_speed = 0.1 + (self.overall_progress * 1.5)
        
        # AJUSTE POR COMPONENTE DO CRÍTICO
        critic_focus = self.adaptive_critic.get_optimization_advice()["focus_component"]
        
        if critic_focus == "propulsion":
            base_speed *= 1.3
        elif critic_focus == "stability":
            base_speed *= 0.8  # Mais devagar para focar em estabilidade
        elif critic_focus == "coordination":
            base_speed *= 1.1
            
        return min(base_speed, 2.0)

    def _force_valence_activation(self, episode_results):
        """Força ativação de valências quando há pouco progresso"""
        try:
            valence_status = self.valence_manager.get_valence_status()
            active_count = len(valence_status.get('active_valences', []))
            
            # Se poucas valências ativas, forçar ativação das básicas
            if active_count < 2:
                self.logger.warning("🚨 FORÇANDO ATIVAÇÃO DE VALÊNCIAS BÁSICAS")
                
                # Forçar movimento_basico se há algum movimento
                distance = episode_results.get("distance", 0)
                if distance > 0.01:
                    self.valence_manager._ensure_valence_active("movimento_basico", min_level=0.2)
                
                # Forçar estabilidade_postural se está razoavelmente estável
                roll = abs(episode_results.get("roll", 0))
                pitch = abs(episode_results.get("pitch", 0))
                stability = 1.0 - min((roll + pitch) / 1.0, 1.0)
                if stability > 0.5:
                    self.valence_manager._ensure_valence_active("estabilidade_postural", min_level=0.15)
                
                # Forçar coordenação se há alternância
                alternating = episode_results.get("alternating", False)
                if alternating:
                    self.valence_manager._ensure_valence_active("coordenacao_fundamental", min_level=0.1)
                    
        except Exception as e:
            self.logger.error(f"Erro ao forçar ativação de valências: {e}")
            
    def get_integrated_status(self) -> Dict:
        """Status completo com integração de todos os sistemas"""
        valence_status = self.valence_manager.get_valence_status()
        buffer_status = self.buffer_manager.get_adaptive_status()
        critic_advice = self.adaptive_critic.get_optimization_advice()
        
        return {
            "system_integration": {
                "overall_progress": self.overall_progress,
                "current_group": self.current_group,
                "episode_count": self.episode_count,
                "crutch_stage": self.crutch_system.current_stage,
                "crutch_level": self.crutch_system.crutch_level,
                "critic_focus": critic_advice["focus_component"],
                "buffer_quality": buffer_status.get('avg_quality', 0),
                "valence_active_count": len(valence_status.get('active_valences', [])),
                "integration_score": self._calculate_integration_score(
                    valence_status, buffer_status
                )
            },
            "valence_system": valence_status,
            "buffer_system": buffer_status,
            "critic_system": {
                "weights": self.adaptive_critic.weights.to_dict(),
                "advice": critic_advice
            },
            "crutch_system": {
                "level": self.crutch_system.crutch_level,
                "stage": self.crutch_system.current_stage,
                "multiplier": self.crutch_system.get_reward_multiplier()
            }
        }

    def _calculate_integration_score(self, valence_status: Dict, buffer_status: Dict) -> float:
        """Calcula quão bem integrados estão os sistemas"""
        scores = []
        
        # ALINHAMENTO CRÍTICO-VALÊNCIA
        critic_focus = self.adaptive_critic.get_optimization_advice()["focus_component"]
        valence_priority = self.adaptive_critic.get_optimization_advice()["valence_priority"]
        
        # Verifica se valências prioritárias estão ativas
        active_valences = valence_status.get('active_valences', [])
        priority_active = sum(1 for v in valence_priority if v in active_valences)
        scores.append(priority_active / len(valence_priority) if valence_priority else 0.5)
        
        # QUALIDADE DO BUFFER
        buffer_quality = buffer_status.get('avg_quality', 0.5)
        scores.append(buffer_quality)
        
        # PROGRESSO GERAL
        scores.append(valence_status.get('overall_progress', 0))
        
        return np.mean(scores)

    def _prepare_valence_metrics_optimized(self, episode_results):
        """Preparação de métricas COMPLETA"""
        extended = episode_results.copy()

        try:
            # MÉTRICAS ESSENCIAIS PARA TODAS AS VALÊNCIAS
            try:
                if "distance" in episode_results and episode_results["distance"] is not None:
                    raw_distance = episode_results["distance"]
                elif hasattr(self.robot, 'episode_distance'):
                    raw_distance = self.robot.episode_distance
                elif hasattr(self.robot, 'get_episode_distance'):
                    raw_distance = self.robot.get_episode_distance()
                else:
                    raw_distance = 0.0

                # Garantir que é numérico
                if not isinstance(raw_distance, (int, float)):
                    raw_distance = 0.0

                extended["distance"] = max(float(raw_distance), 0.0)

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
                "positive_movement_rate": 1.0 if extended["distance"] > 0.1 else 0.0
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
            extended["stride_efficiency"] = extended["distance"] / max(episode_results.get("steps", 1), 1)

            # MÉTRICAS DE MARCHA ROBUSTA
            extended["gait_robustness"] = 0.7
            extended["recovery_success"] = 1.0 if episode_results.get("success", False) else 0.0
            extended["speed_adaptation"] = 0.8
            extended["terrain_handling"] = 0.6

        except Exception as e:
            self.logger.warning(f"Erro ao preparar métricas: {e}")

        return extended
    
    def store_experience(self, state, action, reward, next_state, done, episode_results):
        """Armazena experiência no buffer adaptativo"""
        if not self.enabled:
            return False

        try:
            # Atualizar contador de episódios no buffer manager
            self.buffer_manager.episode_count = self.episode_count

            # Preparar métricas para qualidade
            metrics = self._prepare_valence_metrics_optimized(episode_results)

            # Dados da experiência
            experience_data = {
                "state": state,
                "action": action, 
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "metrics": metrics,
                "group_level": self.current_group,
                "phase_info": self._build_phase_info(
                    self.valence_manager.get_valence_status(),
                    self.buffer_manager.get_adaptive_status()
                )
            }

            # Armazenar no buffer
            success = self.buffer_manager.store_experience(experience_data)
                
            return success

        except Exception as e:
            self.logger.error(f"❌ ERRO ao armazenar experiência: {e}")
            return False
    
    def _generate_comprehensive_report(self):
        """RELATÓRIO COMPLETO"""
        try:
            integrated_status = self.get_integrated_status()
            buffer_status = self.buffer_manager.get_adaptive_status()
            valence_status = self.valence_manager.get_valence_status()

            # ACESSO SEGURO aos dados do buffer
            total_experiences = buffer_status.get('total_experiences', 0)
            avg_quality = buffer_status.get('avg_quality', 0.0)
            buffer_utilization = buffer_status.get('buffer_utilization', 0.0)
            outdated_experiences = buffer_status.get('outdated_experiences', 0)

            recent_rewards = [p['reward'] for p in list(self.performance_history)[-10:]]
            recent_distances = [p.get('distance', 0) for p in list(self.performance_history)[-10:]]
            avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_recent_distance = np.mean(recent_distances) if recent_distances else 0

            # CABEÇALHO E STATUS GERAL
            self.logger.info("=" * 60)
            self.logger.info(f"🎯 EPISÓDIO {self.episode_count} | GRUPO: {self.current_group}")
            self.logger.info(f"📊 PROGRESSO: {integrated_status['system_integration']['overall_progress']:.1%} | "
                            f"DISTÂNCIA: {avg_recent_distance:.3f}m")
            self.logger.info(f"    INTEGRAÇÃO: {integrated_status['system_integration']['integration_score']:.1%} | "
                            f"RECOMPENSA: {avg_recent_reward:.1f}")

            # SISTEMA DE CRÍTICO ADAPTATIVO
            critic_weights = self.adaptive_critic.weights.to_dict()
            critic_total = sum(critic_weights.values())
            critic_status = "✅" if 0.99 <= critic_total <= 1.01 else "❌"

            self.logger.info(f"🧠 CRÍTICO ADAPTATIVO {critic_status}")
            self.logger.info(f"   Propulsão: {critic_weights['propulsion']:.3f} | Estabilidade: {critic_weights['stability']:.3f}")
            self.logger.info(f"   Coordenação: {critic_weights['coordination']:.3f} | Eficiência: {critic_weights['efficiency']:.3f}")

            # SISTEMA DE VALÊNCIAS - DETALHAMENTO INTELIGENTE
            self.logger.info("📈 SISTEMA DE VALÊNCIAS:")

            for valence_name, details in valence_status["valence_details"].items():
                state = details['state']
                level = details['current_level']

                if state != "inactive" and level > 0.01:
                    state_icon = "🟢" if state == "learning" else "🟡" if state == "mastered" else "🔴"
                    progress_bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                    self.logger.info(f"   {state_icon} {valence_name:.<25} {progress_bar} {level:.1%} ({state})")

            # SISTEMA DE MULETAS ADAPTATIVAS
            stage_names = ["MÁXIMO", "ALTO", "MÉDIO", "BAIXO", "MÍNIMO"]
            current_stage = self.crutch_system.current_stage
            stage_name = stage_names[current_stage] if current_stage < len(stage_names) else "CRÍTICO"

            self.logger.info(f"🦯 MULETAS ADAPTATIVAS: {stage_name}")
            self.logger.info(f"    Nível: {self.crutch_system.crutch_level:.3f} |" 
                            f"   Multiplicador: {self.crutch_system.get_reward_multiplier():.1f}x")          

            # SISTEMA DE BUFFER ADAPTATIVO
            self.logger.info("💾 BUFFER ADAPTATIVO:")
            self.logger.info(f"   Experiências: {total_experiences} | "
                            f"Qualidade: {avg_quality:.3f}")
            self.logger.info(f"   Utilização: {buffer_utilization:.1%} | "
                            f"Desatualizadas: {outdated_experiences}")

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"❌ ERRO no relatório completo: {e}")
            # Relatório de emergência
            self.logger.info("=" * 60)
            self.logger.info(f"🎯 EPISÓDIO {self.episode_count} | GRUPO: {self.current_group}")
            self.logger.info("⚠️  Relatório parcial devido a erro")
            self.logger.info("=" * 60)