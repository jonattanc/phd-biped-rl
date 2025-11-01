# dpg_gait_phases.py
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import torch
import torch.nn as nn

class PhaseTransitionResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    REGRESSION = "regression"
    STAGNATION = "stagnation"
    DASS_TRANSITION = "dass_transition"
    IRL_ADAPTATION = "irl_adaptation"

@dataclass
class GaitPhaseConfig:
    """Configuração de uma fase da marcha para DPG com validação robusta"""

    name: str
    target_speed: float  # m/s
    enabled_components: List[str]
    component_weights: Dict[str, float]
    phase_duration: int  # episódios mínimos na fase
    transition_conditions: Dict[str, float]  # condições para transição
    skill_requirements: Dict[str, float]  # habilidades específicas necessárias
    regression_thresholds: Dict[str, float]  # limites para regressão
    dass_samples_required: int = 1000
    policy_compression_target: str = "(64,64)"
    transfer_learning_enabled: bool = True
    irl_confidence_threshold: float = 0.7
    hdpg_convergence_threshold: float = 0.8

class MultiHeadCritic(nn.Module):
    """Crítico multi-head para HDPG"""
    
    def __init__(self, input_dim, hidden_dims=[256, 256], num_heads=4):
        super(MultiHeadCritic, self).__init__()
        self.num_heads = num_heads
        
        # Camadas compartilhadas
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        
        # Heads especializados
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[1], 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_heads)
        ])
        
        # Módulo de atenção para combinar heads
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[1] + num_heads, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        shared_features = self.shared_layers(x)
        head_outputs = [head(shared_features) for head in self.heads]
        head_outputs_tensor = torch.cat(head_outputs, dim=-1)
        attention_input = torch.cat([shared_features, head_outputs_tensor], dim=-1)
        attention_weights = self.attention(attention_input)
        final_output = torch.sum(head_outputs_tensor * attention_weights, dim=-1, keepdim=True)
        
        return final_output, head_outputs_tensor, attention_weights
    
class GaitPhaseDPG:
    """
    Sistema DPG avançado com validação robusta, fallback adaptativo e métricas detalhadas
    """
    def __init__(self, logger, reward_system):
        self.logger = logger
        self.reward_system = reward_system
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.phases = []
        self.performance_history = [] 
        self.progression_history = []  
        self.max_progression_history = 50
        self.skill_assessment_history = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.phase_validation_history = []
        self.stagnation_counter = 0
        self.last_avg_distance = 0.0
        self.regression_count = 0

        self.transition_episodes = 0
        self.transition_active = False
        self.old_weights = None
        self.old_enabled_components = None
        self.transition_total_episodes = 8
        self.dass_samples = []
        self.learned_reward_model = None
        self.hdpg_adaptive_components = {}
        self.multi_head_critic = None
        self.policy_compression_history = []
        self.recent_critic_predictions = []  
        self.recent_critic_values = []       
        self.recent_actual_rewards = [] 
        
        self._initialize_adaptive_reward_components()
        self._initialize_enhanced_gait_phases()

    def _initialize_adaptive_reward_components(self):
        """Componentes baseados no paper HDPG"""
        self.adaptive_components = {
            "gait_quality": {
                "priority": 0.3, 
                "learning_rate": 1.0,
                "recent_rewards": [],
                "performance_metrics": []
            },
            "balance_stability": {
                "priority": 0.4, 
                "learning_rate": 1.2,
                "recent_rewards": [],
                "performance_metrics": []
            },
            "energy_efficiency": {
                "priority": 0.2, 
                "learning_rate": 0.8,
                "recent_rewards": [],
                "performance_metrics": []
            },
            "speed_tracking": {
                "priority": 0.1, 
                "learning_rate": 1.0,
                "recent_rewards": [],
                "performance_metrics": []
            }
        }
        
    def _calculate_dynamic_weights(self, recent_performance):
        """Pesos dinâmicos baseados na equação do HDPG"""
        priorities = {}
        
        for component, metrics in recent_performance.items():
            if not metrics['recent_rewards']:
                priorities[component] = self.adaptive_components[component]['priority']
                continue
                
            μ = np.mean(metrics['recent_rewards'])
            σ = np.var(metrics['recent_rewards']) 

            priorities[component] = (μ + np.exp(σ**2)) * self.adaptive_components[component]['learning_rate']
        
        total = sum(priorities.values())
        if total > 0:
            return {k: (len(priorities) * v / total) for k, v in priorities.items()}
        else:
            return {k: 1.0 for k in priorities.keys()}

    def _initialize_enhanced_gait_phases(self):
        """Progressão GRADUAL com fases intermediárias"""

        # FASE 0: Estabilidade Postural
        phase0 = GaitPhaseConfig(
            name="estabilidade_postural",
            target_speed=0.2,
            enabled_components=["stability_roll", "stability_pitch", "center_bonus", "success_bonus"],
            component_weights={
                "stability_roll": 0.6,
                "stability_pitch": 0.3,
                "center_bonus": 0.08,
                "success_bonus": 0.02,
            },
            phase_duration=15,
            transition_conditions={
                "min_success_rate": 0.15,
                "min_avg_distance": 0.1,
                "max_avg_roll": 0.8,
                "min_avg_steps": 3,
            },
            skill_requirements={
                "basic_balance": 0.3,
                "postural_stability": 0.2,
                "gait_initiation": 0.1,
            },
            regression_thresholds={
                "max_failures": 40,
                "min_success_rate": 0.05,
                "stagnation_episodes": 80,
            },
            dass_samples_required=500,  
            transfer_learning_enabled=False
        )

        # FASE 1: Marcha Básica (0.3-0.5m)
        phase1 = GaitPhaseConfig(
            name="marcha_basica", 
            target_speed=0.4,
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "success_bonus", "center_bonus",
            ],
            component_weights={
                "progress": 0.35,
                "distance_bonus": 0.25,
                "stability_roll": 0.2,
                "stability_pitch": 0.12,
                "alternating_foot_contact": 0.05,
                "success_bonus": 0.02,
                "center_bonus": 0.01,
            },
            phase_duration=20,
            transition_conditions={
                "min_success_rate": 0.25,
                "min_avg_distance": 0.5,    
                "max_avg_roll": 0.7,        
                "min_avg_steps": 6,
                "min_avg_speed": 0.1,
            },
            skill_requirements={
                "basic_balance": 0.4,
                "postural_stability": 0.3,
                "step_consistency": 0.1,    
            },
            regression_thresholds={
                "max_failures": 30,
                "min_success_rate": 0.1,
                "stagnation_episodes": 50,
            },
            dass_samples_required=800,
            transfer_learning_enabled=True
        )

        # FASE 2: Marcha Lenta Estável (0.5-1.0m)
        phase2 = GaitPhaseConfig(
            name="marcha_lenta_estavel",
            target_speed=0.6,
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "success_bonus", "center_bonus",
            ],
            component_weights={
                "progress": 0.4,
                "distance_bonus": 0.3,
                "stability_roll": 0.15,
                "stability_pitch": 0.08,
                "alternating_foot_contact": 0.04,
                "gait_pattern_cross": 0.02,
                "success_bonus": 0.01,
            },
            phase_duration=25,
            transition_conditions={
                "min_success_rate": 0.3,
                "min_avg_distance": 0.8,    
                "max_avg_roll": 0.6,        
                "min_avg_steps": 8,
                "min_avg_speed": 0.2,
                "min_alternating_score": 0.2,
            },
            skill_requirements={
                "basic_balance": 0.5,
                "postural_stability": 0.4,
                "step_consistency": 0.2,
            },
            regression_thresholds={
                "max_failures": 25,
                "min_success_rate": 0.15,
                "stagnation_episodes": 40,
            },
            dass_samples_required=1200,
            transfer_learning_enabled=True
        )

        # FASE 3: Marcha Confiante (1.0-2.0m)
        phase3 = GaitPhaseConfig(
            name="marcha_confiante",
            target_speed=0.8,
            enabled_components=[
                "progress", "distance_bonus", "stability_roll", 
                "stability_pitch", "alternating_foot_contact",
                "gait_pattern_cross", "foot_clearance", "success_bonus",
                "effort_square_penalty", "center_bonus", "pitch_forward_bonus",
            ],
            component_weights={
                "progress": 0.5,
                "distance_bonus": 0.25,
                "stability_roll": 0.08,
                "stability_pitch": 0.05,
                "alternating_foot_contact": 0.05,
                "gait_pattern_cross": 0.04,
                "foot_clearance": 0.03,
                "success_bonus": 0.02,
                "effort_square_penalty": 0.005,
                "center_bonus": 0.005,
                "pitch_forward_bonus": 0.01,
            },
            phase_duration=30,
            transition_conditions={
                "min_success_rate": 0.4,
                "min_avg_distance": 1.0,    
                "max_avg_roll": 0.6,       
                "min_avg_steps": 10,
                "min_avg_speed": 0.3,
                "min_alternating_score": 0.3,
                "min_gait_coordination": 0.2,
            },
            skill_requirements={
                "basic_balance": 0.6,
                "postural_stability": 0.5,
                "step_consistency": 0.3,
                "dynamic_balance": 0.4,
            },
            regression_thresholds={
                "max_failures": 20,
                "min_success_rate": 0.2,
                "stagnation_episodes": 30,
            },
            dass_samples_required=1500,
            transfer_learning_enabled=True
        )

        # FASE 4: Marcha Rápida (2.0-4.0m)
        phase4 = GaitPhaseConfig(
            name="marcha_rapida",
            target_speed=1.2,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "alternating_foot_contact", "gait_pattern_cross",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "effort_square_penalty", 
                "y_axis_deviation_square_penalty", "jerk_penalty",
            ],
            component_weights={
                "progress": 0.2,
                "stability_roll": 0.15,
                "stability_pitch": 0.1,
                "alternating_foot_contact": 0.08,
                "gait_pattern_cross": 0.08,
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.05,
                "success_bonus": 0.03,
                "distance_bonus": 0.04,
                "effort_square_penalty": 0.008,
                "y_axis_deviation_square_penalty": 0.01,
                "jerk_penalty": 0.005,
            },
            phase_duration=35,
            transition_conditions={
                "min_success_rate": 0.5,
                "min_avg_distance": 2.0,    
                "min_avg_speed": 0.6,
                "max_avg_roll": 0.4,
                "min_propulsion_efficiency": 0.3,
                "min_gait_coordination": 0.5,
                "consistency_count": 5,
            },
            skill_requirements={
                "balance_recovery": 0.4,
                "propulsive_phase": 0.3,
                "dynamic_balance": 0.5,
                "step_consistency": 0.4,
            },
            regression_thresholds={
                "max_failures": 15,
                "min_success_rate": 0.3,
                "stagnation_episodes": 25,
            },
            dass_samples_required=1800,
            transfer_learning_enabled=True
        )

        # FASE 5: Marcha Propulsiva (4.0-6.0m)
        phase5 = GaitPhaseConfig(
            name="marcha_propulsiva",
            target_speed=1.5,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "alternating_foot_contact", "gait_pattern_cross",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "effort_square_penalty", 
                "y_axis_deviation_square_penalty", "jerk_penalty",
            ],
            component_weights={
                "progress": 0.4,
                "stability_roll": 0.12,
                "stability_pitch": 0.08,
                "alternating_foot_contact": 0.07,
                "gait_pattern_cross": 0.07,
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.06,
                "success_bonus": 0.04,
                "distance_bonus": 0.05,
                "effort_square_penalty": 0.01,
                "y_axis_deviation_square_penalty": 0.012,
                "jerk_penalty": 0.006,
            },
            phase_duration=40,
            transition_conditions={
                "min_success_rate": 0.6,
                "min_avg_distance": 4.0,    
                "min_avg_speed": 0.8,
                "max_avg_roll": 0.3,
                "min_propulsion_efficiency": 0.4,
                "min_gait_coordination": 0.6,
                "consistency_count": 8,
            },
            skill_requirements={
                "balance_recovery": 0.5,
                "propulsive_phase": 0.4,
                "dynamic_balance": 0.6,
                "energy_efficiency": 0.4,
            },
            regression_thresholds={
                "max_failures": 12,
                "min_success_rate": 0.4,
                "stagnation_episodes": 20,
            },
            dass_samples_required=2000,
            transfer_learning_enabled=True
        )

        # FASE 6: Marcha Eficiente (6.0-9.0m - OBJETIVO FINAL)
        phase6 = GaitPhaseConfig(
            name="marcha_eficiente",
            target_speed=1.8,
            enabled_components=[
                "progress", "stability_roll", "stability_pitch",
                "foot_clearance", "pitch_forward_bonus", "success_bonus",
                "distance_bonus", "gait_pattern_cross", "effort_square_penalty",
                "jerk_penalty", "center_bonus", "warning_penalty",
            ],
            component_weights={
                "progress": 0.15,
                "stability_roll": 0.1,
                "stability_pitch": 0.07,
                "foot_clearance": 0.06,
                "pitch_forward_bonus": 0.07,
                "success_bonus": 0.05,
                "distance_bonus": 0.06,
                "gait_pattern_cross": 0.05,
                "effort_square_penalty": 0.012,
                "jerk_penalty": 0.008,
                "center_bonus": 0.02,
                "warning_penalty": 0.015,
            },
            phase_duration=25,
            transition_conditions={
                "min_success_rate": 0.7,
                "min_avg_distance": 6.0,    
                "min_avg_speed": 1.0,
                "max_avg_roll": 0.25,
                "min_energy_efficiency": 0.6,
                "min_gait_consistency": 0.7,
                "consistency_count": 10,
            },
            skill_requirements={
                "energy_efficiency": 0.6,
                "dynamic_balance": 0.7,
                "step_consistency": 0.6,
                "gait_coordination": 0.7,
            },
            regression_thresholds={
                "max_failures": 8,
                "min_success_rate": 0.5,
                "stagnation_episodes": 15,
            },
            dass_samples_required=2500,
            transfer_learning_enabled=True
        )

        self.phases = [phase0, phase1, phase2, phase3, phase4, phase5, phase6]

    def update_phase(self, episode_results: Dict) -> PhaseTransitionResult:
        """Atualiza fase com validação HDPG completa"""
    
        if self.transition_active:
            return self._update_transition_progress()

        enhanced_results = self._enhance_episode_results(episode_results)

        self.episodes_in_phase += 1
        self.performance_history.append(enhanced_results)
        self.progression_history.append(enhanced_results)

        if len(self.progression_history) > self.max_progression_history:
            self.progression_history.pop(0)
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

        self._update_success_failure_counters(enhanced_results)
        self._collect_dass_samples(enhanced_results)  
        if self.current_phase >= 1:
            self._update_adaptive_components(enhanced_results)
        if self.current_phase >= 2 and len(self.dass_samples) >= 500:
            self._learn_reward_via_irl()
        if self.current_phase >= 3:
            self._update_hdpg_weights()
        if len(self.progression_history) >= 5:  
            can_advance = self._check_enhanced_phase_advancement()

            if can_advance and not self.transition_active:
                self.logger.info("🎯 CONDIÇÕES HDPG ATENDIDAS - Transição aprovada!")
                return self._start_gradual_transition()

        return PhaseTransitionResult.SUCCESS

    def _enhance_episode_results(self, episode_results: Dict) -> Dict:
        """Adiciona métricas calculadas aos resultados do episódio"""
        enhanced = episode_results.copy()

        # Garantir que temos métricas básicas
        distance = episode_results.get("distance", 0)
        roll = abs(episode_results.get("roll", 0))
        steps = episode_results.get("steps", 0)
        speed = episode_results.get("speed", 0)
        success = episode_results.get("success", False)

        # Calcular sucesso da fase atual
        current_config = self.phases[self.current_phase]
        conditions = current_config.transition_conditions

        episode_success = True

        # Verificar condições básicas
        if "min_avg_distance" in conditions:
            min_episode_distance = conditions["min_avg_distance"] * 0.3
            if distance < min_episode_distance:
                episode_success = False

        if "max_avg_roll" in conditions:
            max_episode_roll = conditions["max_avg_roll"] * 1.5
            if roll > max_episode_roll:
                episode_success = False

        if "min_avg_steps" in conditions:
            min_episode_steps = conditions["min_avg_steps"] * 0.5
            if steps < min_episode_steps:
                episode_success = False

        if "min_avg_speed" in conditions:
            min_episode_speed = conditions["min_avg_speed"] * 0.3
            if speed < min_episode_speed:
                episode_success = False

        # Considerar sucesso geral do episódio como fator adicional
        if success:
            episode_success = True  

        enhanced["phase_success"] = episode_success
        enhanced["phase"] = self.current_phase

        # Adicionar métricas padrão se não existirem
        if "total_reward" not in enhanced:
            enhanced["total_reward"] = episode_results.get("reward", 0)

        return enhanced

    def _update_success_failure_counters(self, episode_results: Dict):
        """Atualiza contadores de sucesso e fracasso"""
        phase_success = episode_results.get("phase_success", False)

        if phase_success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        # Atualizar contador de estagnação
        current_distance = episode_results.get("distance", 0)

        # Calcular progresso relativo
        if len(self.progression_history) >= 2:
            recent_distances = [r.get("distance", 0) for r in self.progression_history[-5:]]
            avg_recent_distance = np.mean(recent_distances) if recent_distances else current_distance

            # Verificar se há progresso significativo
            progress_threshold = 0.02  
            if current_distance > avg_recent_distance + progress_threshold:
                self.stagnation_counter = 0  
            elif abs(current_distance - avg_recent_distance) < progress_threshold:
                self.stagnation_counter += 1  
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)  
        else:
            # Inicialização
            self.stagnation_counter = 0

        self.last_avg_distance = current_distance

    def _collect_dass_samples(self, episode_results: Dict):
        """Coleta amostras para DASS"""
        sample = {
            'episode_results': episode_results.copy(),
            'phase': self.current_phase,
            'timestamp': datetime.now(),
            'performance_metrics': self._calculate_performance_metrics(),
            'policy_features': self._extract_policy_features()
        }
        
        self.dass_samples.append(sample)
        
        # Manter tamanho gerenciável
        if len(self.dass_samples) > 5000:
            self.dass_samples = self.dass_samples[-5000:]

    def _learn_reward_via_irl(self):
        """Aprende recompensa via Inverse Reinforcement Learning melhorado"""
        if len(self.dass_samples) < 200: 
            return

        try:
            current_phase_config = self.phases[self.current_phase]

            # 1. Filtrar demonstrações de alta qualidade
            successful_samples = self._extract_high_quality_demonstrations()

            if len(successful_samples) < 80:  
                return

            # 2. Extrair features relevantes
            feature_matrix, feature_names = self._extract_irl_features(successful_samples)

            if feature_matrix.shape[0] < 50:  
                return

            # 3. Aprender pesos via Maximum Margin IRL simplificado
            learned_weights = self._learn_weights_maximum_margin(feature_matrix, successful_samples)

            # 4. Calcular confiança do modelo (CHAMADA CORRIGIDA)
            confidence = self._calculate_irl_model_confidence(feature_matrix, learned_weights, successful_samples)

            # 5. Aplicar smoothing e validar pesos
            validated_weights = self._validate_and_smooth_weights(learned_weights, confidence)

            # 6. Atualizar modelo com informações completas
            self.learned_reward_model = {
                'weights': validated_weights,
                'feature_names': feature_names,
                'confidence': confidence,
                'sample_size': len(successful_samples),
                'phase': self.current_phase,
                'timestamp': datetime.now(),
                'performance_correlation': self._calculate_performance_correlation(validated_weights, successful_samples)
            }

            # 7. Aplicar gradualmente ao sistema de recompensa se confiança alta
            if confidence > current_phase_config.irl_confidence_threshold:
                self._apply_learned_reward_weights(validated_weights, feature_names, confidence)

        except Exception as e:
            self.logger.warning(f"Erro no IRL: {e}")

    def _extract_high_quality_demonstrations(self):
        """Extrai demonstrações de alta qualidade para IRL"""
        quality_samples = []

        for sample in self.dass_samples:
            episode = sample['episode_results']

            # Critérios de qualidade mais rigorosos
            quality_score = self._calculate_demonstration_quality(sample)

            if quality_score > 0.6:  
                quality_samples.append(sample)

        # Ordenar por qualidade e pegar as melhores
        quality_samples.sort(key=lambda x: self._calculate_demonstration_quality(x), reverse=True)

        return quality_samples[:300]  

    def _calculate_demonstration_quality(self, sample) -> float:
        """Calcula qualidade de uma demonstração para IRL"""
        episode = sample['episode_results']

        quality_factors = []

        # 1. Sucesso do episódio
        success_bonus = 2.0 if episode.get('success', False) else 0.0
        quality_factors.append(success_bonus)

        # 2. Distância percorrida (normalizada)
        distance = episode.get('distance', 0)
        target_distance = self.phases[sample['phase']].transition_conditions.get('min_avg_distance', 1.0)
        distance_score = min(distance / max(target_distance, 0.1), 2.0)  
        quality_factors.append(distance_score)

        # 3. Estabilidade
        roll = abs(episode.get('roll', 0))
        pitch = abs(episode.get('pitch', 0))
        stability_score = 2.0 - min((roll + pitch) / 0.5, 2.0)  
        quality_factors.append(stability_score)

        # 4. Eficiência energética
        energy = episode.get('energy_used', 1.0)
        energy_efficiency = distance / max(energy, 0.1)
        energy_score = min(energy_efficiency / 1.0, 1.5)  
        quality_factors.append(energy_score)

        # 5. Padrão de marcha
        gait_score = episode.get('gait_pattern_score', 0.5) * 1.5
        quality_factors.append(gait_score)

        # 6. Velocidade adequada
        speed = episode.get('speed', 0)
        target_speed = self.phases[sample['phase']].target_speed
        speed_match = 1.0 - min(abs(speed - target_speed) / max(target_speed, 0.1), 1.0)
        quality_factors.append(speed_match)

        return sum(quality_factors) / len(quality_factors)

    def _extract_irl_features(self, samples):
        """Extrai matriz de features para IRL"""
        feature_names = [
            'distance', 'stability', 'speed', 'gait_quality', 
            'energy_efficiency', 'step_consistency', 'balance_control'
        ]

        feature_matrix = []

        for sample in samples:
            episode = sample['episode_results']

            features = [
                episode.get('distance', 0),
                1.0 - min(abs(episode.get('roll', 0)) + abs(episode.get('pitch', 0)), 1.0),  # Estabilidade
                episode.get('speed', 0),
                episode.get('gait_pattern_score', 0.5),
                self._calculate_episode_energy_efficiency(episode),
                self._calculate_step_consistency(sample),
                self._calculate_balance_control(episode)
            ]

            feature_matrix.append(features)

        return np.array(feature_matrix), feature_names

    def _learn_weights_maximum_margin(self, feature_matrix, samples):
        """Aprende pesos usando abordagem Maximum Margin"""
        # Calcular feature expectations das demonstrações
        expert_feature_expectations = np.mean(feature_matrix, axis=0)

        # Adicionar pequena regularização para evitar pesos extremos
        regularization = 0.1

        # Inicializar pesos (prioridade para estabilidade e distância)
        weights = np.array([0.2, 0.3, 0.15, 0.15, 0.1, 0.05, 0.05])

        # Ajustar pesos baseado nas feature expectations
        for i in range(len(weights)):
            feature_importance = expert_feature_expectations[i] / (np.sum(expert_feature_expectations) + 1e-8)
            weights[i] = 0.7 * weights[i] + 0.3 * feature_importance

        # Normalizar e aplicar regularização
        weights = np.abs(weights)  # Garantir pesos positivos
        weights = weights / (np.sum(weights) + 1e-8)
        weights = (1 - regularization) * weights + regularization * (1.0 / len(weights))

        return weights.tolist()

    def _calculate_irl_model_confidence(self, feature_matrix, weights, samples) -> float:
        """Calcula confiança do modelo IRL aprendido"""
        if len(samples) < 50:
            return 0.0

        try:
            # 1. Consistência entre amostras
            feature_std = np.std(feature_matrix, axis=0)
            consistency_score = 1.0 - min(np.mean(feature_std) / 0.3, 1.0)

            # 2. Correlação com sucesso
            success_correlation = self._calculate_performance_correlation(weights, samples)

            # 3. Tamanho da amostra
            sample_size_confidence = min(len(samples) / 200.0, 1.0)

            # 4. Variância dos pesos (evitar overfitting)
            weight_variance = np.var(weights)
            weight_confidence = 1.0 - min(weight_variance * 5.0, 1.0)

            confidence = (
                0.3 * consistency_score +
                0.4 * success_correlation + 
                0.2 * sample_size_confidence +
                0.1 * weight_confidence
            )

            return min(confidence, 1.0)

        except Exception:
            return 0.3

    def _calculate_performance_correlation(self, weights, samples) -> float:
        """Calcula quão bem os pesos correlacionam com performance"""
        if len(samples) < 10:
            return 0.5

        predicted_scores = []
        actual_scores = []

        for sample in samples[:50]:  
            episode = sample['episode_results']
            features = [
                episode.get('distance', 0),
                1.0 - min(abs(episode.get('roll', 0)) + abs(episode.get('pitch', 0)), 1.0),
                episode.get('speed', 0),
                episode.get('gait_pattern_score', 0.5),
                self._calculate_episode_energy_efficiency(episode),
                self._calculate_step_consistency(sample),
                self._calculate_balance_control(episode)
            ]

            predicted_score = np.dot(features, weights)
            predicted_scores.append(predicted_score)

            # Score real baseado em sucesso e qualidade
            actual_score = 1.0 if episode.get('success', False) else 0.5
            actual_score += min(episode.get('distance', 0) / 2.0, 0.5)  
            actual_scores.append(min(actual_score, 1.0))

        # Calcular correlação
        correlation = np.corrcoef(predicted_scores, actual_scores)[0, 1]

        return max(0.0, correlation) if not np.isnan(correlation) else 0.3

    # Métodos auxiliares para o IRL melhorado
    def _calculate_episode_energy_efficiency(self, episode) -> float:
        """Calcula eficiência energética de um episódio"""
        distance = episode.get('distance', 0)
        energy = episode.get('energy_used', 1.0)
        return distance / max(energy, 0.1)

    def _calculate_step_consistency(self, sample) -> float:
        """Calcula consistência de passos"""
        episode = sample['episode_results']
        steps = episode.get('steps', 1)
        distance = episode.get('distance', 0)

        if steps > 0 and distance > 0:
            step_length_consistency = 1.0 - min(abs((distance / steps) - 0.3) / 0.3, 1.0)
            return step_length_consistency
        return 0.5

    def _calculate_balance_control(self, episode) -> float:
        """Calcula controle de equilíbrio"""
        roll = abs(episode.get('roll', 0))
        pitch = abs(episode.get('pitch', 0))
        return 1.0 - min((roll + pitch) / 1.0, 1.0)

    def _validate_and_smooth_weights(self, weights, confidence):
        """Valida e suaviza pesos aprendidos"""
        weights = np.array(weights)

        # Garantir que nenhum peso seja extremo
        weights = np.clip(weights, 0.05, 0.5)

        # Normalizar
        weights = weights / np.sum(weights)

        # Aplicar suavização baseada na confiança
        if confidence < 0.7:
            # Se confiança baixa, manter mais próximo dos pesos padrão
            default_weights = np.array([0.25, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05])
            alpha = 1.0 - confidence
            weights = alpha * default_weights + (1 - alpha) * weights

        return weights.tolist()

    def _apply_learned_reward_weights(self, weights, feature_names, confidence):
        """Aplica pesos aprendidos ao sistema de recompensa gradualmente"""
        if not hasattr(self.reward_system, 'components'):
            return

        # Mapear features para componentes de recompensa
        feature_to_component = {
            'distance': ['progress', 'distance_bonus'],
            'stability': ['stability_roll', 'stability_pitch'],
            'speed': ['pitch_forward_bonus'],
            'gait_quality': ['gait_pattern_cross', 'alternating_foot_contact'],
            'energy_efficiency': ['effort_square_penalty'],
            'step_consistency': ['alternating_foot_contact'],
            'balance_control': ['stability_roll', 'stability_pitch', 'center_bonus']
        }

        # Ajustar pesos gradualmente baseado na confiança
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_name in feature_to_component:
                components = feature_to_component[feature_name]
                feature_weight = weights[feature_idx] * confidence

                for component_name in components:
                    if component_name in self.reward_system.components:
                        current_weight = self.reward_system.components[component_name].weight
                        # Ajuste gradual: 10% do caminho para o novo peso
                        new_weight = 0.9 * current_weight + 0.1 * feature_weight
                        self.reward_system.components[component_name].weight = new_weight

    def _update_adaptive_components(self, episode_results: Dict):
        """Atualiza componentes adaptativos com foco em progresso - VERSÃO CORRIGIDA"""
        # Coletar métricas de performance para cada componente
        component_metrics = {}

        for component in self.adaptive_components.keys():
            metrics = self._calculate_component_metrics(component, episode_results)
            self.adaptive_components[component]['recent_rewards'].append(metrics['reward'])
            self.adaptive_components[component]['performance_metrics'].append(metrics)

            # Manter histórico limitado
            if len(self.adaptive_components[component]['recent_rewards']) > 100:
                self.adaptive_components[component]['recent_rewards'].pop(0)
            if len(self.adaptive_components[component]['performance_metrics']) > 50:
                self.adaptive_components[component]['performance_metrics'].pop(0)

            component_metrics[component] = {
                'recent_rewards': self.adaptive_components[component]['recent_rewards'][-20:],
                'performance': metrics
            }

        current_distance = episode_results.get('distance', 0)
        current_speed = episode_results.get('speed', 0)

        # Se está tendo pouco progresso, aumentar peso do progresso
        if current_distance < 0.5:  
            progress_boost = 1.5
            stability_penalty = 0.7
        else:
            progress_boost = 1.0
            stability_penalty = 1.0

        # Calcular pesos dinâmicos com ajuste de progresso
        dynamic_weights = self._calculate_dynamic_weights(component_metrics)

        # Aplicar ajuste de progresso
        for component, weight in dynamic_weights.items():
            if component in ["progress", "distance_bonus", "pitch_forward_bonus"]:
                dynamic_weights[component] = weight * progress_boost
            elif component in ["stability_roll", "stability_pitch"]:
                dynamic_weights[component] = weight * stability_penalty

        # Atualizar pesos no sistema de recompensa
        for component, weight in dynamic_weights.items():
            if component in self.reward_system.components:
                adaptive_weight = weight * self.adaptive_components[component]['learning_rate']
                self.reward_system.components[component].weight = adaptive_weight

        # Coletar dados para análise do crítico
        total_reward = episode_results.get('total_reward', 0)
        self.recent_actual_rewards.append(total_reward)
        if len(self.recent_actual_rewards) > 50:
            self.recent_actual_rewards.pop(0)

    def _calculate_component_metrics(self, component: str, episode_results: Dict) -> Dict:
        """Calcula métricas específicas para cada componente"""
        if component == "gait_quality":
            gait_score = episode_results.get('gait_pattern_score', 0.5)
            alternation = 1.0 if episode_results.get('left_contact', False) != episode_results.get('right_contact', False) else 0.3
            return {'reward': (gait_score + alternation) / 2, 'score': gait_score}
        
        elif component == "balance_stability":
            roll = abs(episode_results.get('roll', 0))
            pitch = abs(episode_results.get('pitch', 0))
            ideal_pitch = -0.1  
            pitch_error = abs(pitch - ideal_pitch)
            pitch_stability = 1.0 - min(pitch_error / 0.5, 1.0)
            roll_stability = 1.0 - min(roll / 1.0, 1.0)
            stability = (roll_stability * 0.7 + pitch_stability * 0.3)
            return {'reward': stability, 'score': stability}
        
        elif component == "energy_efficiency":
            energy = episode_results.get('energy_used', 1.0)
            distance = episode_results.get('distance', 0.1)
            efficiency = distance / (energy + 0.1)
            return {'reward': min(efficiency / 2.0, 1.0), 'score': efficiency}
        
        elif component == "speed_tracking":
            current_speed = episode_results.get('speed', 0)
            target_speed = self.get_current_speed_target()
            tracking = 1.0 - min(abs(current_speed - target_speed) / (target_speed + 0.1), 1.0)
            return {'reward': tracking, 'score': tracking}
        
        return {'reward': 0.5, 'score': 0.5}

    def _update_hdpg_weights(self):
        """Atualiza pesos usando HDPG"""
        if self.current_phase < 3:
            return
            
        # Aplicar ajuste dinâmico de pesos baseado no multi-head critic
        current_config = self.phases[self.current_phase]
        
        # Calcular ajustes baseados na performance recente
        performance_scores = {}
        for component in current_config.enabled_components:
            if component in self.adaptive_components:
                recent_rewards = self.adaptive_components[component]['recent_rewards']
                if recent_rewards:
                    performance_scores[component] = np.mean(recent_rewards[-10:])
                else:
                    performance_scores[component] = 0.5
        
        # Normalizar e ajustar pesos
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            for component, score in performance_scores.items():
                if component in self.reward_system.components:
                    normalized_score = score / total_performance
                    # Suavizar transição de pesos
                    current_weight = self.reward_system.components[component].weight
                    target_weight = current_config.component_weights.get(component, current_weight) * normalized_score
                    new_weight = 0.95 * current_weight + 0.05 * target_weight
                    self.reward_system.components[component].weight = new_weight
    
    def _check_enhanced_phase_advancement(self) -> bool:
        """Verificação HDPG avançada para transição de fase com validação multi-critério"""
        if not self._meets_minimum_requirements():
            return False

        current_phase_config = self.phases[self.current_phase]

        # 1. Verificação básica de condições
        basic_conditions_met = self._check_basic_advancement_conditions()
        if not basic_conditions_met:
            return False

        # 2. Validação HDPG - Convergência do crítico multi-head
        hdpg_convergence_met = self._validate_hdpg_convergence()

        # 3. Estabilidade do gradiente de política
        gradient_stable = self._check_gradient_stability()

        # 4. Balanceamento de componentes adaptativos
        components_balanced = self._check_component_balance() > 0.7

        # 5. Convergência de prioridades HDPG
        priorities_converged = self._check_priority_convergence()

        # 6. Validação IRL 
        irl_valid = True
        if self.current_phase >= 2:
            irl_confidence = self._calculate_irl_confidence()
            irl_valid = irl_confidence > current_phase_config.irl_confidence_threshold

        # Critério HDPG: Todos os componentes devem estar estáveis
        hdpg_conditions = (
            hdpg_convergence_met and 
            gradient_stable and 
            components_balanced and 
            priorities_converged and
            irl_valid
        )

        # Para fases iniciais, ser mais permissivo; 
        if self.current_phase <= 2:
            return basic_conditions_met and (hdpg_convergence_met or gradient_stable)
        else:
            return basic_conditions_met and hdpg_conditions

    def _calculate_irl_confidence(self) -> float:
        """Calcula confiança no modelo IRL aprendido"""
        if self.learned_reward_model is None:
            return 0.0
        
        confidence = self.learned_reward_model.get('confidence', 0.0)
        
        # Ajustar confiança baseado na consistência
        recent_successes = sum(1 for r in self.progression_history[-20:] if r.get('phase_success', False))
        success_consistency = recent_successes / 20.0
        
        return 0.7 * confidence + 0.3 * success_consistency

    def _validate_hdpg_convergence(self) -> bool:
        """Valida convergência do HDPG usando múltiplas métricas"""
        if self.current_phase < 2:  
            return True

        convergence_metrics = {
            "critic_stability": self._check_critic_convergence(),
            "actor_improvement": self._check_actor_improvement(),
            "value_consistency": self._check_value_consistency(),
            "component_balance": self._check_component_balance(),
            "gradient_stability": self._check_gradient_stability(),
            "priority_convergence": self._check_priority_convergence()
        }

        # Pesos diferentes para cada métrica
        weights = {
            "critic_stability": 0.25,
            "actor_improvement": 0.20,
            "value_consistency": 0.15,
            "component_balance": 0.15,
            "gradient_stability": 0.15,
            "priority_convergence": 0.10
        }

        # Calcular score total ponderado
        total_score = 0.0
        for metric, score in convergence_metrics.items():
            total_score += score * weights[metric]

        # Threshold adaptativo por fase
        phase_threshold = 0.6 + (self.current_phase * 0.05) 

        return total_score >= phase_threshold

    def _check_critic_convergence(self) -> float:
        """Verifica convergência do crítico multi-head usando métricas HDPG"""
        if self.multi_head_critic is None:
            # Simular convergência se crítico não está inicializado
            return 0.7 if self.current_phase < 3 else 0.5

        try:
            convergence_metrics = []

            # 1. Consistência entre heads do crítico
            if hasattr(self, 'recent_critic_predictions') and self.recent_critic_predictions:
                head_consistencies = []
                for predictions in self.recent_critic_predictions[-10:]:
                    if len(predictions) >= 2:  # Pelo menos 2 heads
                        head_std = np.std(predictions)
                        head_mean = np.mean(predictions) if np.mean(predictions) != 0 else 1.0
                        consistency = 1.0 - min(head_std / abs(head_mean), 1.0)
                        head_consistencies.append(consistency)

                if head_consistencies:
                    head_consistency = np.mean(head_consistencies)
                    convergence_metrics.append(head_consistency * 0.4)

            # 2. Estabilidade temporal das previsões do crítico
            if hasattr(self, 'recent_critic_values') and len(self.recent_critic_values) >= 15:
                critic_values = self.recent_critic_values[-15:]
                value_changes = [abs(critic_values[i] - critic_values[i-1]) 
                               for i in range(1, len(critic_values))]
                if value_changes:
                    avg_change = np.mean(value_changes)
                    value_stability = 1.0 - min(avg_change / 0.2, 1.0)
                    convergence_metrics.append(value_stability * 0.3)

            # 3. Correlação entre previsões do crítico e recompensas reais
            if (hasattr(self, 'recent_critic_predictions') and 
                hasattr(self, 'recent_actual_rewards') and
                len(self.recent_actual_rewards) >= 10):

                correlations = []
                min_len = min(len(self.recent_critic_predictions), 
                             len(self.recent_actual_rewards))

                for i in range(min(10, min_len)):
                    critic_preds = self.recent_critic_predictions[-(i+1)]
                    actual_reward = self.recent_actual_rewards[-(i+1)]

                    if len(critic_preds) > 0:
                        avg_prediction = np.mean(critic_preds)
                        # Correlação simplificada
                        error = abs(avg_prediction - actual_reward)
                        accuracy = 1.0 - min(error / (abs(actual_reward) + 0.1), 1.0)
                        correlations.append(accuracy)

                if correlations:
                    avg_correlation = np.mean(correlations)
                    convergence_metrics.append(avg_correlation * 0.3)

            # 4. Se não há métricas específicas, usar estabilidade geral
            if not convergence_metrics:
                recent_rewards = []
                for component in self.adaptive_components.values():
                    if component['recent_rewards']:
                        recent_rewards.extend(component['recent_rewards'][-5:])

                if recent_rewards:
                    reward_std = np.std(recent_rewards)
                    reward_stability = 1.0 - min(reward_std / 0.3, 1.0)
                    return reward_stability

            return np.mean(convergence_metrics) if convergence_metrics else 0.6

        except Exception as e:
            self.logger.debug(f"Erro na verificação do crítico: {e}")
            return 0.5

    def _check_basic_advancement_conditions(self) -> bool:
        """Verifica condições básicas de avanço de fase"""
        if self.current_phase >= len(self.phases) - 1:
            return False

        current_phase_config = self.phases[self.current_phase]
        conditions = current_phase_config.transition_conditions

        # Verificar condições obrigatórias
        mandatory_conditions_met = True

        # 1. Taxa de sucesso mínima
        if "min_success_rate" in conditions:
            current_success = self._calculate_success_rate()
            if current_success < conditions["min_success_rate"]:
                mandatory_conditions_met = False
                self.logger.debug(f"Condição de sucesso não atendida: {current_success:.3f} < {conditions['min_success_rate']}")

        # 2. Distância média mínima
        if "min_avg_distance" in conditions:
            current_avg_distance = self._calculate_average_distance()
            if current_avg_distance < conditions["min_avg_distance"]:
                mandatory_conditions_met = False
                self.logger.debug(f"Condição de distância não atendida: {current_avg_distance:.3f} < {conditions['min_avg_distance']}")

        # 3. Estabilidade máxima (roll)
        if "max_avg_roll" in conditions:
            current_avg_roll = self._calculate_average_roll()
            if current_avg_roll > conditions["max_avg_roll"]:
                mandatory_conditions_met = False
                self.logger.debug(f"Condição de estabilidade não atendida: {current_avg_roll:.3f} > {conditions['max_avg_roll']}")

        # 4. Velocidade mínima (se aplicável)
        if "min_avg_speed" in conditions:
            current_avg_speed = self._calculate_average_speed()
            if current_avg_speed < conditions["min_avg_speed"]:
                mandatory_conditions_met = False
                self.logger.debug(f"Condição de velocidade não atendida: {current_avg_speed:.3f} < {conditions['min_avg_speed']}")

        # 5. Passos mínimos (fases iniciais)
        if "min_avg_steps" in conditions:
            recent_steps = [r.get("steps", 0) for r in self.progression_history[-5:]]
            avg_steps = np.mean(recent_steps) if recent_steps else 0
            if avg_steps < conditions["min_avg_steps"]:
                mandatory_conditions_met = False
                self.logger.debug(f"Condição de passos não atendida: {avg_steps:.1f} < {conditions['min_avg_steps']}")

        # 6. Condições específicas por fase
        phase_specific_conditions = True

        if self.current_phase == 2:
            # Fase 2: Marcha Lenta Estável
            if "min_alternating_score" in conditions:
                alternation_score = self._calculate_gait_coordination(self.progression_history[-8:])
                if alternation_score < conditions["min_alternating_score"]:
                    phase_specific_conditions = False
                    self.logger.debug(f"Coordenação insuficiente: {alternation_score:.3f} < {conditions['min_alternating_score']}")

        elif self.current_phase == 3:
            # Fase 3: Marcha Confiante
            if "min_gait_coordination" in conditions:
                coordination = self._calculate_gait_coordination(self.progression_history[-10:])
                if coordination < conditions["min_gait_coordination"]:
                    phase_specific_conditions = False
                    self.logger.debug(f"Coordenação de marcha insuficiente: {coordination:.3f} < {conditions['min_gait_coordination']}")

        elif self.current_phase >= 4:
            # Fases 4+: Marcha Rápida e além
            if "min_propulsion_efficiency" in conditions:
                propulsion = self._calculate_propulsion_efficiency()
                if propulsion < conditions["min_propulsion_efficiency"]:
                    phase_specific_conditions = False
                    self.logger.debug(f"Eficiência propulsiva insuficiente: {propulsion:.3f} < {conditions['min_propulsion_efficiency']}")

            if "min_gait_coordination" in conditions:
                coordination = self._calculate_gait_coordination(self.progression_history[-10:])
                if coordination < conditions["min_gait_coordination"]:
                    phase_specific_conditions = False
                    self.logger.debug(f"Coordenação insuficiente: {coordination:.3f} < {conditions['min_gait_coordination']}")

        # 7. Consistência (para fases avançadas)
        if self.current_phase >= 4 and "consistency_count" in conditions:
            recent_successes = sum(1 for r in self.progression_history[-conditions["consistency_count"]:] 
                                  if r.get("phase_success", False))
            consistency_met = recent_successes >= conditions["consistency_count"] * 0.7

            if not consistency_met:
                phase_specific_conditions = False
                self.logger.debug(f"Consistência insuficiente: {recent_successes}/{conditions['consistency_count']}")

        return mandatory_conditions_met and phase_specific_conditions
    
    def _check_actor_improvement(self) -> float:
        """Verifica se o actor está melhorando consistentemente"""
        if len(self.progression_history) < 10:
            return 0.5

        # Analisar tendência de melhoria nas últimas 10 épocas
        recent_distances = [r.get('distance', 0) for r in self.progression_history[-10:]]

        if len(recent_distances) >= 5:
            # Calcular tendência linear
            x = np.arange(len(recent_distances))
            slope, _ = np.polyfit(x, recent_distances, 1)

            # Score baseado na inclinação positiva
            improvement = 0.5 + min(max(slope / 0.05, -0.5), 0.5)
            return max(0.0, improvement)

        return 0.5

    def _check_value_consistency(self) -> float:
        """Verifica consistência dos valores Q estimados"""
        if len(self.progression_history) < 8:
            return 0.5

        # Analisar consistência das recompensas
        recent_rewards = [r.get('total_reward', 0) for r in self.progression_history[-8:]]
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)

        if reward_mean > 0:
            cv = reward_std / reward_mean
            consistency = 1.0 - min(cv, 2.0) / 2.0
            return consistency

        return 0.5

    def _check_component_balance(self) -> float:
        """Verifica balanceamento entre componentes"""
        component_performances = []
        for component in self.adaptive_components.values():
            if component['recent_rewards']:
                perf = np.mean(component['recent_rewards'][-10:])
                component_performances.append(perf)
        
        if len(component_performances) < 2:
            return 0.5
            
        # Balanceamento ideal: todos os componentes com performance similar
        std_performance = np.std(component_performances)
        balance_score = 1.0 - min(std_performance, 1.0)
        return balance_score

    def _check_gradient_stability(self) -> float:
        """Verifica estabilidade do gradiente com métricas HDPG"""
        if len(self.progression_history) < 15:
            return 0.5

        try:
            stability_metrics = []

            # 1. Estabilidade da distância 
            distances = [r.get('distance', 0) for r in self.progression_history[-15:]]
            if len(distances) >= 10:
                distance_changes = [abs(distances[i] - distances[i-1]) for i in range(1, len(distances))]
                avg_distance_change = np.mean(distance_changes) if distance_changes else 0.0
                distance_stability = 1.0 - min(avg_distance_change / 0.2, 1.0)
                stability_metrics.append(distance_stability)

            # 2. Estabilidade da recompensa
            rewards = [r.get('total_reward', 0) for r in self.progression_history[-15:]]
            if len(rewards) >= 10:
                reward_std = np.std(rewards)
                reward_mean = np.mean(rewards) if np.mean(rewards) != 0 else 1.0
                reward_stability = 1.0 - min(reward_std / abs(reward_mean), 1.0)
                stability_metrics.append(reward_stability)

            # 3. Estabilidade dos componentes adaptativos
            component_stabilities = []
            for component in self.adaptive_components.values():
                if component['recent_rewards'] and len(component['recent_rewards']) >= 10:
                    rewards = component['recent_rewards'][-10:]
                    reward_std = np.std(rewards)
                    reward_mean = np.mean(rewards) if np.mean(rewards) != 0 else 1.0
                    comp_stability = 1.0 - min(reward_std / abs(reward_mean), 1.0)
                    component_stabilities.append(comp_stability)

            if component_stabilities:
                component_stability = np.mean(component_stabilities)
                stability_metrics.append(component_stability)

            # 4. Consistência do sucesso
            successes = [1 if r.get('phase_success', False) else 0 for r in self.progression_history[-15:]]
            if len(successes) >= 10:
                success_consistency = np.mean(successes)
                stability_metrics.append(success_consistency)

            return np.mean(stability_metrics) if stability_metrics else 0.5

        except Exception:
            return 0.5

    def _check_priority_convergence(self) -> bool:
        """Verifica se as prioridades HDPG convergiram"""
        if not hasattr(self, 'previous_priorities') or not self.previous_priorities:
            self.previous_priorities = {}
            return False

        # Calcular prioridades atuais baseadas na performance recente
        current_priorities = {}
        for component, data in self.adaptive_components.items():
            if data['recent_rewards'] and len(data['recent_rewards']) >= 8:
                # Prioridade = média móvel + variância (equação HDPG)
                recent_rewards = data['recent_rewards'][-8:]
                μ = np.mean(recent_rewards)
                σ = np.var(recent_rewards)
                current_priorities[component] = (μ + np.exp(σ**2)) * data['learning_rate']

        if not current_priorities or not self.previous_priorities:
            self.previous_priorities = current_priorities.copy()
            return False

        # Calcular mudança nas prioridades
        total_change = 0.0
        count = 0

        for component, current_prio in current_priorities.items():
            if component in self.previous_priorities:
                previous_prio = self.previous_priorities[component]
                if previous_prio > 0:  
                    change = abs(current_prio - previous_prio) / previous_prio
                    total_change += change
                    count += 1

        # Atualizar histórico
        self.previous_priorities = current_priorities.copy()

        if count == 0:
            return False

        avg_change = total_change / count

        # Considerar convergido se mudança média < 10%
        converged = avg_change < 0.10

        self.logger.debug(f"Priority convergence: change={avg_change:.3f}, converged={converged}")

        return converged

    def _start_gradual_transition(self) -> PhaseTransitionResult:
        """Inicia transição gradual para próxima fase"""
        if self.transition_active:
            return PhaseTransitionResult.FAILURE
    
        if self.current_phase >= len(self.phases) - 1:
            return PhaseTransitionResult.SUCCESS

        self.old_weights = {}
        self.old_enabled_components = {}
        
        current_config = self.phases[self.current_phase]
        for component_name in current_config.enabled_components:
            if component_name in self.reward_system.components:
                self.old_weights[component_name] = self.reward_system.components[component_name].weight
                self.old_enabled_components[component_name] = self.reward_system.components[component_name].enabled

        self.transition_active = True
        self.transition_episodes = 0
        self.transition_total_episodes = 10  

        self._smart_buffer_transition()
        
        return PhaseTransitionResult.SUCCESS
    
    def _update_transition_progress(self) -> PhaseTransitionResult:
        """Atualiza progresso da transição gradual"""
        self.transition_episodes += 1
        transition_progress = self.transition_episodes / self.transition_total_episodes
        self._apply_gradual_phase_config(transition_progress)
        
        if self.transition_episodes >= self.transition_total_episodes:
            return self._complete_phase_transition()
        
        return PhaseTransitionResult.SUCCESS
    
    def _apply_gradual_phase_config(self, progress: float):
        """Aplica configuração gradualmente durante transição"""
        current_phase_config = self.phases[self.current_phase]
        next_phase_config = self.phases[self.current_phase + 1]
        
        for component_name, component in self.reward_system.components.items():
            old_weight = self.old_weights.get(component_name, 0.0)
            new_weight = next_phase_config.component_weights.get(component_name, 0.0)
            interpolated_weight = old_weight * (1 - progress) + new_weight * progress
            component.weight = interpolated_weight
            old_enabled = self.old_enabled_components.get(component_name, False)
            new_enabled = component_name in next_phase_config.enabled_components
            component.enabled = old_enabled or new_enabled

    def _complete_phase_transition(self) -> PhaseTransitionResult:
        """Completa a transição para próxima fase"""
        old_phase = self.current_phase
        old_phase_name = self.phases[old_phase].name
        episodes_in_old_phase = self.episodes_in_phase

        self.current_phase += 1
        self.episodes_in_phase = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.stagnation_counter = 0
        self.regression_count = 0
        
        self.transition_active = False
        self.transition_episodes = 0
        self.old_weights = None
        self.old_enabled_components = None
        
        self._apply_phase_config()
        
        keep_episodes = min(8, len(self.progression_history))
        self.progression_history = self.progression_history[-keep_episodes:]
        
        new_phase_config = self.phases[self.current_phase]
        self._generate_phase_transition_report(old_phase, old_phase_name, new_phase_config, episodes_in_old_phase)

        return PhaseTransitionResult.SUCCESS

    def _generate_phase_transition_report(self, old_phase: int, old_phase_name: str, new_phase_config, episodes_in_old_phase: int):
        """Gera relatório detalhado da transição de fase"""
        try:
            current_metrics = self._calculate_performance_metrics()
            old_phase_config = self.phases[old_phase]
            
            print(f"\n🎯 RELATÓRIO DE TRANSIÇÃO DE FASE - Episódio {len(self.performance_history)}")
            print(f"   {old_phase_name.upper()} → {new_phase_config.name.upper()}")
            print(f"   Fase {old_phase} → Fase {self.current_phase}")
            print("")
            
            # MÉTRICAS DE PERFORMANCE NA FASE ANTERIOR
            print("   MÉTRICAS DE PERFORMANCE (fase anterior):")
            print(f"     ✅ Taxa de sucesso: {current_metrics['success_rate']:.1%}")
            print(f"     📏 Distância média: {current_metrics['avg_distance']:.2f}m")
            print(f"     🚀 Velocidade média: {current_metrics['avg_speed']:.2f} m/s")
            print(f"     ⚖️  Estabilidade (roll): {current_metrics['avg_roll']:.3f}")
            print(f"     🔄 Consistência: {current_metrics['consistency']:.1%}")
            print(f"     ⚡ Eficiência energética: {current_metrics['energy_efficiency']:.1%}")
            print("")
            
            # REQUISITOS ATENDIDOS - CORRIGIDO
            print("   REQUISITOS ATENDIDOS:")
            
            success_rate_met = current_metrics['success_rate'] >= old_phase_config.transition_conditions['min_success_rate']
            success_icon = "✅" if success_rate_met else "❌"
            print(f"     {success_icon} Taxa de sucesso: {current_metrics['success_rate']:.3f} >= {old_phase_config.transition_conditions['min_success_rate']}")
            
            distance_met = current_metrics['avg_distance'] >= old_phase_config.transition_conditions['min_avg_distance']
            distance_icon = "✅" if distance_met else "❌"
            print(f"     {distance_icon} Distância média: {current_metrics['avg_distance']:.3f}m >= {old_phase_config.transition_conditions['min_avg_distance']}m")
            
            roll_met = current_metrics['avg_roll'] <= old_phase_config.transition_conditions['max_avg_roll']
            roll_icon = "✅" if roll_met else "❌"
            print(f"     {roll_icon} Estabilidade: {current_metrics['avg_roll']:.3f} <= {old_phase_config.transition_conditions['max_avg_roll']}")
            
            # CORREÇÃO: Usar episodes_in_old_phase em vez de self.episodes_in_phase
            min_episodes = old_phase_config.phase_duration
            episodes_met = episodes_in_old_phase >= min_episodes
            episodes_icon = "✅" if episodes_met else "❌"
            print(f"     {episodes_icon} Episódios mínimos: {episodes_in_old_phase} >= {min_episodes}")
            
            # Verificar condições específicas por fase
            if old_phase >= 2:  # Fases 2+
                if "min_alternating_score" in old_phase_config.transition_conditions:
                    alternation_score = self._calculate_gait_coordination(self.progression_history[-8:])
                    alternation_met = alternation_score >= old_phase_config.transition_conditions["min_alternating_score"]
                    alternation_icon = "✅" if alternation_met else "❌"
                    print(f"     {alternation_icon} Coordenação: {alternation_score:.3f} >= {old_phase_config.transition_conditions['min_alternating_score']}")
            
            print("")
            
            # HABILIDADES DESENVOLVIDAS
            skills = self._assess_phase_skills()
            print("   HABILIDADES DESENVOLVIDAS:")
            for skill, req in old_phase_config.skill_requirements.items():
                current = skills.get(skill, 0)
                improvement = current - self._get_default_skills().get(skill, 0)
                status = "✅" if current >= req else "🔄"
                improvement_icon = "📈" if improvement > 0.1 else "➡️" if improvement > 0 else "📉"
                print(f"     {status} {skill}: {current:.3f} / {req:.3f} {improvement_icon} ({improvement:+.3f})")
            print("")
            
            # NOVOS OBJETIVOS DA PRÓXIMA FASE
            print("   NOVOS OBJETIVOS DA FASE:")
            print(f"     🎯 Velocidade alvo: {new_phase_config.target_speed} m/s")
            print(f"     📏 Distância mínima: {new_phase_config.transition_conditions.get('min_avg_distance', 'N/A')}m")
            print(f"     ✅ Taxa de sucesso: {new_phase_config.transition_conditions.get('min_success_rate', 'N/A')}")
            print(f"     ⚖️  Estabilidade máxima: {new_phase_config.transition_conditions.get('max_avg_roll', 'N/A')}")
            
            # Adicionar requisitos específicos da nova fase
            if self.current_phase >= 2:
                if "min_alternating_score" in new_phase_config.transition_conditions:
                    print(f"     🔄 Coordenação mínima: {new_phase_config.transition_conditions['min_alternating_score']}")
                if "min_gait_coordination" in new_phase_config.transition_conditions:
                    print(f"     🦶 Coordenação de marcha: {new_phase_config.transition_conditions['min_gait_coordination']}")
            
            # COMPONENTES ATIVOS
            enabled_count = len(new_phase_config.enabled_components)
            print(f"     🔧 Componentes ativos: {enabled_count}")
            
            # NOVOS COMPONENTES ADICIONADOS
            old_components = set(self.phases[old_phase].enabled_components)
            new_components = set(new_phase_config.enabled_components)
            added_components = new_components - old_components
            if added_components:
                print(f"     ➕ Novos componentes: {', '.join(added_components)}")
            print("")
            
            # ESTATÍSTICAS DO APRENDIZADO
            print("   ESTATÍSTICAS DO APRENDIZADO:")
            print(f"     📊 Total de episódios: {len(self.performance_history)}")
            print(f"     🎯 Episódios na fase anterior: {episodes_in_old_phase}")
            print(f"     🎯 Sucessos consecutivos: {self.consecutive_successes}")
            print(f"     💀 Falhas consecutivas: {self.consecutive_failures}")
            print(f"     🌀 Contador de estagnação: {self.stagnation_counter}")
            print(f"     📦 Amostras DASS: {len(self.dass_samples)}")
            
            if self.learned_reward_model:
                print(f"     🧠 Confiança IRL: {self.learned_reward_model.get('confidence', 0):.1%}")
            
            # PROGRESSO GERAL
            print(f"     📈 Progresso geral: {self.current_phase}/{len(self.phases)-1} fases")
            print("")
            print("   " + "="*50)
            
        except Exception as e:
            self.logger.warning(f"Erro ao gerar relatório de transição: {e}")
        
    def _smart_buffer_transition(self):
        """Transição inteligente do buffer preservando melhores experiências"""
        try:
            if hasattr(self.reward_system, 'agent') and self.reward_system.agent:
                if hasattr(self.reward_system.agent.model, 'replay_buffer'):
                    buffer = self.reward_system.agent.model.replay_buffer
                    buffer_size_before = len(buffer)
                    
                    if self.current_phase <= 1:
                        buffer.clear()
                    else:
                        preserved_count = self._preserve_high_value_experiences(buffer, preserve_ratio=0.3)
                    
                    self._refill_replay_buffer()
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na transição do buffer: {e}")
            self._clear_agent_replay_buffer()

    def _preserve_high_value_experiences(self, buffer, preserve_ratio=0.3) -> int:
        """Preserva as melhores experiências do buffer baseado em qualidade"""
        try:
            if not hasattr(buffer, 'buffer') or len(buffer) == 0:
                return 0
            
            experiences = []
            for i in range(len(buffer)):
                experience = buffer.buffer[i]
                quality_score = self._calculate_experience_quality(experience)
                experiences.append((quality_score, experience))
            
            experiences.sort(key=lambda x: x[0], reverse=True)
            
            preserve_count = int(len(experiences) * preserve_ratio)
            preserved_experiences = experiences[:preserve_count]
            
            buffer.clear()
            for quality_score, experience in preserved_experiences:
                buffer.add(*experience)
            
            return preserve_count
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao preservar experiências: {e}")
            return 0
        
    def _calculate_experience_quality(self, experience) -> float:
        """Calcula qualidade de uma experiência baseado em estabilidade e progresso"""
        try:
            obs, next_obs, action, reward, done, info = experience
            
            quality_score = 0.0
            
            quality_score += min(abs(reward) * 0.1, 1.0)
            
            if not done:
                quality_score += 0.5
            
            if hasattr(action, '__len__'):
                action_smoothness = 1.0 - min(np.std(action) * 2.0, 1.0)
                quality_score += action_smoothness * 0.3
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            return 0.5  
        
    def _cancel_current_transition(self):
        """Cancela transição atual e restaura estado anterior"""
        self.transition_active = False
        self.transition_episodes = 0
        self.old_weights = None
        self.old_enabled_components = None
    
    def _regress_to_previous_phase(self) -> PhaseTransitionResult:
        """Regride para a fase anterior com transição gradual"""
        if self.transition_active:
            self._cancel_current_transition()
        
        if self.current_phase == 0:
            return PhaseTransitionResult.FAILURE

        self.old_weights = {}
        self.old_enabled_components = {}
        
        current_config = self.phases[self.current_phase]
        for component_name in current_config.enabled_components:
            if component_name in self.reward_system.components:
                self.old_weights[component_name] = self.reward_system.components[component_name].weight
                self.old_enabled_components[component_name] = self.reward_system.components[component_name].enabled

        self.transition_active = True
        self.transition_episodes = 0
        self.transition_total_episodes = 8  
        self._smart_buffer_transition()

        old_phase_name = self.phases[self.current_phase].name
        new_phase_name = self.phases[self.current_phase - 1].name
        self.logger.warning(f"🔄 REGRESSÃO INICIADA: {old_phase_name} → {new_phase_name}")

        return PhaseTransitionResult.REGRESSION

    def _clear_agent_replay_buffer(self):
        """Limpar buffer de replay do agente quando as recompensas mudam"""
        try:
            if hasattr(self.reward_system, 'agent') and self.reward_system.agent:
                if hasattr(self.reward_system.agent.model, 'replay_buffer'):
                    buffer_size_before = len(self.reward_system.agent.model.replay_buffer)
                    self.reward_system.agent.model.replay_buffer.clear()
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível limpar buffer: {e}")
        
    def _refill_replay_buffer(self):
        """Preencher rapidamente o buffer após limpeza"""
        try:
            if hasattr(self.reward_system, 'agent') and self.reward_system.agent:
                agent = self.reward_system.agent
                target_size = agent.prefill_steps // 2  

                if hasattr(agent.model, 'replay_buffer'):
                    current_size = len(agent.model.replay_buffer)

                    if current_size < target_size:
                        obs, _ = self.reward_system.agent.env.reset()
                        steps_collected = 0

                        while steps_collected < target_size and not self.reward_system.agent.env.exit_value.value:
                            action = self.reward_system.agent.env.robot.get_example_action(
                                self.reward_system.agent.env.episode_steps * self.reward_system.agent.env.time_step_s
                            )

                            next_obs, reward, terminated, truncated, info = self.reward_system.agent.env.step(action)
                            done = terminated or truncated

                            agent.model.replay_buffer.add(
                                obs.flatten(), 
                                next_obs.flatten(), 
                                action, 
                                reward, 
                                done, 
                                [info]
                            )

                            obs = next_obs
                            steps_collected += 1

                            if done:
                                obs, _ = self.reward_system.agent.env.reset()

        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao preencher buffer: {e}")
        
    def _meets_minimum_requirements(self) -> bool:
        """Verifica requisitos mínimos para transição"""
        current_phase_config = self.phases[self.current_phase]

        min_duration = current_phase_config.phase_duration
        if self.current_phase == 0:
            min_duration = max(5, min_duration)  

        has_minimum_duration = self.episodes_in_phase >= min_duration
        has_sufficient_history = len(self.progression_history) >= 3  

        if not has_minimum_duration:
            self.logger.debug(f"Aguardando duração mínima: {self.episodes_in_phase}/{min_duration}")
            return False

        if not has_sufficient_history:
            self.logger.debug(f"Aguardando histórico suficiente: {len(self.progression_history)}/3")
            return False

        return True

    def _calculate_energy_efficiency(self, recent_results: List[Dict]) -> float:
        """Calcula eficiência energética de forma mais realista"""
        if not recent_results:
            return 0.5

        efficiencies = []
        for result in recent_results:
            distance = result.get("distance", 0)
            energy = result.get("energy_used", 1.0)
            if energy > 0.1:  # Evitar divisão por zero
                efficiency = distance / energy
                efficiencies.append(min(efficiency / 2.0, 1.0))
        
        return np.mean(efficiencies) if efficiencies else 0.5

    def _calculate_balance_recovery_score(self, recent_results: List[Dict]) -> float:
        """Calcula capacidade de recuperar equilíbrio"""
        if len(recent_results) < 3:
            return 0.5

        recovery_events = 0
        total_critical_events = 0

        for i in range(1, len(recent_results)):
            prev_roll = abs(recent_results[i - 1].get("roll", 0))
            curr_roll = abs(recent_results[i].get("roll", 0))

            if prev_roll > 0.4:
                total_critical_events += 1
                if curr_roll < 0.3 and curr_roll < prev_roll * 0.8:
                    recovery_events += 1

        if total_critical_events == 0:
            return 0.8

        return min(recovery_events / total_critical_events, 1.0)

    def _calculate_gait_coordination(self, recent_results: List[Dict]) -> float:
        """Calcular coordenação de marcha """
        if not recent_results:
            return 0.3

        coordination_scores = []
        for result in recent_results:
            left_contact = result.get("left_contact", False)
            right_contact = result.get("right_contact", False)
            alternation = 1.0 if left_contact != right_contact else 0.0

            cross_pattern = result.get("gait_pattern_score", 0.0)

            coordination = (alternation * 0.6 + cross_pattern * 0.4)
            coordination_scores.append(coordination)

        return np.mean(coordination_scores) if coordination_scores else 0.3

    def _calculate_propulsion_efficiency(self) -> float:
        """Calcular eficiência propulsiva para Fase 2"""
        if not self.progression_history:
            return 0.3

        recent_results = self.progression_history[-10:]
        speeds = [r.get("speed", 0) for r in recent_results]
        efforts = [r.get("energy_used", 1) for r in recent_results]

        efficiencies = []
        for speed, effort in zip(speeds, efforts):
            if effort > 0.1:  
                efficiency = speed / effort
                efficiencies.append(min(efficiency / 2.0, 1.0))

        return np.mean(efficiencies) if efficiencies else 0.3 

    def _apply_phase_config(self):
        """Aplica a configuração da fase atual ao sistema de recompensa"""
        current_phase = self.phases[self.current_phase]

        for component_name, component in self.reward_system.components.items():
            if component_name in current_phase.enabled_components:
                component.enabled = True
                target_weight = current_phase.component_weights.get(component_name, component.weight)
                component.weight = target_weight
            else:
                component.enabled = False

    def get_current_speed_target(self) -> float:
        """Retorna a velocidade alvo da fase atual"""
        return self.phases[self.current_phase].target_speed

    def get_detailed_status(self) -> Dict:
        """Retorna status detalhado com ambos históricos"""
        current_phase = self.phases[self.current_phase]
        success_rate = self._calculate_success_rate()
        avg_distance = self._calculate_average_distance()
        avg_speed = self._calculate_average_speed()
        avg_roll = self._calculate_average_roll()

        status = {
            "current_phase": current_phase.name,
            "phase_index": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "target_speed": current_phase.target_speed,
            # Métricas de performance
            "performance_metrics": {
                "success_rate": success_rate,
                "avg_distance": avg_distance,
                "avg_speed": avg_speed,
                "avg_roll": avg_roll,
                "progression_history_size": len(self.progression_history),  
                "full_history_size": len(self.performance_history)         
            },
            "success_rate": success_rate,
            "avg_distance": avg_distance,
            "avg_speed": avg_speed,
            "avg_roll": avg_roll,
            # Históricos
            "performance_history_size": len(self.performance_history),
            "progression_history_size": len(self.progression_history),
            # Contadores
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "regression_count": self.regression_count,
            # Requisitos da fase atual
            "requirements": {
                "min_success_rate": current_phase.transition_conditions.get("min_success_rate", 0),
                "min_avg_distance": current_phase.transition_conditions.get("min_avg_distance", 0),
                "max_avg_roll": current_phase.transition_conditions.get("max_avg_roll", 0),
                "min_avg_steps": current_phase.transition_conditions.get("min_avg_steps", 0),
            },
            # Transição
            "transition_active": self.transition_active,
            "transition_episodes": self.transition_episodes,
            "transition_total_episodes": self.transition_total_episodes,
            "transition_progress": self.transition_episodes / self.transition_total_episodes if self.transition_active else 1.0,
        }

        return status

    def _calculate_success_rate(self) -> float:
        """Calcula taxa de sucesso baseada no histórico de progressão"""
        if not self.progression_history:
            return 0.0
        successes = sum(1 for r in self.progression_history if r.get("phase_success", False))

        return successes / len(self.progression_history)

    def _calculate_average_distance(self) -> float:
        if not self.progression_history:  
            return 0.0
            
        return np.mean([r.get("distance", 0) for r in self.progression_history])

    def _calculate_average_speed(self) -> float:
        if not self.progression_history:  
            return 0.0
        
        return np.mean([r.get("speed", 0) for r in self.progression_history])

    def _calculate_average_roll(self) -> float:
        if not self.progression_history:
            return 0.0
        
        return np.mean([abs(r.get("roll", 0)) for r in self.progression_history])
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calcula métricas de performance atuais baseadas no histórico recente"""
        if not self.progression_history:
            return {
                "success_rate": 0.0,
                "avg_distance": 0.0,
                "avg_speed": 0.0,
                "avg_roll": 0.0,
                "consistency": 0.0,
                "energy_efficiency": 0.5
            }

        recent_history = self.progression_history[-10:] 

        # Taxa de sucesso
        successes = sum(1 for r in recent_history if r.get("phase_success", False))
        success_rate = successes / len(recent_history) if recent_history else 0.0

        # Distância média
        avg_distance = np.mean([r.get("distance", 0) for r in recent_history])

        # Velocidade média
        avg_speed = np.mean([r.get("speed", 0) for r in recent_history])

        # Estabilidade (roll médio)
        avg_roll = np.mean([abs(r.get("roll", 0)) for r in recent_history])

        # Consistência
        distances = [r.get("distance", 0) for r in recent_history]
        if len(distances) >= 2:
            distance_std = np.std(distances)
            distance_mean = np.mean(distances) if np.mean(distances) > 0 else 1.0
            consistency = 1.0 - min(distance_std / distance_mean, 1.0)
        else:
            consistency = 0.0

        # Eficiência energética
        energy_values = []
        for r in recent_history:
            distance = r.get("distance", 0)
            energy = r.get("energy_used", 1.0)
            if energy > 0.1:
                energy_values.append(distance / energy)

        energy_efficiency = np.mean(energy_values) / 2.0 if energy_values else 0.5

        return {
            "success_rate": success_rate,
            "avg_distance": avg_distance,
            "avg_speed": avg_speed,
            "avg_roll": avg_roll,
            "consistency": consistency,
            "energy_efficiency": min(energy_efficiency, 1.0)
        }

    def _extract_policy_features(self) -> Dict[str, float]:
        """Extrai features da política atual para análise DASS"""
        features = {
            "phase": self.current_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "stagnation_counter": self.stagnation_counter,
            "performance_trend": self._calculate_performance_trend(),
            "component_balance": self._calculate_component_balance_score(),
            "learning_stability": self._calculate_learning_stability()
        }

        # Adicionar métricas de performance atuais
        performance_metrics = self._calculate_performance_metrics()
        features.update(performance_metrics)

        return features

    def _calculate_performance_trend(self) -> float:
        """Calcula tendência de performance (positiva/negativa)"""
        if len(self.progression_history) < 5:
            return 0.5

        # Usar distâncias como indicador de tendência
        distances = [r.get("distance", 0) for r in self.progression_history[-5:]]
        if len(distances) < 2:
            return 0.5

        # Calcular inclinação da regressão linear
        x = np.arange(len(distances))
        slope, _ = np.polyfit(x, distances, 1)

        # Normalizar para 0-1 (0.5 = estável)
        trend = 0.5 + min(max(slope / 0.1, -0.5), 0.5)  

        return trend

    def _calculate_component_balance_score(self) -> float:
        """Calcula quão balanceados estão os componentes de recompensa"""
        if not hasattr(self, 'adaptive_components'):
            return 0.5

        performances = []
        for component in self.adaptive_components.values():
            if component['recent_rewards']:
                perf = np.mean(component['recent_rewards'][-5:])
                performances.append(perf)

        if len(performances) < 2:
            return 0.5

        # Score baseado na variância 
        variance = np.var(performances)
        balance_score = 1.0 - min(variance * 2.0, 1.0)

        return balance_score

    def _calculate_learning_stability(self) -> float:
        """Calcula estabilidade do aprendizado baseado na consistência"""
        if len(self.progression_history) < 3:
            return 0.5

        recent_changes = []
        for i in range(1, min(5, len(self.progression_history))):
            current = self.progression_history[-i].get('distance', 0)
            previous = self.progression_history[-i-1].get('distance', 0) if len(self.progression_history) > i+1 else current
            change = abs(current - previous)
            recent_changes.append(change)

        if not recent_changes:
            return 0.5

        avg_change = np.mean(recent_changes)
        stability = 1.0 - min(avg_change / 0.3, 1.0)  

        return stability

    def _assess_phase_skills(self) -> Dict[str, float]:
        """Cálculo de habilidades"""
        if len(self.progression_history) < 2:
            return self._get_default_skills()

        recent_results = self.progression_history[-8:]  

        # MÉTRICAS BÁSICAS 
        success_rate = self._calculate_success_rate()
        avg_roll = self._calculate_average_roll()
        avg_distance = self._calculate_average_distance()
        avg_speed = self._calculate_average_speed()

        # 1. BASIC BALANCE - estabilidade geral 
        z_positions = [r.get("imu_z", 0.8) for r in recent_results if r.get("imu_z") is not None]
        avg_height = np.mean(z_positions) if z_positions else 0.8
        height_stability = min(avg_height / 0.8, 1.0)  

        roll_values = [abs(r.get("roll", 0)) for r in recent_results]
        avg_roll_current = np.mean(roll_values) if roll_values else 0.0
        roll_stability = 1.0 - min(avg_roll_current / 1.0, 1.0)  

        basic_balance = (roll_stability * 0.7 + height_stability * 0.3)

        # 2. POSTURAL STABILITY - controle postural 
        if len(roll_values) > 1:
            roll_consistency = 1.0 - min(np.std(roll_values) / 0.5, 1.0)
        else:
            roll_consistency = 0.5
        postural_stability = (roll_stability * 0.6 + roll_consistency * 0.4)

        # 3. GAIT INITIATION - início da marcha
        gait_initiation = success_rate

        # 4. STEP CONSISTENCY - consistência de passos 
        distances = [r.get("distance", 0) for r in recent_results]
        if len(distances) >= 3 and np.mean(distances) > 0.1:
            std_dev = np.std(distances)
            avg_dist = np.mean(distances)
            cv = std_dev / avg_dist if avg_dist > 0 else 1.0
            step_consistency = max(0.0, 1.0 - min(cv, 2.0) / 2.0)
        else:
            step_consistency = 0.1

        # 5. DYNAMIC BALANCE - equilíbrio dinâmico 
        dynamic_balance_factors = []

        # Fator 1: Estabilidade durante movimento
        if avg_speed > 0.1:  
            speed_balance = 1.0 - min(avg_roll_current / (0.4 + avg_speed * 0.5), 1.0)
            dynamic_balance_factors.append(speed_balance * 0.6)
        else:
            dynamic_balance_factors.append(0.3)  

        # Fator 2: Recuperação de equilíbrio
        recovery_score = self._calculate_balance_recovery_score(recent_results)
        dynamic_balance_factors.append(recovery_score * 0.4)

        dynamic_balance = np.mean(dynamic_balance_factors) if dynamic_balance_factors else 0.3

        # 6. BALANCE RECOVERY - recuperação de equilíbrio
        balance_recovery = self._calculate_balance_recovery_score(recent_results)

        # 7. PROPULSIVE PHASE - fase propulsiva
        propulsion_efficiency = self._calculate_propulsion_efficiency()

        # 8. ENERGY EFFICIENCY - eficiência energética
        energy_efficiency = self._calculate_energy_efficiency(recent_results)

        # 9. GAIT COORDINATION - coordenação da marcha
        gait_coordination = self._calculate_gait_coordination(recent_results)

        all_skills = {
            "basic_balance": min(basic_balance, 1.0),
            "postural_stability": min(postural_stability, 1.0),
            "gait_initiation": gait_initiation,
            "step_consistency": step_consistency,
            "dynamic_balance": min(dynamic_balance, 1.0),  
            "balance_recovery": balance_recovery,
            "propulsive_phase": propulsion_efficiency,
            "energy_efficiency": energy_efficiency,
            "gait_coordination": gait_coordination,
        }

        # DEBUG: Log das habilidades calculadas
        self.logger.debug(f"Habilidades calculadas: dynamic_balance={dynamic_balance:.3f}, "
                         f"roll_stability={roll_stability:.3f}, avg_speed={avg_speed:.3f}")

        return all_skills
    
    def _get_default_skills(self) -> Dict[str, float]:
        """Retorna habilidades padrão quando não há histórico suficiente"""
        return {
            "basic_balance": 0.3,
            "postural_stability": 0.2, 
            "gait_initiation": 0.1,
            "step_consistency": 0.1,
            "dynamic_balance": 0.1,  
            "balance_recovery": 0.1,
            "propulsive_phase": 0.1,
            "energy_efficiency": 0.1,
            "gait_coordination": 0.1,
        }