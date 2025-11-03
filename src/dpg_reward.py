# dpg_reward.py
import numpy as np
from typing import Dict, List, Callable, Tuple
from dataclasses import dataclass


class AdaptiveIRL:
    """Sistema IRL Adaptável por Necessidade"""
    
    def __init__(self, logger):
        self.logger = logger
        self.irl_modes = {
            "disabled": {"samples_required": 0, "complexity": 0, "features": []},
            "light": {"samples_required": 200, "complexity": 1, "features": ["progress", "stability"]},
            "standard": {"samples_required": 500, "complexity": 2, "features": ["progress", "stability", "efficiency", "consistency"]},
            "advanced": {"samples_required": 1000, "complexity": 3, "features": ["progress", "stability", "efficiency", "consistency", "coordination", "adaptation"]}
        }
        
        self.learned_models = {}
        self.demonstration_buffer = []

    def get_irl_mode(self, group_level: int, learning_progress: float, sample_count: int) -> str:
        """Seleciona modo IRL baseado na necessidade"""
        
        # Grupo 1: IRL light apenas se progresso suficiente
        if group_level == 1:
            if learning_progress > 0.6 and sample_count >= 200:
                return "light"
            return "disabled"
        
        # Grupo 2: IRL standard para desenvolvimento
        elif group_level == 2:
            if learning_progress > 0.7 and sample_count >= 500:
                return "standard"
            elif sample_count >= 200:
                return "light"
            return "disabled"
        
        # Grupo 3: IRL advanced para domínio
        elif group_level == 3:
            if learning_progress > 0.8 and sample_count >= 1000:
                return "advanced"
            elif sample_count >= 500:
                return "standard"
            return "light"
        
        return "disabled"
    
    def collect_demonstration(self, experience: Dict, quality: float):
        """Coleta demonstração para IRL"""
        if quality > 0.7:  # Apenas demonstrações de alta qualidade
            self.demonstration_buffer.append({
                'experience': experience,
                'quality': quality,
                'timestamp': np.datetime64('now')
            })
            
            # Manter buffer limitado
            if len(self.demonstration_buffer) > 2000:
                self.demonstration_buffer.pop(0)
    
    def execute_irl_learning(self, mode: str, group_level: int) -> Dict:
        """Executa aprendizado IRL no modo especificado"""
        if mode == "disabled" or len(self.demonstration_buffer) < 100:
            return {}
        
        mode_config = self.irl_modes[mode]
        demonstrations = self._filter_high_quality_demos()
        
        if len(demonstrations) < mode_config["samples_required"]:
            return {}
        
        try:
            if mode == "light":
                learned_weights = self._light_irl(demonstrations, mode_config["features"])
            elif mode == "standard":
                learned_weights = self._standard_irl(demonstrations, mode_config["features"])
            elif mode == "advanced":
                learned_weights = self._advanced_irl(demonstrations, mode_config["features"])
            else:
                return {}
            
            # Calcular confiança do modelo
            confidence = self._calculate_model_confidence(learned_weights, demonstrations)
            
            model = {
                'weights': learned_weights,
                'features': mode_config["features"],
                'confidence': confidence,
                'mode': mode,
                'group_level': group_level,
                'sample_size': len(demonstrations),
                'timestamp': np.datetime64('now')
            }
            
            self.learned_models[f"group_{group_level}"] = model
            self.logger.info(f"🧠 IRL {mode} aprendido: {len(demonstrations)} amostras, confiança: {confidence:.2f}")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"IRL learning failed: {e}")
            return {}
    
    def _light_irl(self, demonstrations: List, features: List[str]) -> Dict[str, float]:
        """IRL simplificado para grupos iniciais"""
        feature_importance = {}
        
        for feature in features:
            if feature == "progress":
                # Progresso é sempre importante
                progresses = [d['experience']['metrics'].get('distance', 0) for d in demonstrations]
                feature_importance[feature] = min(np.mean(progresses) * 2.0, 1.0)
            
            elif feature == "stability":
                # Estabilidade é crucial
                stabilities = []
                for d in demonstrations:
                    roll = d['experience']['metrics'].get('roll', 0)
                    pitch = d['experience']['metrics'].get('pitch', 0)
                    stability = 1.0 - min(abs(roll) + abs(pitch), 1.0)
                    stabilities.append(stability)
                feature_importance[feature] = np.mean(stabilities)
        
        # Normalizar pesos
        total = sum(feature_importance.values())
        if total > 0:
            return {k: v/total for k, v in feature_importance.items()}
        
        return feature_importance
    
    def _standard_irl(self, demonstrations: List, features: List[str]) -> Dict[str, float]:
        """IRL padrão para desenvolvimento"""
        feature_scores = {}
        
        for feature in features:
            scores = []
            for demo in demonstrations:
                score = self._calculate_feature_score(feature, demo['experience'])
                scores.append(score * demo['quality'])  # Ponderar pela qualidade
            
            feature_scores[feature] = np.mean(scores) if scores else 0.5
        
        # Aplicar suavização
        smoothed = self._smooth_feature_weights(feature_scores)
        
        # Normalizar
        total = sum(smoothed.values())
        if total > 0:
            return {k: v/total for k, v in smoothed.items()}
        
        return smoothed
    
    def _advanced_irl(self, demonstrations: List, features: List[str]) -> Dict[str, float]:
        """IRL avançado para domínio"""
        # Usar abordagem mais sofisticada com correlações
        feature_matrix = []
        performance_scores = []
        
        for demo in demonstrations:
            features_vec = []
            for feature in features:
                features_vec.append(self._calculate_feature_score(feature, demo['experience']))
            feature_matrix.append(features_vec)
            performance_scores.append(demo['quality'] * demo['experience']['reward'])
        
        # Aprender pesos via análise de correlação
        if len(feature_matrix) > 50:
            correlations = []
            for i in range(len(features)):
                corr = np.corrcoef([vec[i] for vec in feature_matrix], performance_scores)[0,1]
                correlations.append(max(0, corr) if not np.isnan(corr) else 0.0)
            
            weights = {features[i]: correlations[i] for i in range(len(features))}
            
            # Normalizar e aplicar regularização
            total = sum(weights.values())
            if total > 0:
                normalized = {k: v/total for k, v in weights.items()}
                return self._apply_regularization(normalized)
        
        return self._standard_irl(demonstrations, features)  # Fallback
    
    def _calculate_feature_score(self, feature: str, experience: Dict) -> float:
        """Calcula score para uma feature específica"""
        metrics = experience['metrics']
        
        if feature == "progress":
            return min(metrics.get('distance', 0) / 2.0, 1.0)
        
        elif feature == "stability":
            roll = abs(metrics.get('roll', 0))
            pitch = abs(metrics.get('pitch', 0))
            return 1.0 - min(roll + pitch, 1.0)
        
        elif feature == "efficiency":
            distance = metrics.get('distance', 0)
            steps = metrics.get('steps', 1)
            return min(distance / steps, 1.0)
        
        elif feature == "consistency":
            # Para simplificar, usar estabilidade como proxy
            return 1.0 - min(metrics.get('roll', 0) * 2.0, 1.0)
        
        elif feature == "coordination":
            left_contact = metrics.get('left_contact', False)
            right_contact = metrics.get('right_contact', False)
            return 1.0 if left_contact != right_contact else 0.3
        
        elif feature == "adaptation":
            # Score baseado na variação (simplificado)
            return 0.7  # Placeholder
        
        return 0.5
    
    def _smooth_feature_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Suaviza pesos para evitar extremos"""
        smoothed = {}
        for feature, weight in weights.items():
            # Aplicar transformação suave
            smoothed[feature] = np.sqrt(weight)  # Reduz extremos
        return smoothed
    
    def _apply_regularization(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Aplica regularização para evitar overfitting"""
        regularized = {}
        for feature, weight in weights.items():
            # Regularização L2 leve
            regularized[feature] = weight * 0.9 + 0.1 * (1.0 / len(weights))
        return regularized
    
    def _filter_high_quality_demos(self) -> List:
        """Filtra demonstrações de alta qualidade"""
        return [d for d in self.demonstration_buffer if d['quality'] > 0.8]
    
    def _calculate_model_confidence(self, weights: Dict, demonstrations: List) -> float:
        """Calcula confiança no modelo aprendido"""
        if not demonstrations:
            return 0.0
        
        # Confiança baseada na consistência das demonstrações
        consistencies = []
        for demo in demonstrations[:50]:  # Amostrar primeiras 50
            predicted_score = 0.0
            for feature, weight in weights.items():
                feature_score = self._calculate_feature_score(feature, demo['experience'])
                predicted_score += feature_score * weight
            
            actual_performance = demo['quality'] * demo['experience']['reward']
            error = abs(predicted_score - actual_performance)
            consistency = 1.0 - min(error, 1.0)
            consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.5
    
    def get_learned_weights(self, group_level: int) -> Dict[str, float]:
        """Retorna pesos aprendidos para um grupo"""
        model = self.learned_models.get(f"group_{group_level}")
        return model.get('weights', {}) if model else {}
    
    def get_irl_status(self) -> Dict:
        """Retorna status do sistema IRL"""
        return {
            "demonstration_count": len(self.demonstration_buffer),
            "learned_models": len(self.learned_models),
            "current_modes": list(self.irl_modes.keys()),
            "model_confidences": {k: v.get('confidence', 0) for k, v in self.learned_models.items()}
        }


@dataclass
class RewardComponent:
    name: str
    weight: float
    calculator: Callable
    enabled: bool = True
    adaptive_weight: float = 1.0


class RewardCalculator:
    """
    ESPECIALISTA EM RECOMPENSAS com IRL Adaptável
    """
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.components = self._initialize_components()
        self.irl_system = AdaptiveIRL(logger)
        
    def _initialize_components(self) -> Dict[str, RewardComponent]:
        """Inicializa componentes de recompensa"""
        return {
            "stability": RewardComponent("stability", 4.0, self._calculate_stability_reward),
            "basic_progress": RewardComponent("basic_progress", 3.0, self._calculate_basic_progress_reward),
            "posture": RewardComponent("posture", 3.5, self._calculate_posture_reward),
            "direction": RewardComponent("direction", 2.0, self._calculate_direction_reward),
            "velocity": RewardComponent("velocity", 1.5, self._calculate_velocity_reward),
            "phase_angles": RewardComponent("phase_angles", 1.0, self._calculate_phase_angles_reward),
            "propulsion": RewardComponent("propulsion", 0.5, self._calculate_propulsion_reward),
            "clearance": RewardComponent("clearance", 0.5, self._calculate_clearance_reward),
            "coordination": RewardComponent("coordination", 1.0, self._calculate_coordination_reward),
            "efficiency": RewardComponent("efficiency", 0.8, self._calculate_efficiency_reward),
            "success_bonus": RewardComponent("success_bonus", 5.0, self._calculate_success_bonus),
            "effort_penalty": RewardComponent("effort_penalty", 0.008, self._calculate_effort_penalty),
        }
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
        """Calcula recompensa com influência do IRL"""
        total_reward = 0.0
        enabled_components = phase_info['enabled_components']
        group_level = phase_info.get('group_level', 1)
        
        # Coletar demonstração para IRL
        experience_metrics = self._extract_experience_metrics(sim)
        experience_quality = self._estimate_experience_quality(experience_metrics)
        
        # Coletar para IRL se qualidade suficiente
        if experience_quality > 0.5:
            self.irl_system.collect_demonstration({
                'metrics': experience_metrics,
                'reward': 0.0,  # Será calculado
                'phase_info': phase_info
            }, experience_quality)
        
        # Executar IRL se necessário
        irl_mode = self.irl_system.get_irl_mode(
            group_level,
            phase_info.get('learning_progress', 0.5),
            len(self.irl_system.demonstration_buffer)
        )
        
        if irl_mode != "disabled":
            irl_model = self.irl_system.execute_irl_learning(irl_mode, group_level)
            if irl_model and irl_model['confidence'] > 0.6:
                self._apply_irl_weights(irl_model['weights'])
        
        # Calcular recompensa base
        for component_name in enabled_components:
            if component_name in self.components:
                component = self.components[component_name]
                if component.enabled:
                    component_reward = component.calculator(sim, phase_info)
                    weighted_reward = component.weight * component.adaptive_weight * component_reward
                    total_reward += weighted_reward
        
        # Aplicar penalidades
        penalties = self._calculate_global_penalties(sim, action, group_level)
        total_reward -= penalties
        
        return total_reward
    
    def _apply_irl_weights(self, irl_weights: Dict[str, float]):
        """Aplica pesos aprendidos pelo IRL aos componentes"""
        for feature, weight in irl_weights.items():
            # Mapear features IRL para componentes de recompensa
            component_map = {
                "progress": ["basic_progress", "velocity", "propulsion"],
                "stability": ["stability", "posture"],
                "efficiency": ["efficiency", "effort_penalty"],
                "coordination": ["coordination", "clearance", "phase_angles"]
            }
            
            for component_name in component_map.get(feature, []):
                if component_name in self.components:
                    # Ajuste suave baseado no IRL
                    current_weight = self.components[component_name].adaptive_weight
                    new_weight = 0.7 * current_weight + 0.3 * weight
                    self.components[component_name].adaptive_weight = new_weight
    
    def _extract_experience_metrics(self, sim) -> Dict:
        """Extrai métricas da experiência para IRL"""
        return {
            "distance": getattr(sim, "episode_distance", 0),
            "speed": getattr(sim, "robot_x_velocity", 0),
            "roll": abs(getattr(sim, "robot_roll", 0)),
            "pitch": abs(getattr(sim, "robot_pitch", 0)),
            "steps": getattr(sim, "episode_steps", 0),
            "left_contact": getattr(sim, "robot_left_foot_contact", False),
            "right_contact": getattr(sim, "robot_right_foot_contact", False),
            "success": getattr(sim, "episode_success", False)
        }
    
    def _estimate_experience_quality(self, metrics: Dict) -> float:
        """Estima qualidade da experiência para IRL"""
        quality = 0.0
        
        if metrics.get("success", False):
            quality += 0.8
        
        distance = metrics.get("distance", 0)
        if distance > 0.5:
            quality += 0.6
        elif distance > 0.1:
            quality += 0.3
        
        stability = 1.0 - min(metrics.get("roll", 0), 0.8)
        quality += stability * 0.4

        alternating = metrics.get("left_contact", False) != metrics.get("right_contact", False)
        if alternating:
            quality += 0.2
        
        speed = metrics.get("speed", 0)
        if 0.1 < speed < 1.0:
            quality += 0.2
        
        return min(quality, 1.0)
    
    # Implementações dos componentes de recompensa 
    def _calculate_stability_reward(self, sim, phase_info) -> float:
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))
        roll_penalty = min(roll * 1.5, 0.8)
        pitch_penalty = min(pitch * 1.0, 0.6)
        total_penalty = (roll_penalty * 0.8) + (pitch_penalty * 0.2)
    
        return 1.0 - total_penalty
    
    def _calculate_basic_progress_reward(self, sim, phase_info) -> float:
        distance = getattr(sim, "episode_distance", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        distance_reward = min(distance / 0.5, 3.0)  
        if velocity > 0.1:
            velocity_reward = min(velocity * 2.0, 2.0)
        else:
            velocity_reward = 0.0
        total_reward = (distance_reward * 0.8) + (velocity_reward * 0.2)

        return min(total_reward, 3.0)
    
    def _calculate_direction_reward(self, sim, phase_info) -> float:
        """Recompensa por manter direção correta"""
        y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
        y_position = abs(getattr(sim, "robot_y_position", 0))
        lateral_penalty = min(y_velocity * 0.5, 0.3)
        position_penalty = min(y_position * 0.3, 0.2)
        total_penalty = lateral_penalty + position_penalty

        return 1.0 - min(total_penalty, 0.5)

    def _calculate_posture_reward(self, sim, phase_info) -> float:
        pitch = getattr(sim, "robot_pitch", 0)
        if pitch > 0.3: 
            penalty = min(pitch * 2.0, 1.0)
            return 1.0 - penalty
        else:
            return 1.0
    
    def _calculate_velocity_reward(self, sim, phase_info) -> float:
        vx = getattr(sim, "robot_x_velocity", 0)
        target_speed = phase_info['target_speed']
        if vx < -0.1:
            return -0.5
        v_min = target_speed * 0.1
        v_max = target_speed * 1.5
        if v_max - v_min > 0:
            normalized_vel = (vx - v_min) / (v_max - v_min)
            return np.clip(normalized_vel, 0.0, 1.0)
        return 0.0
    
    def _calculate_phase_angles_reward(self, sim, phase_info) -> float:
        try:
            left_knee = abs(getattr(sim, "robot_left_knee_angle", 0))
            right_knee = abs(getattr(sim, "robot_right_knee_angle", 0))
            ideal_knee = 0.4
            knee_reward = np.exp(-0.5 * (left_knee - ideal_knee)**2 / 0.25**2)
            knee_reward += np.exp(-0.5 * (right_knee - ideal_knee)**2 / 0.25**2)
            if left_knee > 0.3 and right_knee > 0.3:
                knee_reward += 0.3
            
            return min(knee_reward / 2.0, 1.0)
        except:
            return 0.5
    
    def _calculate_propulsion_reward(self, sim, phase_info) -> float:
        pitch = getattr(sim, "robot_pitch", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        if pitch < -0.05 and velocity > 0.1:
            return min(abs(pitch) * velocity * 3.0, 1.0)
        elif pitch < -0.1:
            return 0.3
        return 0.0
    
    def _calculate_clearance_reward(self, sim, phase_info) -> float:
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            clearance_reward = 0.0
            if not left_contact:
                left_height = getattr(sim, "robot_left_foot_height", 0)
                if left_height > 0.05:
                    clearance_reward += 0.9
            if not right_contact:
                right_height = getattr(sim, "robot_right_foot_height", 0)
                if right_height > 0.05:
                    clearance_reward += 0.7
            return clearance_reward / 2.0
        except:
            return 0.0
    
    def _calculate_coordination_reward(self, sim, phase_info) -> float:
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            left_arm = getattr(sim, "robot_left_shoulder_front_angle", 0)
            right_arm = getattr(sim, "robot_right_shoulder_front_angle", 0)
            coordination = 0.0
            if not right_contact and left_arm > 0.15:
                coordination += 0.7
            if not left_contact and right_arm > 0.15:
                coordination += 0.7
            if left_contact != right_contact:
                coordination += 0.3
            
            return min(coordination, 1.0)
        except:
            return 0.0
    
    def _calculate_efficiency_reward(self, sim, phase_info) -> float:
        distance = getattr(sim, "episode_distance", 0)
        steps = getattr(sim, "episode_steps", 1)
        efficiency = distance / steps
        return min(efficiency * 3.0, 1.0)
    
    def _calculate_success_bonus(self, sim, phase_info) -> float:
        success = getattr(sim, "episode_success", False)
        return 1.0 if success else 0.0
    
    def _calculate_effort_penalty(self, sim, phase_info) -> float:
        try:
            joint_velocities = getattr(sim, "joint_velocities", [0])
            effort = sum(v**2 for v in joint_velocities)
            return min(effort * 0.1, 2.0)
        except:
            return 0.0
    
    def _calculate_global_penalties(self, sim, action, group_level: int) -> float:
        """Calcula penalidades globais adaptadas ao grupo"""
        penalties = 0.0
        
        # Penalidade de ação extrema 
        if hasattr(action, '__len__'):
            action_penalty = np.sum(np.abs(action)) * 0.005
            group_tolerance = 1.0 - (group_level * 0.2) 
            penalties += min(action_penalty * group_tolerance, 0.5)
        
        # Penalidade por queda iminente
        height = getattr(sim, "robot_z_position", 0.8)
        if height < 0.5:
            penalties += (0.5 - height) * 2.0

        # Penalidade por movimento lateral excessivo
        y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
        if y_velocity > 0.3:
            penalties += (y_velocity - 0.3) * 1.0

        # Penalidade por inclinação excessiva
        roll = abs(getattr(sim, "robot_roll", 0))
        if roll > 0.8:
            penalties += (roll - 0.8) * 0.5
        
        return penalties
    
    def adjust_component_weights(self, weight_adjustments: Dict[str, float]):
        """Ajusta pesos dos componentes baseado em feedback externo"""
        for component_name, adjustment in weight_adjustments.items():
            if component_name in self.components:
                self.components[component_name].adaptive_weight = adjustment
    
    def enable_components(self, component_names: List[str]):
        """Habilita componentes específicos"""
        for name in self.components:
            self.components[name].enabled = (name in component_names)
    
    def get_component_weights(self) -> Dict[str, float]:
        """Retorna pesos atuais dos componentes"""
        return {
            name: component.weight * component.adaptive_weight
            for name, component in self.components.items()
            if component.enabled
        }
    
    def get_component_performance(self, sim, phase_info) -> Dict[str, float]:
        """Retorna performance individual de cada componente"""
        performance = {}
        group_level = phase_info.get('group_level', 1)
        
        for name, component in self.components.items():
            if component.enabled and group_level in component.group_affinity:
                raw_reward = component.calculator(sim, phase_info)
                weighted_reward = self._adjust_component_weight(
                    component, group_level, phase_info.get('sub_phase', 0)
                ) * raw_reward
                
                performance[name] = {
                    'raw': raw_reward,
                    'weighted': weighted_reward,
                    'weight': component.weight,
                    'adaptive_weight': component.adaptive_weight,
                    'group_affinity': component.group_affinity
                }
        
        return performance
    
    def get_reward_status(self) -> Dict:
        """Retorna status do sistema de recompensa com IRL"""
        status = {
            "components_enabled": len([c for c in self.components.values() if c.enabled]),
            "adaptive_weights": {name: c.adaptive_weight for name, c in self.components.items()}
        }
        
        # Adicionar status IRL
        status.update(self.irl_system.get_irl_status())
        
        return status