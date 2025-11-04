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
    ESPECIALISTA EM RECOMPENSAS - Versão Otimizada
    """
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.components = self._initialize_components()
    
    def _initialize_components(self) -> Dict[str, RewardComponent]:
        """Inicializa componentes de recompensa"""
        return {
            "stability": RewardComponent("stability", 3.0, self._calculate_stability_reward),
            "basic_progress": RewardComponent("basic_progress", 2.0, self._calculate_basic_progress_reward),
            "posture": RewardComponent("posture", 2.5, self._calculate_posture_reward),
            "velocity": RewardComponent("velocity", 1.5, self._calculate_velocity_reward),
            "phase_angles": RewardComponent("phase_angles", 1.0, self._calculate_phase_angles_reward),
            "propulsion": RewardComponent("propulsion", 1.0, self._calculate_propulsion_reward),
            "clearance": RewardComponent("clearance", 0.8, self._calculate_clearance_reward),
            "coordination": RewardComponent("coordination", 1.2, self._calculate_coordination_reward),
            "efficiency": RewardComponent("efficiency", 0.8, self._calculate_efficiency_reward),
            "success_bonus": RewardComponent("success_bonus", 5.0, self._calculate_success_bonus),
            "effort_penalty": RewardComponent("effort_penalty", 0.005, self._calculate_effort_penalty),
        }
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
        """Calcula recompensa usando sistema de valências com cache"""
        total_reward = 0.0

        enabled_components = phase_info['enabled_components']
        valence_weights = phase_info.get('valence_weights', {})
        irl_weights = phase_info.get('irl_weights', {})
        use_irl = len(irl_weights) > 0
        
        for component_name in enabled_components:
            if component_name in self.components:
                component = self.components[component_name]
                component_reward = component.calculator(sim, phase_info)
                
                if use_irl and component_name in irl_weights:
                    weight = irl_weights[component_name]
                    irl_bonus = 0.2  
                elif valence_weights and component_name in valence_weights:
                    weight = valence_weights[component_name]
                    irl_bonus = 0.0
                else:
                    weight = component.weight * component.adaptive_weight
                    irl_bonus = 0.0
                
                weighted_reward = weight * component_reward * (1.0 + irl_bonus)
                total_reward += weighted_reward
        
        penalties = self._calculate_global_penalties(sim, action)
        total_reward -= penalties
        
        return max(total_reward, -0.5)
    
    def _calculate_global_penalties(self, sim, action) -> float:
        """Calcula penalidades globais mais balanceadas"""
        penalties = 0.0

        # 1. Penalidade de ação extrema 
        if hasattr(action, '__len__'):
            action_magnitude = np.sqrt(np.sum(np.square(action)))
            action_penalty = action_magnitude * 0.005  
            penalties += min(action_penalty, 0.3)  

        # 2. Penalidade por queda iminente 
        height = getattr(sim, "robot_z_position", 0.8)
        if height < 0.5:
            fall_penalty = (0.5 - height) * 1.5 
            penalties += min(fall_penalty, 0.8)  

        # 3. Penalidade por movimento lateral excessivo 
        y_velocity = abs(getattr(sim, "robot_y_velocity", 0))
        if y_velocity > 0.3:  
            lateral_penalty = (y_velocity - 0.3) * 1.0  
            penalties += min(lateral_penalty, 0.5)  

        # 4. Penalidade por inclinação excessiva 
        roll = abs(getattr(sim, "robot_roll", 0))
        if roll > 0.7:  
            roll_penalty = (roll - 0.7) * 0.8  
            penalties += min(roll_penalty, 0.4)  

        pitch = abs(getattr(sim, "robot_pitch", 0))
        if pitch > 0.7:  
            pitch_penalty = (pitch - 0.7) * 0.6  
            penalties += min(pitch_penalty, 0.3)  

        return penalties
    
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
            "success": getattr(sim, "episode_success", False),
            "energy_used": getattr(sim, "robot_energy_used", 1.0),
            "gait_score": getattr(sim, "robot_gait_pattern_score", 0.5),
        }
    
    def _estimate_experience_quality(self, metrics: Dict) -> float:
        """Estima qualidade da experiência para IRL"""
        quality = 0.0
        
        # Sucesso é muito importante
        if metrics.get("success", False):
            quality += 0.6
        
        # Progresso em distância
        distance = metrics.get("distance", 0)
        if distance > 1.0:
            quality += 0.3
        elif distance > 0.5:
            quality += 0.2
        elif distance > 0.1:
            quality += 0.1
        
        # Estabilidade
        roll = metrics.get("roll", 0)
        pitch = metrics.get("pitch", 0)
        stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
        quality += stability * 0.3
        
        # Coordenação
        alternating = metrics.get("left_contact", False) != metrics.get("right_contact", False)
        if alternating:
            quality += 0.2
        
        # Velocidade adequada
        speed = metrics.get("speed", 0)
        if 0.1 < speed < 2.0:
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _calculate_stability_reward(self, sim, phase_info) -> float:
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))
        stability = 1.0 - min((roll * 1.2 + pitch * 0.8) / 2.0, 1.0)
        return stability
    
    def _calculate_basic_progress_reward(self, sim, phase_info) -> float:
        distance = getattr(sim, "episode_distance", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        distance_reward = min(distance / 2.0, 1.0)
        velocity_reward = min(abs(velocity) / 1.5, 1.0) if velocity > 0 else 0.0
        
        return (distance_reward * 0.7 + velocity_reward * 0.3)
    
    def _calculate_posture_reward(self, sim, phase_info) -> float:
        pitch = getattr(sim, "robot_pitch", 0)
        if pitch < -0.1:
            return min(abs(pitch) * 2.0, 1.0)
        elif pitch > 0.3:  
            return 1.0 - min((pitch - 0.3) * 2.0, 1.0)
        else:
            return 0.8  
    
    def _calculate_velocity_reward(self, sim, phase_info) -> float:
        target_speed = phase_info.get('target_speed', 1.0)
        current_speed = getattr(sim, "robot_x_velocity", 0)
        if current_speed < 0:
            return -0.2  
        speed_ratio = current_speed / target_speed if target_speed > 0 else 0
        if 0.8 <= speed_ratio <= 1.2:
            return 1.0 
        elif 0.5 <= speed_ratio < 0.8 or 1.2 < speed_ratio <= 1.5:
            return 0.5  
        else:
            return 0.1 
    
    def _calculate_phase_angles_reward(self, sim, phase_info) -> float:
        try:
            left_knee = abs(getattr(sim, "robot_left_knee_angle", 0))
            right_knee = abs(getattr(sim, "robot_right_knee_angle", 0))
            ideal_knee = 0.4
            left_score = np.exp(-2.0 * (left_knee - ideal_knee)**2)
            right_score = np.exp(-2.0 * (right_knee - ideal_knee)**2)
            return (left_score + right_score) / 2.0
        except:
            return 0.5
    
    def _calculate_propulsion_reward(self, sim, phase_info) -> float:
        pitch = getattr(sim, "robot_pitch", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        if pitch < -0.05 and velocity > 0.1:
            return min(abs(pitch) * velocity * 4.0, 1.0)
        return 0.0
    
    def _calculate_clearance_reward(self, sim, phase_info) -> float:
        try:
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)
            left_height = getattr(sim, "robot_left_foot_height", 0)
            right_height = getattr(sim, "robot_right_foot_height", 0)
            clearance_reward = 0.0
            if not left_contact and left_height > 0.05:
                clearance_reward += 0.6
            if not right_contact and right_height > 0.05:
                clearance_reward += 0.6
            return min(clearance_reward, 1.0)
        except:
            return 0.0
    
    def _calculate_coordination_reward(self, sim, phase_info) -> float:
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
    
    def _calculate_efficiency_reward(self, sim, phase_info) -> float:
        distance = getattr(sim, "episode_distance", 0)
        steps = max(getattr(sim, "episode_steps", 1), 1)
        energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
        steps_efficiency = distance / steps
        energy_efficiency = distance / energy
        combined_efficiency = (steps_efficiency * 0.6 + energy_efficiency * 0.4)
        return min(combined_efficiency * 2.0, 1.0)
    
    def _calculate_success_bonus(self, sim, phase_info) -> float:
        success = getattr(sim, "episode_success", False)
        return 1.0 if success else 0.0
    
    def _calculate_effort_penalty(self, sim, phase_info) -> float:
        try:
            joint_velocities = getattr(sim, "joint_velocities", [0])
            effort = sum(v**2 for v in joint_velocities) / len(joint_velocities) if joint_velocities else 0
            return min(effort * 0.1, 0.5)
        except:
            return 0.0
    
    def get_reward_status(self) -> Dict:
        """Retorna status do sistema de recompensa"""
        active_components = [name for name, comp in self.components.items() if comp.enabled]
        status = {
            "active_components": active_components,
            "total_components": len(self.components),
            "irl_demonstrations": len(self.irl_system.demonstration_buffer) if hasattr(self.irl_system, 'demonstration_buffer') else 0,
        }
        
        return status