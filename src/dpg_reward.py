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
        consistencies = []
        for demo in demonstrations[:50]: 
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
            "dynamic_balance": RewardComponent("dynamic_balance", 1.5, self._calculate_dynamic_balance_reward),
            "smoothness": RewardComponent("smoothness", 1.0, self._calculate_smoothness_reward),
            "rhythm": RewardComponent("rhythm", 1.2, self._calculate_rhythm_reward),
            "gait_pattern": RewardComponent("gait_pattern", 1.0, self._calculate_gait_pattern_reward),
            "biomechanics": RewardComponent("biomechanics", 0.8, self._calculate_biomechanics_reward),
            "robustness": RewardComponent("robustness", 0.5, self._calculate_robustness_reward),
            "adaptation": RewardComponent("adaptation", 0.5, self._calculate_adaptation_reward),
            "recovery": RewardComponent("recovery", 0.8, self._calculate_recovery_reward),
        }
    
    def calculate(self, sim, action, phase_info: Dict) -> float:
        """Calcula recompensa usando sistema de valências com cache - VERSÃO BALANCEADA"""
        total_reward = 0.0

        enabled_components = phase_info['enabled_components']
        valence_weights = phase_info.get('valence_weights', {})
        irl_weights = phase_info.get('irl_weights', {})

        use_irl = len(irl_weights) > 0 and phase_info.get('learning_progress', 0) > 0.7
        irl_bonus = 0.1 if use_irl else 0.0 

        for component_name in enabled_components:
            if component_name in self.components:
                component = self.components[component_name]
                component_reward = component.calculator(sim, phase_info)

                if use_irl and component_name in irl_weights:
                    weight = irl_weights[component_name]
                elif valence_weights and component_name in valence_weights:
                    weight = valence_weights[component_name]
                else:
                    weight = component.weight * component.adaptive_weight

                weighted_reward = weight * component_reward
                total_reward += weighted_reward

        penalties = self._calculate_global_penalties(sim, action)
        total_reward -= penalties

        return max(total_reward, -0.5)
    
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
        distance = getattr(sim, "episode_distance", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        if distance > 2.0:  
            distance_reward = 3.0
        elif distance > 1.0:  
            distance_reward = 2.0
        elif distance > 0.5:  
            distance_reward = 1.0
        else:  
            distance_reward = 0.3
        velocity_reward = min(abs(velocity) / 0.8, 2.0) if velocity > 0.02 else 0.0

        return (distance_reward * 0.8 + velocity_reward * 0.2)
    
    def _calculate_posture_reward(self, sim, phase_info) -> float:
        try:
            pitch = getattr(sim, "robot_pitch", 0)
            ideal_pitch = 0.15
            pitch_tolerance = 0.25
            pitch_diff = abs(pitch - ideal_pitch)
            if pitch_diff <= pitch_tolerance:
                pitch_score = 1.0 - (pitch_diff / pitch_tolerance) * 0.3  
            else:
                pitch_score = 0.7 - min((pitch_diff - pitch_tolerance) * 0.5, 0.6)  
            left_hip_lateral = abs(getattr(sim, "robot_left_hip_lateral_angle", 0))
            right_hip_lateral = abs(getattr(sim, "robot_right_hip_lateral_angle", 0))
            max_leg_spread = max(left_hip_lateral, right_hip_lateral)
            if max_leg_spread <= 0.4:    
                leg_spread_score = 1.0
            elif max_leg_spread <= 0.8:  
                leg_spread_score = 0.7
            else:                       
                leg_spread_score = 0.3
            com_height = getattr(sim, "robot_z_position", 0.8)
            height_diff = abs(com_height - 0.8)
            if height_diff <= 0.15:      
                height_score = 1.0
            elif height_diff <= 0.25:     
                height_score = 0.6
            else:                        
                height_score = 0.2
            total_score = (
                pitch_score * 0.5 +      
                leg_spread_score * 0.3 + 
                height_score * 0.2       
            )
            if abs(pitch) > 0.7 or max_leg_spread > 1.0:
                total_score *= 0.5  

            return max(total_score, 0.0)

        except Exception:
            pitch = abs(getattr(sim, "robot_pitch", 0))
            return 0.8 if pitch < 0.3 else 0.3
    
    def _calculate_velocity_reward(self, sim, phase_info) -> float:
        """Recompensa por velocidade consistente e eficiente"""
        current_velocity = getattr(sim, "robot_x_velocity", 0)
        target_velocity = 2.0  # ~7 km/h - objetivo realista
        
        # Recompensa velocidade próxima do alvo com baixa variação
        velocity_ratio = current_velocity / target_velocity
        if 0.8 <= velocity_ratio <= 1.2:  # ±20% do alvo
            return 1.0
        elif 0.5 <= velocity_ratio < 0.8 or 1.2 < velocity_ratio <= 1.5:
            return 0.6
        else:
            return 0.1
    
    def _calculate_gait_pattern_reward(self, sim, phase_info) -> float:
        """Recompensa por padrão de marcha natural"""
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)
        
        # Marcha alternada ideal
        if left_contact != right_contact:
            base_score = 0.8
        else:
            base_score = 0.3
        
        # Bônus por ritmo consistente
        step_consistency = 1.0 - min(getattr(sim, "gait_variability", 0.3), 1.0)
        
        return base_score * 0.7 + step_consistency * 0.3
    
    def _calculate_clearance_reward(self, sim, phase_info) -> float:
        """Recompensa por clearance automático dos pés"""
        left_height = getattr(sim, "robot_left_foot_height", 0)
        right_height = getattr(sim, "robot_right_foot_height", 0)
        left_contact = getattr(sim, "robot_left_foot_contact", False)
        right_contact = getattr(sim, "robot_right_foot_contact", False)
        
        clearance_score = 0.0
        # Recompensa pés altos quando não estão em contato
        if not left_contact and left_height > 0.05:
            clearance_score += 0.5
        if not right_contact and right_height > 0.05:
            clearance_score += 0.5
            
        return min(clearance_score, 1.0)
    
    def _calculate_dynamic_balance_reward(self, sim, phase_info) -> float:
        """Recompensa por estabilidade dinâmica durante movimento"""
        roll_vel = abs(getattr(sim, "robot_roll_vel", 0))
        pitch_vel = abs(getattr(sim, "robot_pitch_vel", 0))
        
        # Estabilidade angular durante movimento
        angular_stability = 1.0 - min((roll_vel + pitch_vel) / 3.0, 1.0)
        
        # Consistência da altura do COM
        com_height = getattr(sim, "robot_z_position", 0.8)
        height_consistency = 1.0 - min(abs(com_height - 0.8) / 0.2, 1.0)
        
        return (angular_stability * 0.6 + height_consistency * 0.4)
    
    def _calculate_smoothness_reward(self, sim, phase_info) -> float:
        """Recompensa por movimentos suaves e fluidos"""
        acceleration = abs(getattr(sim, "robot_x_acceleration", 0))
        jerk = abs(getattr(sim, "robot_jerk", 0))
        
        acceleration_smoothness = 1.0 - min(acceleration / 5.0, 1.0)
        jerk_smoothness = 1.0 - min(jerk / 20.0, 1.0)
        
        return (acceleration_smoothness * 0.7 + jerk_smoothness * 0.3)
    
    def _calculate_rhythm_reward(self, sim, phase_info) -> float:
        """Recompensa por ritmo consistente na marcha"""
        try:
            step_period = getattr(sim, "gait_step_period", 0.5)
            # Ritmo ideal ~0.5-0.6s por passo
            rhythm_quality = 1.0 - min(abs(step_period - 0.55) / 0.3, 1.0)
            
            # Consistência do comprimento da passada
            step_length_var = getattr(sim, "step_length_variability", 0.2)
            length_consistency = 1.0 - min(step_length_var / 0.3, 1.0)
            
            return (rhythm_quality * 0.6 + length_consistency * 0.4)
        except:
            return 0.5
    
    def _calculate_biomechanics_reward(self, sim, phase_info) -> float:
        """Recompensa por eficiência biomecânica"""
        distance = getattr(sim, "episode_distance", 0)
        energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
        
        # Eficiência energética
        energy_efficiency = min(distance / energy, 2.0) / 2.0
        
        # Eficiência por passada
        steps = max(getattr(sim, "episode_steps", 1), 1)
        stride_efficiency = min(distance / steps, 0.1) / 0.1
        
        return (energy_efficiency * 0.6 + stride_efficiency * 0.4)
    
    def _calculate_robustness_reward(self, sim, phase_info) -> float:
        """Recompensa por robustez da marcha"""
        recovery_events = getattr(sim, "recovery_success_count", 0)
        total_perturbations = max(getattr(sim, "total_perturbations", 1), 1)
        
        recovery_rate = recovery_events / total_perturbations
        return min(recovery_rate, 1.0)
    
    def _calculate_adaptation_reward(self, sim, phase_info) -> float:
        """Recompensa por adaptação a diferentes condições"""
        speed_adaptation = getattr(sim, "speed_adaptation_score", 0.5)
        terrain_handling = getattr(sim, "terrain_handling_score", 0.5)
        
        return (speed_adaptation * 0.6 + terrain_handling * 0.4)
    
    def _calculate_recovery_reward(self, sim, phase_info) -> float:
        """Recompensa por recuperação eficiente de perturbações"""
        recovery_time = getattr(sim, "recovery_time", 2.0)
        recovery_efficiency = 1.0 - min(recovery_time / 5.0, 1.0)
        
        return recovery_efficiency

    def _calculate_phase_angles_reward(self, sim, phase_info) -> float:
        try:
            left_knee = getattr(sim, "robot_left_knee_angle", 0)
            right_knee = getattr(sim, "robot_right_knee_angle", 0)
            ideal_extension = 0.2 
            ideal_swing = 1.2     
            left_score = 0.0
            right_score = 0.0
            left_contact = getattr(sim, "robot_left_foot_contact", False)
            right_contact = getattr(sim, "robot_right_foot_contact", False)

            if left_contact:
                left_score = np.exp(-3.0 * (left_knee - ideal_extension)**2)
            else:  
                left_score = np.exp(-2.0 * (left_knee - ideal_swing)**2)

            if right_contact:
                right_score = np.exp(-3.0 * (right_knee - ideal_extension)**2)
            else:
                right_score = np.exp(-2.0 * (right_knee - ideal_swing)**2)

            coordination_bonus = 0.0
            if left_contact != right_contact:  
                coordination_bonus = 0.3

            overflex_penalty = 0.0
            if left_knee > 1.5 or right_knee > 1.5:
                overflex_penalty = 0.2

            extension_bonus = 0.0
            if not left_contact and left_knee < 0.8:
                extension_bonus += 0.2
            if not right_contact and right_knee < 0.8:
                extension_bonus += 0.2

            base_score = (left_score + right_score) / 2.0
            final_score = base_score + coordination_bonus + extension_bonus - overflex_penalty

            return min(max(final_score, 0.0), 1.0)

        except Exception as e:
            self.logger.warning(f"Erro no cálculo de recompensa de ângulos: {e}")
            return 0.3  
    
    def _calculate_propulsion_reward(self, sim, phase_info) -> float:
        pitch = getattr(sim, "robot_pitch", 0)
        velocity = getattr(sim, "robot_x_velocity", 0)
        if pitch < -0.05 and velocity > 0.1:
            return min(abs(pitch) * velocity * 4.0, 1.0)
        return 0.0
    
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
    
    def _calculate_efficiency_reward(self, sim, phase_info) -> float:
        try:
            distance = getattr(sim, "episode_distance", 0)
            steps = max(getattr(sim, "episode_steps", 1), 1)
            energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
            steps_efficiency = distance / steps
            normalized_steps_eff = min(steps_efficiency / 0.08, 1.0)
            energy_efficiency = distance / energy
            normalized_energy_eff = min(energy_efficiency / 2.0, 1.0)
            current_velocity = abs(getattr(sim, "robot_x_velocity", 0))
            target_velocity = phase_info.get('target_speed', 1.0)
            velocity_efficiency = 1.0 - min(abs(current_velocity - target_velocity) / target_velocity, 1.0)
            com_velocity_y = abs(getattr(sim, "robot_y_velocity", 0))
            lateral_efficiency = 1.0 - min(com_velocity_y / 0.2, 1.0)
            combined_efficiency = (
                normalized_steps_eff * 0.40 +    
                normalized_energy_eff * 0.30 +   
                velocity_efficiency * 0.20 +    
                lateral_efficiency * 0.10       
            )
            if distance > 1.0 and combined_efficiency > 0.6:
                combined_efficiency = min(combined_efficiency + 0.1, 1.0)

            return min(combined_efficiency, 1.0)

        except Exception:
            try:
                distance = getattr(sim, "episode_distance", 0)
                steps = max(getattr(sim, "episode_steps", 1), 1)
                energy = max(getattr(sim, "robot_energy_used", 1.0), 0.1)
                steps_efficiency = distance / steps
                energy_efficiency = distance / energy
                return min((steps_efficiency * 0.6 + energy_efficiency * 0.4) * 2.0, 1.0)
            except:
                return 0.3
    
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
            "component_weights": {name: comp.weight for name, comp in self.components.items() if comp.enabled}
        }
        
        return status