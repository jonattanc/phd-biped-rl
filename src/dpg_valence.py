# dpg_valence.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class ValenceState(Enum):
    INACTIVE = "inactive"
    LEARNING = "learning" 
    CONSOLIDATING = "consolidating"
    MASTERED = "mastered"
    REGRESSING = "regressing"


@dataclass
class ValenceConfig:
    """Configuração de uma valência individual"""
    name: str
    target_level: float
    metrics: List[str]
    reward_components: List[str]
    dependencies: List[str]
    activation_threshold: float = 0.3
    mastery_threshold: float = 0.85
    regression_threshold: float = 0.6
    max_learning_rate: float = 0.1
    min_episodes: int = 10


class ValenceTracker:
    """Rastreamento de performance por valência"""
    
    def __init__(self, valence_name: str):
        self.valence_name = valence_name
        self.current_level = 0.0
        self.history = []  
        self.learning_rate = 0.0
        self.consistency_score = 0.5
        self.episodes_active = 0
        self.state = ValenceState.INACTIVE
    
    def update_level(self, new_level: float, episode: int):
        """Atualiza nível com cálculo de taxa de aprendizado"""
        old_level = self.current_level
        self.current_level = new_level
        self.history.append((episode, new_level))
        
        # Calcular taxa de aprendizado (suavizada)
        if len(self.history) > 1:
            recent_growth = new_level - old_level
            self.learning_rate = 0.8 * self.learning_rate + 0.2 * recent_growth
        
        # Manter histórico limitado
        if len(self.history) > 100:
            self.history.pop(0)
    
    def calculate_consistency(self) -> float:
        """Calcula consistência baseada na variância recente"""
        if len(self.history) < 8:
            return 0.3
            
        recent_levels = [level for _, level in self.history[-8:]]
        variance = np.std(recent_levels)
        consistency = 1.0 - min(variance * 3.0, 1.0)
        self.consistency_score = consistency
        return consistency


class Mission:
    """Missão de curto prazo para acelerar aprendizado"""
    
    def __init__(self, valence_name: str, target_improvement: float, duration_episodes: int):
        self.valence_name = valence_name
        self.target_improvement = target_improvement
        self.duration_episodes = duration_episodes
        self.start_level = 0.0
        self.episodes_remaining = duration_episodes
        self.completed = False
        self.bonus_multiplier = 1.5
    
    def update(self, current_level: float) -> float:
        """Atualiza missão e retorna bônus se aplicável"""
        if self.completed or self.episodes_remaining <= 0:
            return 1.0
            
        self.episodes_remaining -= 1
        improvement = current_level - self.start_level
        
        if improvement >= self.target_improvement:
            self.completed = True
            return self.bonus_multiplier
        elif self.episodes_remaining <= 0:
            return 0.8  # Penalidade leve por falha
            
        return 1.0


class ValenceManager:
    """SISTEMA DE VALÊNCIAS ADAPTATIVAS"""
    
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config or {}
        
        # Sistema de valências
        self.valences = self._initialize_valences()
        self.valence_performance = {}
        self.active_valences = set()
        self.valence_weights = {}
        self.mastery_callback = None
        
        # Sistema de missões
        self.current_missions = []
        self.mission_history = []
        
        # Estado do sistema
        self.episode_count = 0
        self.overall_progress = 0.0
        self.performance_history = []
        self.irl_system = LightValenceIRL(logger)
        
        # Inicializar valence_performance
        for valence_name in self.valences.keys():
            self.valence_performance[valence_name] = ValenceTracker(valence_name)
        
        
    def _initialize_valences(self) -> Dict[str, ValenceConfig]:
        """Inicializa as valências fundamentais para locomoção bípede"""
        return {
            # VALÊNCIA FUNDAMENTAL: Estabilidade Postural
            "estabilidade_postural": ValenceConfig(
                name="estabilidade_postural",
                target_level=0.9,
                metrics=["roll", "pitch", "z_position", "stability"],
                reward_components=["stability", "posture", "basic_progress"],
                dependencies=[],
                activation_threshold=0.1,
                mastery_threshold=0.85,
                min_episodes=8
            ),
            
            # VALÊNCIA: Propulsão Básica
            "propulsao_basica": ValenceConfig(
                name="propulsao_basica", 
                target_level=0.8,
                metrics=["x_velocity", "distance", "positive_movement_rate"],
                reward_components=["velocity", "basic_progress", "direction"],
                dependencies=["estabilidade_postural"],
                activation_threshold=0.3,
                mastery_threshold=0.8,
                min_episodes=10
            ),
            
            # VALÊNCIA: Coordenação Rítmica
            "coordenacao_ritmica": ValenceConfig(
                name="coordenacao_ritmica",
                target_level=0.7,
                metrics=["gait_pattern_score", "alternating_score", "clearance_score"],
                reward_components=["coordination", "phase_angles", "clearance"],
                dependencies=["propulsao_basica"],
                activation_threshold=0.4,
                mastery_threshold=0.75,
                min_episodes=12
            ),
            
            # VALÊNCIA: Eficiência Propulsiva
            "eficiencia_propulsiva": ValenceConfig(
                name="eficiencia_propulsiva",
                target_level=0.8,
                metrics=["propulsion_efficiency", "energy_used", "speed"],
                reward_components=["efficiency", "propulsion", "velocity"],
                dependencies=["coordenacao_ritmica"],
                activation_threshold=0.5,
                mastery_threshold=0.8,
                min_episodes=15
            ),
            
            # VALÊNCIA AVANÇADA: Adaptação Dinâmica
            "adaptacao_dinamica": ValenceConfig(
                name="adaptacao_dinamica",
                target_level=0.75,
                metrics=["flight_quality", "recovery_events", "consistency"],
                reward_components=["efficiency", "coordination", "adaptation"],
                dependencies=["eficiencia_propulsiva"],
                activation_threshold=0.6,
                mastery_threshold=0.8,
                min_episodes=18
            )
        }
    
    def update_valences(self, episode_results: Dict) -> Dict[str, float]:
        """
        Atualiza todas as valências baseado nos resultados do episódio
        Retorna: pesos das valências para cálculo de recompensa
        """
        self.episode_count += 1
        self.performance_history.append(episode_results)
        self.update_irl_system(episode_results)
        
        valence_levels = {}
        for valence_name, valence_config in self.valences.items():
            level = self._calculate_valence_level(valence_name, episode_results)
            valence_levels[valence_name] = level
            
            perf = self.valence_performance[valence_name]
            perf.update_level(level, self.episode_count)
            perf.episodes_active += 1 if valence_name in self.active_valences else 0
        
        self._update_valence_states(valence_levels)
        self.valence_weights = self._calculate_valence_weights(valence_levels)
        mission_bonus = self._update_missions(valence_levels)
        self.overall_progress = self._calculate_overall_progress(valence_levels)
        
        return self.valence_weights, mission_bonus
    
    def update_irl_system(self, episode_results):
        """Atualiza sistema IRL com resultados do episódio"""
        valence_status = self.get_valence_status()
        self.irl_system.collect_demonstration(episode_results, valence_status)
        
        if self.episode_count % 50 == 0:
            new_irl_weights = self.irl_system.get_irl_weights(valence_status)
            if hasattr(self, 'irl_weights'):
                if new_irl_weights: 
                    self.irl_weights.update(new_irl_weights)
            else:
                self.irl_weights = {
                'progress': 0.3,
                'stability': 0.4, 
                'efficiency': 0.2,
                'coordination': 0.1
                }
                if new_irl_weights:  
                    self.irl_weights.update(new_irl_weights)

    def get_irl_weights(self):
        """Retorna pesos IRL atuais"""
        try:
            if hasattr(self, 'irl_weights') and self.irl_weights:
                return self.irl_weights
            else:
                self.irl_weights = {
                    'progress': 0.3,
                    'stability': 0.4, 
                    'efficiency': 0.2,
                    'coordination': 0.1
                }
                return self.irl_weights

        except Exception as e:
            self.logger.warning(f"Erro ao obter pesos IRL: {e}")
            return {'progress': 0.3, 'stability': 0.4, 'efficiency': 0.2, 'coordination': 0.1}
    
    def _calculate_valence_level(self, valence_name: str, results: Dict) -> float:
        """Calcula nível atual de uma valência específica COM PROTEÇÃO"""
        valence_config = self.valences[valence_name]
        level = 0.0
        metric_count = 0

        for metric in valence_config.metrics:
            if metric in results:
                raw_value = results[metric]
                normalized_value = self._normalize_metric(metric, raw_value)
                level += normalized_value
                metric_count += 1

        if metric_count > 0:
            level /= metric_count

        level = max(0.0, level)  

        if results.get("success", False):
            level = min(level * 1.1, 1.0)  
        elif results.get("distance", 0) > 1.0:  
            level = min(level * 1.05, 1.0) 

        if valence_name == "propulsao_basica":
            distance = results.get("distance", 0)
            speed = results.get("speed", 0)
            if level == 0.0 and (distance > 0 or speed > 0):
                level = 0.4
            elif level == 0.0 and self.episode_count > 200:
                level = 0.3

        return min(level, 1.0)
    
    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normaliza métricas para escala 0-1"""
        normalization_rules = {
            "roll": lambda x: 1.0 - min(abs(x) / 1.0, 1.0),
            "pitch": lambda x: 1.0 - min(abs(x) / 1.0, 1.0),
            "z_position": lambda x: 1.0 if x > 0.6 else x / 0.6,
            "x_velocity": lambda x: min(abs(x) / 2.0, 1.0),
            "distance": lambda x: min(x / 3.0, 1.0),
            "gait_pattern_score": lambda x: x,
            "alternating_score": lambda x: x,
            "clearance_score": lambda x: x,
            "propulsion_efficiency": lambda x: x,
            "energy_used": lambda x: 1.0 - min(x / 5.0, 1.0),
            "flight_quality": lambda x: x,
            "positive_movement_rate": lambda x: x,
            "stability": lambda x: x,
            "speed": lambda x: min(x / 1.5, 1.0),
            "consistency": lambda x: x
        }
        
        normalizer = normalization_rules.get(metric, lambda x: min(abs(x), 1.0))
        return normalizer(value)
    
    def set_mastery_callback(self, callback):
        """Define callback para quando valências atingem mastered"""
        self.mastery_callback = callback
        
    def _notify_valence_mastered(self, valence_name):
        """Notifica quando valência atinge mastered"""
        if self.mastery_callback:
            self.mastery_callback(valence_name)
            
    def _update_valence_states(self, valence_levels: Dict[str, float]):
        """Atualiza estados das valências com proteção contra regressão"""
        for valence_name, current_level in valence_levels.items():
            if valence_name == "propulsao_basica" and current_level > 0.2:
                if valence_name not in self.active_valences:
                    self.active_valences.add(valence_name)
                    self.valence_performance[valence_name].state = ValenceState.LEARNING

            perf = self.valence_performance[valence_name]
            config = self.valences[valence_name]
            old_state = perf.state

            if (old_state == ValenceState.MASTERED and 
                current_level >= config.mastery_threshold - 0.25):  
                perf.state = ValenceState.MASTERED
                self.active_valences.add(valence_name)
                continue

            if (old_state == ValenceState.MASTERED and 
                current_level < perf.current_level - 0.4):
                perf.state = ValenceState.MASTERED
                continue
        
            dependencies_met = all(
                self.valence_performance[dep].current_level >= config.activation_threshold
                for dep in config.dependencies
            )

            if not dependencies_met:
                perf.state = ValenceState.INACTIVE
                self.active_valences.discard(valence_name)
            elif current_level >= config.mastery_threshold and perf.episodes_active >= config.min_episodes:
                perf.state = ValenceState.MASTERED
                self.active_valences.add(valence_name)
            elif current_level < config.regression_threshold and old_state == ValenceState.MASTERED:
                if current_level < config.regression_threshold - 0.2: 
                    perf.state = ValenceState.REGRESSING
                    self.active_valences.add(valence_name)
                else:
                    perf.state = ValenceState.MASTERED  
            elif dependencies_met and valence_name not in self.active_valences:
                perf.state = ValenceState.LEARNING
                self.active_valences.add(valence_name)
            elif perf.state == ValenceState.LEARNING and perf.consistency_score > 0.6: 
                perf.state = ValenceState.CONSOLIDATING
            elif perf.state == ValenceState.REGRESSING and current_level >= config.mastery_threshold:
                perf.state = ValenceState.MASTERED

            if perf.state != old_state:
                if (old_state != ValenceState.MASTERED and 
                    perf.state == ValenceState.MASTERED):
                    self._notify_valence_mastered(valence_name)
    
    def _calculate_valence_weights(self, valence_levels: Dict[str, float]) -> Dict[str, float]:
        """Calcula pesos dinâmicos baseados em déficit de performance"""
        weights = {}
        total_weight = 0.0
        
        for valence_name in self.active_valences:
            config = self.valences[valence_name]
            current_level = valence_levels[valence_name]
            perf = self.valence_performance[valence_name]
            
            # Cálculo do déficit (quanto falta para o alvo)
            deficit = max(0, config.target_level - current_level)
            
            # Fator de urgência baseado no estado
            state_multiplier = {
                ValenceState.LEARNING: 2.0,
                ValenceState.REGRESSING: 1.8,
                ValenceState.CONSOLIDATING: 1.2,
                ValenceState.MASTERED: 0.3,
                ValenceState.INACTIVE: 0.0
            }.get(perf.state, 1.0)
            
            # Peso baseado no déficit e urgência
            weight = deficit * state_multiplier
            
            # Bônus para valências com baixa consistência
            if perf.consistency_score < 0.5:
                weight *= 1.5
            
            weights[valence_name] = weight
            total_weight += weight
        
        # Normalizar pesos
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _update_missions(self, valence_levels: Dict[str, float]) -> float:
        """Atualiza missões ativas e retorna bônus agregado"""
        total_bonus = 1.0
        
        # Atualizar missões existentes
        for mission in self.current_missions[:]:
            valence_level = valence_levels.get(mission.valence_name, 0.0)
            bonus = mission.update(valence_level)
            total_bonus *= bonus
            
            if mission.completed or mission.episodes_remaining <= 0:
                self.current_missions.remove(mission)
                self.mission_history.append(mission)
                
        # Gerar novas missões se necessário
        if len(self.current_missions) < 2:  
            new_mission = self._generate_mission(valence_levels)
            if new_mission:
                self.current_missions.append(new_mission)
        
        return total_bonus
    
    def _generate_mission(self, valence_levels: Dict[str, float]) -> Optional[Mission]:
        """Gera nova missão baseada nas valências mais problemáticas"""
        candidate_valences = []
        
        for valence_name in self.active_valences:
            existing_mission = any(
                mission.valence_name == valence_name 
                for mission in self.current_missions
            )
            if existing_mission:
                continue

            perf = self.valence_performance[valence_name]
            config = self.valences[valence_name]
            current_level = valence_levels[valence_name]
            
            # Apenas valências que precisam de melhoria
            if (perf.state in [ValenceState.LEARNING, ValenceState.REGRESSING] and 
                current_level < config.target_level - 0.1):
                
                deficit = config.target_level - current_level
                urgency = deficit * (2.0 if perf.state == ValenceState.REGRESSING else 1.0)
                
                candidate_valences.append((valence_name, urgency, deficit))
        
        if not candidate_valences:
            return None
        
        # Selecionar valência mais urgente
        candidate_valences.sort(key=lambda x: x[1], reverse=True)
        selected_valence, urgency, deficit = candidate_valences[0]
        
        # Definir meta realista
        target_improvement = min(deficit * 0.6, 0.3)  # 60% do déficit, máximo 0.3
        duration = max(10, min(25, int(30 / (urgency + 0.1))))  # 10-25 episódios
        
        mission = Mission(selected_valence, target_improvement, duration)
        mission.start_level = valence_levels[selected_valence]
        
        return mission
    
    def _calculate_overall_progress(self, valence_levels: Dict[str, float]) -> float:
        """Calcula progresso geral considerando todas as valências"""
        if not valence_levels:
            return 0.0
        
        total_weighted = 0.0
        total_weights = 0.0
        
        for valence_name, level in valence_levels.items():
            config = self.valences[valence_name]
            perf = self.valence_performance[valence_name]
            
            # Peso baseado na importância da valência
            weight = 1.0
            if valence_name in ["estabilidade_postural", "propulsao_basica"]:
                weight = 1.5  # Valências fundamentais têm mais peso
            
            # Progresso normalizado pelo alvo
            normalized_progress = min(level / config.target_level, 1.0)
            total_weighted += normalized_progress * weight
            total_weights += weight
        
        return total_weighted / total_weights if total_weights > 0 else 0.0
    
    def get_active_reward_components(self) -> List[str]:
        """Retorna componentes de recompensa das valências ativas"""
        components = set()
        
        for valence_name in self.active_valences:
            valence_config = self.valences[valence_name]
            components.update(valence_config.reward_components)
        
        return list(components)
    
    def get_valence_status(self) -> Dict:
        """Retorna status detalhado de todas as valências"""
        status = {
            "overall_progress": self.overall_progress,
            "episode_count": self.episode_count,
            "active_valences": list(self.active_valences),
            "current_missions": [
                {
                    "valence": mission.valence_name,
                    "progress": f"{mission.target_improvement:.2f}",
                    "episodes_remaining": mission.episodes_remaining
                }
                for mission in self.current_missions
            ],
            "valence_details": {}
        }
        
        for valence_name, perf in self.valence_performance.items():
            config = self.valences[valence_name]
            status["valence_details"][valence_name] = {
                "current_level": perf.current_level,
                "target_level": config.target_level,
                "state": perf.state.value,
                "episodes_active": perf.episodes_active,
                "learning_rate": perf.learning_rate,
                "consistency": perf.consistency_score,
                "dependencies": config.dependencies
            }
        
        return status
    
    def get_valence_weights_for_reward(self) -> Dict[str, float]:
        """Retorna pesos formatados para o sistema de recompensa"""
        # Converter pesos de valência em pesos de componentes
        component_weights = {}
        
        for valence_name, valence_weight in self.valence_weights.items():
            valence_config = self.valences[valence_name]
            
            for component in valence_config.reward_components:
                if component not in component_weights:
                    component_weights[component] = 0.0
                component_weights[component] += valence_weight
        
        # Normalizar
        total = sum(component_weights.values())
        if total > 0:
            component_weights = {k: v/total for k, v in component_weights.items()}
        
        return component_weights
    

class LightValenceIRL:
    """Sistema IRL leve integrado com valências"""
    
    def __init__(self, logger):
        self.logger = logger
        self.demonstration_buffer = []
        self.learned_weights = {}
        self.sample_count = 0
        self._active = False
        
    def should_activate(self, valence_status):
        """Ativa quando valências base estão consolidadas"""
        try:
            base_valences = ['estabilidade_postural', 'propulsao_basica']
            base_levels = []
            
            for v in base_valences:
                if v in valence_status['valence_details']:
                    level = valence_status['valence_details'][v]['current_level']
                    state = valence_status['valence_details'][v]['state']
                    if level > 0.3 or state in ['learning', 'consolidating', 'mastered']:
                        activated_count += 1
            
            should_activate = activated_count >= 1
            if should_activate and not self._active:
                self._active = True
                
            return should_activate
            
        except Exception as e:
            self.logger.warning(f"Erro ao verificar ativação IRL: {e}")
            return False
    
    def collect_demonstration(self, episode_results, valence_status):
        """Coleta demonstrações apenas de episódios de alta qualidade"""
        quality = self._calculate_demo_quality(episode_results)
        if quality > 0.5:  
            self.demonstration_buffer.append({
                'results': episode_results,
                'quality': quality,
                'valence_status': valence_status,
                'timestamp': self.sample_count
            })
            self.sample_count += 1
            
            if len(self.demonstration_buffer) > 200:
                self.demonstration_buffer.pop(0)
    
    def _calculate_demo_quality(self, results):
        """Calcula qualidade da demonstração"""
        quality = 0.0
        
        if results.get('success', False):
            quality += 0.4
        elif results.get('distance', 0) > 0.5:  
            quality += 0.3
        elif results.get('speed', 0) > 0.2: 
            quality += 0.2
            
        roll = abs(results.get('roll', 0))
        pitch = abs(results.get('pitch', 0))
        stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
        if stability > 0.6:
            quality += 0.2
            
        return min(quality, 1.0)
    
    def get_irl_weights(self, valence_status):
        """Retorna pesos IRL se disponíveis e relevantes"""
        if len(self.demonstration_buffer) < 10:
            return self.learned_weights
            
        high_quality_demos = [d for d in self.demonstration_buffer if d['quality'] > 0.6]
        if not high_quality_demos:
            return self.learned_weights
            
        new_weights = self._learn_simple_weights(high_quality_demos)
        if new_weights:
            self.learned_weights = new_weights
            
        return self.learned_weights
    
    def _learn_simple_weights(self, demonstrations):
        """Aprendizado simples de pesos IRL"""
        feature_scores = {
            'progress': 0.0,
            'stability': 0.0, 
            'efficiency': 0.0,
            'coordination': 0.0
        }
        feature_counts = {k: 0 for k in feature_scores.keys()}
        
        for demo in demonstrations:
            results = demo['results']
            
            # Progresso
            if results.get('distance', 0) > 0.5:
                feature_scores['progress'] += results['distance']
                feature_counts['progress'] += 1
                
            # Estabilidade
            roll = abs(results.get('roll', 0))
            pitch = abs(results.get('pitch', 0))
            stability = 1.0 - min((roll + pitch) / 2.0, 1.0)
            feature_scores['stability'] += stability
            feature_counts['stability'] += 1
            
            # Eficiência
            if results.get('propulsion_efficiency', 0) > 0:
                feature_scores['efficiency'] += results['propulsion_efficiency']
                feature_counts['efficiency'] += 1
                
            # Coordenação
            if results.get('alternating', False):
                feature_scores['coordination'] += 1.0
            feature_counts['coordination'] += 1
        
        # Calcular médias
        for feature in feature_scores:
            if feature_counts[feature] > 0:
                feature_scores[feature] /= feature_counts[feature]
        
        # Normalizar
        total = sum(feature_scores.values())
        if total > 0:
            return {k: v/total for k, v in feature_scores.items()}
        return {}