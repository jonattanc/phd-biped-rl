# dpg_buffer.py
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class Experience:
    """Estrutura para armazenar experiências"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]
    group: int
    sub_phase: int
    quality: float
    skills: Dict[str, float]  # Habilidades demonstradas


class SkillTransferMap:
    """Mapeamento de habilidades transferíveis entre grupos"""
    
    def __init__(self):
        self.skill_transfer_rules = {
            # Fundação → Desenvolvimento
            (1, 2): {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico"],
                "skill_weights": {"estabilidade": 0.6, "controle_postural": 0.3, "progresso_basico": 0.1},
                "relevance_threshold": 0.7
            },
            # Desenvolvimento → Domínio
            (2, 3): {
                "transferable_skills": ["coordenação", "controle_velocidade", "eficiência"],
                "skill_weights": {"coordenação": 0.4, "controle_velocidade": 0.4, "eficiência": 0.2},
                "relevance_threshold": 0.8
            },
            # Regressões
            (2, 1): {
                "transferable_skills": ["estabilidade", "controle_postural"],
                "skill_weights": {"estabilidade": 0.7, "controle_postural": 0.3},
                "relevance_threshold": 0.6
            },
            (3, 2): {
                "transferable_skills": ["coordenação", "eficiência"],
                "skill_weights": {"coordenação": 0.6, "eficiência": 0.4},
                "relevance_threshold": 0.7
            }
        }
    
    def get_transfer_rules(self, old_group: int, new_group: int) -> Dict:
        """Obtém regras de transferência para transição"""
        return self.skill_transfer_rules.get((old_group, new_group), {
            "transferable_skills": [],
            "skill_weights": {},
            "relevance_threshold": 0.5
        })
    
    def calculate_skill_relevance(self, experience: Experience, target_group: int) -> float:
        """Calcula relevância da experiência para o grupo alvo"""
        rules = self.get_transfer_rules(experience.group, target_group)
        
        if not rules["transferable_skills"]:
            return 0.0
        
        relevance = 0.0
        for skill, weight in rules["skill_weights"].items():
            skill_value = experience.skills.get(skill, 0.0)
            relevance += skill_value * weight
        
        return relevance


class SmartBufferManager:
    """
    ESPECIALISTA EM MEMÓRIA com Preservação Inteligente
    """
    
    def __init__(self, logger, config, max_core_experiences=1000):
        self.logger = logger
        self.config = config
        self.max_core_experiences = max_core_experiences
        
        # Sistema de memória hierárquico
        self.group_buffers = {}
        self.core_buffer = deque(maxlen=max_core_experiences)
        self.current_group_buffer = []
        
        # Sistema de preservação
        self.skill_map = SkillTransferMap()
        self.preservation_stats = {
            "total_transitions": 0,
            "experiences_preserved": 0,
            "preservation_rate": 0.0
        }
        
        # Estatísticas
        self.experience_count = 0
        self.group_transitions = 0
    
    def store_experience(self, experience_data: Dict):
        """Armazena experiência com análise de habilidades"""
        experience = self._create_enhanced_experience(experience_data)
        
        group = experience_data.get("group_level", 1)
        sub_phase = experience_data["phase_info"].get("sub_phase", 0)
        
        # Armazenar hierarquicamente
        self._store_hierarchical(experience, group, sub_phase)
        
        # Armazenar no core se for fundamental
        if self._is_fundamental_experience(experience):
            self.core_buffer.append(experience)
        
        self.experience_count += 1
    
    def _create_enhanced_experience(self, data: Dict) -> Experience:
        """Cria experiência com análise de habilidades"""
        state = data["state"]
        action = data["action"]
        reward = data["reward"]
        phase_info = data["phase_info"]
        metrics = data["metrics"]
        
        quality = self._calculate_experience_quality(state, action, reward, metrics)
        skills = self._analyze_experience_skills(metrics, phase_info)
        
        return Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=state,
            done=False,
            info=phase_info,
            group=data.get("group_level", 1),
            sub_phase=phase_info.get("sub_phase", 0),
            quality=quality,
            skills=skills
        )
    
    def _analyze_experience_skills(self, metrics: Dict, phase_info: Dict) -> Dict[str, float]:
        """Analisa habilidades demonstradas na experiência"""
        skills = {}
        
        # Habilidade de estabilidade
        roll = metrics.get("roll", 0)
        pitch = metrics.get("pitch", 0)
        skills["estabilidade"] = 1.0 - min(abs(roll) + abs(pitch), 1.0)
        
        # Habilidade de progresso
        distance = metrics.get("distance", 0)
        skills["progresso_basico"] = min(distance / 2.0, 1.0)
        
        # Habilidade de coordenação
        left_contact = metrics.get("left_contact", False)
        right_contact = metrics.get("right_contact", False)
        skills["coordenação"] = 1.0 if left_contact != right_contact else 0.3
        
        # Habilidade de eficiência
        steps = metrics.get("steps", 1)
        skills["eficiência"] = min(distance / max(steps, 1), 1.0)
        
        # Habilidade de controle de velocidade
        speed = metrics.get("speed", 0)
        target_speed = phase_info.get('target_speed', 1.0)
        speed_error = abs(speed - target_speed)
        skills["controle_velocidade"] = 1.0 - min(speed_error / target_speed, 1.0)
        
        # Habilidade de controle postural
        skills["controle_postural"] = 1.0 - min(abs(pitch) * 2.0, 1.0)
        
        return skills
    
    def transition_with_preservation(self, old_group: int, new_group: int, adaptive_config: Dict):
        """Transição inteligente com preservação de aprendizado"""
        self.group_transitions += 1
        
        # 1. Coletar experiências do grupo antigo
        old_experiences = self.group_buffers.get(old_group, [])
        
        # 2. Filtrar experiências relevantes
        relevant_experiences = self._filter_relevant_experiences(old_experiences, new_group)
        
        # 3. Combinar com experiências fundamentais
        preserved_experiences = relevant_experiences + list(self.core_buffer)
        
        # 4. Aplicar política de preservação
        preservation_policy = adaptive_config.get("learning_preservation", "medium")
        final_experiences = self._apply_preservation_policy(preserved_experiences, preservation_policy)
        
        # 5. Atualizar buffers
        self.group_buffers[new_group] = final_experiences
        self.current_group_buffer = final_experiences
        
        # Atualizar estatísticas
        self.preservation_stats["total_transitions"] += 1
        self.preservation_stats["experiences_preserved"] += len(final_experiences)
        self.preservation_stats["preservation_rate"] = (
            self.preservation_stats["experiences_preserved"] / 
            (self.preservation_stats["total_transitions"] * 1000 + 1e-8)
        )
        
        self.logger.info(f"🔄 Preservação: {old_group}→{new_group}, "
                        f"Experiências: {len(final_experiences)}")
    
    def _filter_relevant_experiences(self, experiences: List[Experience], new_group: int) -> List[Experience]:
        """Filtra experiências relevantes para o novo grupo"""
        relevant = []
        
        for exp in experiences:
            relevance = self.skill_map.calculate_skill_relevance(exp, new_group)
            rules = self.skill_map.get_transfer_rules(exp.group, new_group)
            
            if relevance >= rules["relevance_threshold"]:
                relevant.append(exp)
        
        # Ordenar por relevância
        relevant.sort(key=lambda x: self.skill_map.calculate_skill_relevance(x, new_group), 
                     reverse=True)
        
        return relevant
    
    def _apply_preservation_policy(self, experiences: List[Experience], policy: str) -> List[Experience]:
        """Aplica política de preservação"""
        policy_limits = {
            "high": 800,    # Alta preservação
            "medium": 500,  # Preservação média
            "low": 300      # Baixa preservação
        }
        
        limit = policy_limits.get(policy, 500)
        return experiences[:limit]
    
    def _calculate_experience_quality(self, state, action, reward, metrics) -> float:
        """Calcula qualidade da experiência"""
        quality = 0.0
        
        # Fator de recompensa
        quality += min(abs(reward) * 0.2, 1.0)
        
        # Fator de progresso
        progress = metrics.get("distance", 0)
        if progress > 0:
            quality += min(progress * 2.0, 1.0)
        
        # Fator de estabilidade
        stability = 1.0 - min(metrics.get("roll", 0) + metrics.get("pitch", 0), 1.0)
        quality += stability * 0.3
        
        return min(quality, 1.0)
    
    def _is_fundamental_experience(self, experience: Experience) -> bool:
        """Verifica se experiência é fundamental"""
        return (experience.quality > 0.7 and 
                experience.reward > 0.5 and
                experience.skills.get("estabilidade", 0) > 0.6)
    
    def _store_hierarchical(self, experience: Experience, group: int, sub_phase: int):
        """Armazena experiência na hierarquia"""
        if group not in self.group_buffers:
            self.group_buffers[group] = []
        self.group_buffers[group].append(experience)
        
        self.current_group_buffer.append(experience)
        
        # Limitar tamanho
        if len(self.current_group_buffer) > 2000:
            self.current_group_buffer = self.current_group_buffer[-1500:]
    
    def get_training_batch(self, batch_size=32):
        """Retorna batch para treinamento"""
        if not self.current_group_buffer:
            return None
        
        available = self.current_group_buffer + list(self.core_buffer)
        
        if len(available) < batch_size:
            batch_size = len(available)
        
        # Amostragem por qualidade
        qualities = [exp.quality for exp in available]
        probabilities = np.array(qualities) / sum(qualities)
        
        indices = np.random.choice(len(available), size=batch_size, p=probabilities, replace=False)
        return [available[i] for i in indices]
    
    def get_status(self):
        """Retorna status com estatísticas de preservação"""
        return {
            "total_experiences": self.experience_count,
            "core_experiences": len(self.core_buffer),
            "current_group_experiences": len(self.current_group_buffer),
            "group_transitions": self.group_transitions,
            "preservation_stats": self.preservation_stats,
            "groups_with_buffer": list(self.group_buffers.keys())
        }
    
    def get_metrics(self) -> Dict:
        """Retorna métricas para monitoramento"""
        if not self.current_group_buffer:
            return {
                "buffer_avg_quality": 0,
                "buffer_avg_reward": 0,
                "core_buffer_size": len(self.core_buffer),
                "current_buffer_size": len(self.current_group_buffer),
                "learning_convergence": 0,
                "memory_efficiency": 0,
            }

        avg_quality = np.mean([exp.quality for exp in self.current_group_buffer])
        avg_reward = np.mean([exp.reward for exp in self.current_group_buffer])

        return {
            "buffer_avg_quality": avg_quality,
            "buffer_avg_reward": avg_reward,
            "core_buffer_size": len(self.core_buffer),
            "current_buffer_size": len(self.current_group_buffer),
            "learning_convergence": 0.5,  # Placeholder
            "memory_efficiency": self.preservation_stats.get("preservation_rate", 0.0),
        }