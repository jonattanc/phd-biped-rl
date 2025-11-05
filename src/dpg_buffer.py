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
    skills: Dict[str, float] 


class SkillTransferMap:
    """Mapeamento de habilidades transferíveis entre grupos"""
    
    def __init__(self):
        self.skill_transfer_rules = {
            # Fundação → Desenvolvimento
            (1, 2): {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico", "coordenação"],
                "skill_weights": {"estabilidade": 0.4, "controle_postural": 0.3, "progresso_basico": 0.2, "coordenação": 0.1},
                "relevance_threshold": 0.5
            },
            # Desenvolvimento → Domínio
            (2, 3): {
                "transferable_skills": ["coordenação", "controle_velocidade", "eficiência", "estabilidade"],
                "skill_weights": {"coordenação": 0.3, "controle_velocidade": 0.3, "eficiência": 0.2, "estabilidade": 0.1},
                "relevance_threshold": 0.6
            },
            # Regressões
            (2, 1): {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico"],
                "skill_weights": {"estabilidade": 0.5, "controle_postural": 0.3, "progresso_basico": 0.2},
                "relevance_threshold": 0.5 
            },
            (3, 2): {
                "transferable_skills": ["coordenação", "eficiência", "estabilidade"],
                "skill_weights": {"coordenação": 0.4, "eficiência": 0.3, "estabilidade": 0.3},
                "relevance_threshold": 0.6 
            }
        }
    
    def get_transfer_rules(self, old_group: int, new_group: int) -> Dict:
        """Obtém regras de transferência para transição"""
        if old_group == new_group:
            return {
                "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico", "coordenação", "eficiência", "controle_velocidade"],
                "skill_weights": {
                    "estabilidade": 0.2, 
                    "controle_postural": 0.2, 
                    "progresso_basico": 0.2, 
                    "coordenação": 0.15, 
                    "eficiência": 0.15,
                    "controle_velocidade": 0.1
                },
                "relevance_threshold": 0.3  
            }
        
        return self.skill_transfer_rules.get((old_group, new_group), {
            "transferable_skills": ["estabilidade", "controle_postural", "progresso_basico"],
            "skill_weights": {"estabilidade": 0.4, "controle_postural": 0.4, "progresso_basico": 0.2},
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
        
        if experience.quality > 0.8:
            relevance *= 1.3  
        elif experience.quality > 0.6:
            relevance *= 1.1  
            
        return min(relevance, 1.0)


class SmartBufferManager:
    """
    ESPECIALISTA EM MEMÓRIA com Preservação Inteligente
    """
    
    def __init__(self, logger, config, max_core_experiences=2000):
        self.logger = logger
        self.config = config
        self.max_core_experiences = max_core_experiences
        
        # Sistema de memória hierárquico
        self.group_buffers = {0: []}
        self.core_buffer = deque(maxlen=max_core_experiences)
        self.current_group_buffer = []
        
        # Sistema de preservação
        self.skill_map = SkillTransferMap()
        self.preservation_stats = {
            "total_transitions": 0,
            "experiences_preserved": 0,
            "preservation_rate": 0.0
        }
        
        # Otimizações
        self._last_sort_episode = 0
        self.sort_interval = 100  # Ordenar a cada 100 episódios
        self._high_quality_cache = None
        self._cache_valid = False
    
        # Estatísticas
        self.experience_count = 0
        self.group_transitions = 0
    
    def store_experience(self, experience_data: Dict):
        """Armazena experiência com análise de habilidades"""
        try:
            if not experience_data:
                return
            if "state" in experience_data and hasattr(experience_data["state"], 'tolist'):
                experience_data["state"] = experience_data["state"].tolist()
            if "action" in experience_data and hasattr(experience_data["action"], 'tolist'):
                experience_data["action"] = experience_data["action"].tolist()
            if "reward" in experience_data:
                experience_data["reward"] = float(experience_data["reward"])
            phase_info = experience_data.get("phase_info", {})
            current_group = 1
            dpg_manager = getattr(self, '_dpg_manager', None)
            if dpg_manager and hasattr(dpg_manager, 'current_group'):
                current_group = dpg_manager.current_group
            elif hasattr(self, 'current_group_buffer') and self.current_group_buffer:
                current_group = self.get_current_group()

            phase_info['group_level'] = current_group
            phase_info['group'] = current_group
            experience = self._create_enhanced_experience(experience_data)
            experience.quality = max(0.0, experience.quality)
            self._store_hierarchical(experience, current_group, phase_info.get('sub_phase', 0))

            if self._is_fundamental_experience(experience) or experience.quality > 0.5:
                if len(self.core_buffer) < self.max_core_experiences:
                    self.core_buffer.append(experience)
                else:
                    min_quality_exp = min(self.core_buffer, key=lambda x: x.quality)
                    if experience.quality > min_quality_exp.quality:
                        self.core_buffer.remove(min_quality_exp)
                        self.core_buffer.append(experience)

            self.experience_count += 1

        except Exception as e:
            self.logger.warning(f"Erro ao armazenar experiência: {e}")
    
    def get_current_group(self) -> int:
        """Retorna o grupo atual baseado no buffer atual"""
        if not hasattr(self, 'current_group_buffer') or self.current_group_buffer is None:
            return 1  

        for group, buffer in self.group_buffers.items():
            if buffer and len(buffer) > 0 and buffer is self.current_group_buffer:
                return group

        return getattr(self, '_last_known_group', 1)

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
        """Transição inteligente com preservação de aprendizado """
        self.group_transitions += 1

        # Garantir que ambos os grupos existem
        if old_group not in self.group_buffers:
            self.group_buffers[old_group] = []

        if new_group not in self.group_buffers:
            self.group_buffers[new_group] = []

        # 1. Coletar experiências do grupo antigo
        old_experiences = self.group_buffers.get(old_group, [])

        if old_group == new_group:
            preserved_experiences = old_experiences + list(self.core_buffer)
        else:
            # 2. Filtrar experiências relevantes
            relevant_experiences = self._filter_relevant_experiences(old_experiences, new_group)

            # 3. Combinar com experiências fundamentais
            preserved_experiences = relevant_experiences + list(self.core_buffer)

        # 4. Aplicar política de preservação
        preservation_policy = adaptive_config.get("learning_preservation", "high")
        final_experiences = self._apply_preservation_policy(preserved_experiences, preservation_policy)

        # 5. Atualizar buffers
        self.group_buffers[new_group] = final_experiences
        self.current_group_buffer = final_experiences

        # Atualizar estatísticas
        self.preservation_stats["total_transitions"] += 1
        self.preservation_stats["experiences_preserved"] += len(final_experiences)

        # Calcular taxa de preservação
        if self.preservation_stats["total_transitions"] > 0:
            total_preserved_all = sum(len(buf) for buf in self.group_buffers.values())
            total_possible = self.experience_count
            if total_possible > 0:
                self.preservation_stats["preservation_rate"] = total_preserved_all / total_possible
    
    def _filter_relevant_experiences(self, experiences: List[Experience], new_group: int) -> List[Experience]:
        """Filtra experiências relevantes para o novo grupo"""
        relevant = []
        
        for exp in experiences:
            relevance = self.skill_map.calculate_skill_relevance(exp, new_group)
            rules = self.skill_map.get_transfer_rules(exp.group, new_group)
            
            if relevance >= rules["relevance_threshold"] or exp.quality > 0.8:
                relevant.append(exp)
        
        relevant.sort(key=lambda x: self.skill_map.calculate_skill_relevance(x, new_group) * 0.7 + x.quality * 0.3, reverse=True)
        
        return relevant
    
    def _apply_preservation_policy(self, experiences: List[Experience], policy: str) -> List[Experience]:
        """Aplica política de preservação"""
        policy_limits = {
            "high": 3000,    
            "medium": 2000,  
            "low": 1500     
        }
        
        limit = policy_limits.get(policy, 2000)
        if len(experiences) <= limit:
            return experiences
        high_quality = [exp for exp in experiences if exp.quality > 0.7]
        medium_quality = [exp for exp in experiences if 0.4 <= exp.quality <= 0.7]
        low_quality = [exp for exp in experiences if exp.quality < 0.4]
        
        preserved = []
        
        high_limit = int(limit * 0.7)
        if len(high_quality) > high_limit:
            high_quality.sort(key=lambda x: x.quality, reverse=True)
            preserved.extend(high_quality[:high_limit])
        else:
            preserved.extend(high_quality)
        
        remaining_slots = limit - len(preserved)
        medium_limit = int(remaining_slots * 0.9)  
        
        if len(medium_quality) > medium_limit:
            medium_quality.sort(key=lambda x: x.quality, reverse=True)
            preserved.extend(medium_quality[:medium_limit])
        else:
            preserved.extend(medium_quality)
        
        remaining_slots = limit - len(preserved)
        if remaining_slots > 0 and low_quality:
            low_quality.sort(key=lambda x: x.reward, reverse=True)
            preserved.extend(low_quality[:remaining_slots])
        
        return preserved
    
    def _calculate_experience_quality(self, state, action, reward, metrics) -> float:
        """Calcula qualidade da experiência"""
        quality = 0.0
        
        # Fator de recompensa
        quality += min(abs(reward) * 0.3, 1.0)
        
        # Fator de progresso
        progress = metrics.get("distance", 0)
        if progress > 0:
            quality += min(progress * 2.0, 1.0)
        
        # Fator de estabilidade
        stability = 1.0 - min(metrics.get("roll", 0) + metrics.get("pitch", 0), 1.0)
        quality += stability * 0.4

        # Fator de sucesso
        success = metrics.get("success", False)
        if success:
            quality += 0.5
        
        return min(quality, 1.0)
    
    def _is_fundamental_experience(self, experience: Experience) -> bool:
        """Verifica se experiência é fundamental"""
        return (experience.quality > 0.6 and 
                experience.reward > 0.3 and
                experience.skills.get("estabilidade", 0) > 0.5)
    
    def _store_hierarchical(self, experience: Experience, group: int, sub_phase: int):
        """Armazena experiência na hierarquia"""
        if group not in self.group_buffers:
            self.group_buffers[group] = []
            self.logger.info(f"📁 Criado novo grupo {group} no buffer")

        self.group_buffers[group].append(experience)
        self.current_group_buffer = self.group_buffers[group]

        if len(self.group_buffers[group]) > 3000:
            self.group_buffers[group].sort(key=lambda x: x.quality, reverse=True)
            self.group_buffers[group] = self.group_buffers[group][:2500]
            self.current_group_buffer = self.group_buffers[group]
    
    def get_training_batch(self, batch_size=32):
        """Retorna batch para treinamento - VERSÃO ESTABILIZADA"""
        if not self.current_group_buffer:
            return None

        available = self.current_group_buffer + list(self.core_buffer)

        if len(available) < batch_size:
            batch_size = len(available)

        scored_experiences = []
        for exp in available:
            stability_score = exp.skills.get("estabilidade", 0)
            progress_score = exp.skills.get("progresso_basico", 0)
            stability_penalty = 0.0
            if stability_score < 0.4:  
                stability_penalty = 0.5
            balance_bonus = 1.0
            if stability_score > 0.6 and progress_score > 0.4:
                balance_bonus = 1.3
            final_score = exp.quality * stability_penalty
            scored_experiences.append((exp, max(final_score, 0.1)))

        scored_experiences.sort(key=lambda x: x[1], reverse=True)

        stable_experiences = [exp for exp, score in scored_experiences if exp.skills.get("estabilidade", 0) > 0.5]
        diverse_experiences = [exp for exp, score in scored_experiences if exp.skills.get("estabilidade", 0) <= 0.5]

        stable_count = min(len(stable_experiences), int(batch_size * 0.7))
        diverse_count = batch_size - stable_count

        if stable_count > 0:
            selected = stable_experiences[:stable_count]
        else:
            selected = []

        if diverse_count > 0 and len(diverse_experiences) > 0:
            selected.extend(diverse_experiences[:diverse_count])

        return selected
    
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
            "learning_convergence": 0.5,  
            "memory_efficiency": self.preservation_stats.get("preservation_rate", 0.0),
        }