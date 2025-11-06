# dpg_buffer.py
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import heapq
from collections import deque
from functools import lru_cache
import time

from dpg_valence import ValenceState

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
    
    def __lt__(self, other):
        """Define comparação para heapq baseado na qualidade"""
        return self.quality < other.quality
    
    def __eq__(self, other):
        """Define igualdade para heapq"""
        if not isinstance(other, Experience):
            return False
        return self.quality == other.quality

class QualityCache:
    """Cache inteligente para cálculos de qualidade"""
    
    def __init__(self, max_size=500):
        self._cache = {}
        self._max_size = max_size
        self._access_count = {}
    
    def get_quality(self, state_fingerprint: str, metrics: Dict) -> float:
        """Obtém qualidade do cache se disponível"""
        cache_key = self._generate_quality_key(state_fingerprint, metrics)
        if cache_key in self._cache:
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._cache[cache_key]
        return None
    
    def set_quality(self, state_fingerprint: str, metrics: Dict, quality: float):
        """Armazena qualidade no cache"""
        cache_key = self._generate_quality_key(state_fingerprint, metrics)
        
        if len(self._cache) >= self._max_size:
            self._evict_least_used()
        
        self._cache[cache_key] = quality
        self._access_count[cache_key] = 1
    
    def _generate_quality_key(self, state_fp: str, metrics: Dict) -> str:
        """Gera chave única para cache"""
        essential_metrics = {
            'distance': metrics.get('distance', 0),
            'speed': metrics.get('speed', 0),
            'roll': metrics.get('roll', 0),
            'pitch': metrics.get('pitch', 0),
            'success': metrics.get('success', False)
        }
        return f"{state_fp}_{str(essential_metrics)}"
    
    def _evict_least_used(self):
        """Remove entradas menos usadas do cache"""
        if not self._cache:
            return
        min_key = min(self._access_count.keys(), key=lambda k: self._access_count[k])
        del self._cache[min_key]
        del self._access_count[min_key]

class StateCompressor:
    """Compressor eficiente de estados para economizar memória"""
    
    def __init__(self):
        self._mean = None
        self._std = None
        self._is_trained = False
        
    def compress_state(self, state: np.ndarray) -> np.ndarray:
        """Compressão simplificada do estado"""
        if len(state) > 15:  # Apenas comprime estados grandes
            # Redução dimensional simples - pega características principais
            if len(state) >= 10:
                # Mantém primeiras 8 dimensões + últimas 2 (normalmente mais importantes)
                compressed = np.concatenate([state[:8], state[-2:]])
                return compressed
            else:
                return state[:8]  # Fallback
        return state
    
    def train_compressor(self, states: List[np.ndarray]):
        """Treina compressor - placeholder para implementação futura"""
        if len(states) < 50:
            return
        # Em uma implementação completa, aqui viria PCA incremental
        self._is_trained = True

class PrioritizedBuffer:
    """Buffer com amostragem prioritária"""
    
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self._quality_heap = []  # Heap para rápida recuperação das melhores
        self.pos = 0
        
    def add(self, experience: Experience, priority: float = None) -> bool:
        """Adiciona experiência com prioridade - CORRIGIDO para retornar sucesso"""
        if priority is None:
            priority = self._calculate_priority(experience)
            
        try:
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.priorities.append(priority)
            else:
                # Encontra a experiência com menor prioridade
                min_priority = min(self.priorities)
                if priority > min_priority:
                    min_idx = self.priorities.index(min_priority)
                    self.buffer[min_idx] = experience
                    self.priorities[min_idx] = priority
                else:
                    return False  
            
            # Mantém heap de qualidade atualizado
            heapq.heappush(self._quality_heap, (-experience.quality, experience))
            if len(self._quality_heap) > 1000:
                heapq.heappop(self._quality_heap)
                
            return True  
            
        except Exception as e:
            return False
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Amostra experiências baseado em prioridade"""
        if not self.buffer:
            return []
            
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        
        # Amostragem por prioridade
        priorities = np.array(self.priorities) + 1e-5  # Evita divisão por zero
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]
    
    def get_high_quality(self, count: int) -> List[Experience]:
        """Recupera experiências de alta qualidade rapidamente"""
        if not self._quality_heap:
            return self.sample(count)
            
        return [exp for _, exp in heapq.nlargest(count, self._quality_heap)]
    
    def _calculate_priority(self, experience: Experience) -> float:
        """Calcula prioridade baseada em múltiplos fatores"""
        return (experience.quality * 0.6 + 
                min(experience.reward * 0.1, 0.3) + 
                sum(experience.skills.values()) * 0.1)

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
        """Calcula relevância da experiência para o grupo alvo - CORRIGIDO"""
        rules = self.get_transfer_rules(experience.group, target_group)

        if not rules["transferable_skills"]:
            return 0.0

        relevance = 0.0
        total_weight = 0.0

        # Itera sobre skills transferíveis
        for skill in rules["transferable_skills"]:
            weight = rules["skill_weights"].get(skill, 0.0)
            skill_value = experience.skills.get(skill, 0.0)
            relevance += skill_value * weight
            total_weight += weight

        # Normaliza pela soma dos pesos
        if total_weight > 0:
            relevance /= total_weight

        # Bônus por qualidade
        if experience.quality > 0.8:
            relevance *= 1.3  
        elif experience.quality > 0.6:
            relevance *= 1.1  

        # Bônus adicional para transições entre grupos próximos
        group_diff = abs(experience.group - target_group)
        if group_diff == 1:  
            relevance *= 1.2

        return min(relevance, 1.0)

class OptimizedBufferManager:
    """BUFFER DE ALTA PERFORMANCE - Versão otimizada"""
    
    def __init__(self, logger, config, max_experiences=5000):
        self.logger = logger
        self.config = config
        self.max_experiences = max_experiences
        
        # Sistema de buffers otimizado
        self.group_buffers = {}
        for group in [1, 2, 3]:
            self.group_buffers[group] = PrioritizedBuffer(capacity=1500)
        self.core_buffer = PrioritizedBuffer(capacity=1000)
        self.current_group = 1
        if self.current_group in self.group_buffers:
            self.current_group_buffer = self.group_buffers[self.current_group].buffer
        else:
            self.current_group_buffer = []
        
        # Sistemas de otimização
        self.quality_cache = QualityCache(max_size=1000)
        self.state_compressor = StateCompressor()
        self.skill_map = SkillTransferMap()
        
        # Estatísticas
        self.episode_count = 0
        self.stored_count = 0
        self.rejected_count = 0
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_savings": 0,
            "avg_quality_stored": 0.0
        }
        
        # Controles adaptativos
        self._last_cleanup = 0
        self.cleanup_interval = 200
        self._dpg_manager = None
        
    def store_experience(self, experience_data: Dict):
        """Armazenamento OTIMIZADO com cache e compressão"""
        try:
            if not experience_data:
                return

            # Verificação rápida de qualidade básica
            metrics = experience_data.get("metrics", {})
            distance = metrics.get("distance", 0)
            speed = metrics.get("speed", 0)
            reward = experience_data.get("reward", 0)

            # CALCULA QUALIDADE COM DEBUG
            quality = self._calculate_quality_with_debug(experience_data)

            # CORREÇÃO RADICAL: Armazena TUDO para debug
            current_group = self.get_current_group()
            experience_data["group_level"] = current_group
            experience = self._create_compressed_experience(experience_data)
            experience.quality = quality

            # ARMAZENA SEM CRITÉRIOS
            self._store_without_criteria(experience, current_group)
            self.stored_count += 1
            self.episode_count += 1

        except Exception as e:
            self.logger.error(f"❌❌ ERRO CRÍTICO no armazenamento: {e}")

    def _calculate_quality_with_debug(self, data: Dict) -> float:
        """Cálculo de qualidade com DEBUG COMPULSIVO"""
        try:
            metrics = data.get("metrics", {})
            reward = data.get("reward", 0)

            distance = float(metrics.get("distance", 0))
            speed = float(metrics.get("speed", 0))
            success = bool(metrics.get("success", False))

            # CÁLCULO SIMPLES E DIRETO
            quality = 0.0

            # 1. DISTÂNCIA (80% do peso)
            if distance > 0:
                distance_component = min(distance / 2.0, 1.0) * 0.8
                quality += distance_component

            # 2. VELOCIDADE (20% do peso)  
            if speed > 0:
                speed_component = min(speed / 1.5, 1.0) * 0.2
                quality += speed_component

            # 3. BÔNUS AGRESSIVO
            if distance > 1.0:
                quality = min(quality + 0.3, 1.0)
            elif distance > 0.5:
                quality = min(quality + 0.15, 1.0)
            if success:
                quality = 1.0

            # GARANTIA: Qualidade mínima para movimento
            if distance > 0.1 and quality == 0:
                quality = 0.1
                self.logger.info(f"   🛡️  Garantia mínima: 0.10")

            return quality

        except Exception as e:
            self.logger.error(f"❌ ERRO no cálculo de qualidade: {e}")
            return 0.0

    def _store_without_criteria(self, experience: Experience, group: int):
        """Armazenamento SEM CRITÉRIOS - apenas para debug"""
        try:
            # Armazena em TODOS os buffers
            priority = 1.0  # Prioridade máxima

            self.group_buffers[group].add(experience, priority)
            self.current_group_buffer = self.group_buffers[group].buffer

            # Também no core buffer
            self.core_buffer.add(experience, priority)

        except Exception as e:
            self.logger.error(f"❌ ERRO no armazenamento: {e}")
    
    def _create_compressed_experience(self, data: Dict) -> Experience:
        """Cria experiência com estado comprimido"""
        try:
            # Garante que estados são numpy arrays
            state = np.array(data["state"], dtype=np.float32)
            action = np.array(data["action"], dtype=np.float32)

            compressed_state = self.state_compressor.compress_state(state)
            compressed_next = self.state_compressor.compress_state(
                data.get("next_state", data["state"])
            )

            # Calcula qualidade ANTES de criar a experiência
            quality = self._calculate_quality(data)

            experience = Experience(
                state=compressed_state,
                action=action,
                reward=float(data["reward"]),
                next_state=compressed_next,
                done=False,
                info=data.get("phase_info", {}),
                group=data.get("group_level", 1),
                sub_phase=0,
                quality=quality,  # JÁ CALCULADA
                skills=self._analyze_skills(data.get("metrics", {}))
            )

            return experience

        except Exception as e:
            self.logger.error(f"❌ ERRO na criação de experiência: {e}")
            return Experience(
                state=np.zeros(10),
                action=np.zeros(6),
                reward=0,
                next_state=np.zeros(10),
                done=False,
                info={},
                group=1,
                sub_phase=0,
                quality=0.0,
                skills={}
            )
    
    def _should_store(self, experience: Experience) -> bool:
        """Critérios INTELIGENTES de armazenamento"""
        try:
            metrics = experience.info.get("metrics", {})
            distance = metrics.get("distance", 0)

            if distance > 0.1:
                return True
            if distance > 0.05 and experience.quality > 0.3:
                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ ERRO nos critérios de armazenamento: {e}")
            return True

    def _is_fundamental_skill(self, experience: Experience) -> bool:
        """Habilidades fundamentais CORRIGIDAS"""
        metrics = experience.info.get("metrics", {})
        distance = metrics.get("distance", 0)

        # Foco em movimento POSITIVO com estabilidade
        return (distance > 0.2 and 
                experience.skills.get("estabilidade", 0) > 0.5 and
                experience.skills.get("progresso_basico", 0) > 0.3)
    
    def _is_novel_experience(self, experience: Experience) -> bool:
        """Verifica se experiência é nova/diversa"""
        # Implementação simplificada - verifica similaridade com recentes
        recent_experiences = self._get_recent_experiences(20)
        if not recent_experiences:
            return True
            
        similarities = []
        for exp in recent_experiences:
            exp_state = np.array(exp.state)
            current_state = np.array(experience.state)
            sim = np.linalg.norm(current_state - exp_state)
            similarities.append(sim)
            
        avg_similarity = np.mean(similarities) if similarities else 0
        return avg_similarity > 0.3  # Considera nova se for diferente o suficiente
    
    def _get_recent_experiences(self, count: int) -> List[Experience]:
        """Obtém experiências recentes para análise de novidade"""
        current_buffer = self.group_buffers.get(self.current_group)
    
        # Verifica se o buffer existe e tem experiências
        if not current_buffer or not current_buffer.buffer:
            return []

        if len(current_buffer.buffer) < count:
            return current_buffer.buffer.copy()  # retorna cópia

        return current_buffer.buffer[-count:].copy() 
    
    def _store_optimized(self, experience: Experience, group: int):
        """Armazenamento com priorização inteligente"""
        # Calcula prioridade multifatorial
        priority = self._calculate_experience_priority(experience)
        
        # Armazena no buffer do grupo
        self.group_buffers[group].add(experience, priority)
        self.current_group_buffer = self.group_buffers[group].buffer
        
        # Se for excepcional, vai para core buffer
        if experience.quality > 0.8 or self._is_core_experience(experience):
            self.core_buffer.add(experience, priority * 1.2)
    
    def _calculate_experience_priority(self, experience: Experience) -> float:
        """Calcula prioridade baseada em múltiplos fatores"""
        quality_factor = experience.quality * 0.5
        reward_factor = min(experience.reward * 0.2, 0.3)
        skill_factor = sum(experience.skills.values()) * 0.15
        novelty_factor = self._calculate_novelty(experience) * 0.15
        
        return quality_factor + reward_factor + skill_factor + novelty_factor
    
    def _calculate_novelty(self, experience: Experience) -> float:
        """Calcula fator de novidade"""
        recent = self._get_recent_experiences(10)
        if not recent:
            return 1.0
            
        similarities = []
        for exp in recent:
            exp_state = np.array(exp.state)
            current_state = np.array(experience.state)
            sim = np.linalg.norm(current_state - exp_state)
            similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0
        return min(avg_similarity, 1.0)  # Mais similar = menos novidade
    
    def _is_core_experience(self, experience: Experience) -> bool:
        """Verifica se experiência deve ir para core buffer"""
        return (experience.quality > 0.7 and 
                experience.reward > 2.0 and
                experience.skills.get("estabilidade", 0) > 0.7)
    
    def _calculate_quality(self, data: Dict) -> float:
        """Calcula qualidade com FOCO EM MOVIMENTO POSITIVO"""
        try:
            metrics = data.get("metrics", {})
            reward = data.get("reward", 0)

            # Obtém valores de forma SEGURA
            distance = float(metrics.get("distance", 0))
            speed = float(metrics.get("speed", 0))
            success = bool(metrics.get("success", False))
            roll = abs(metrics.get("roll", 0))
            pitch = abs(metrics.get("pitch", 0))

            # Cálculo DIRETO e SIMPLES
            quality = 0.0

            # 1. Componente de DISTÂNCIA (50%)
            if distance <= 0:
                quality += 0.01
            if distance > 3.0: quality += 1.0
            if distance > 2.0: quality += 0.8
            if distance > 1.5: quality += 0.7
            if distance > 1.0: quality += 0.6
            if distance > 0.5: quality += 0.4
            if distance > 0.2: quality += 0.3
            if distance > 0.1: quality += 0.2
            if distance > 0.05: quality += 0.1

            # 2. Componente de ESTABILIDADE (30%)
            stability = 1.0 - min((roll + pitch) / 2.0, 1.0)  # Média de roll e pitch, normalizada para [0,1]
            stability_component = stability * 0.3
            quality += stability_component

            # 3. Componente de VELOCIDADE (20%)
            if speed > 0:
                speed_component = min(speed / 1.5, 1.0) * 0.2
                quality += speed_component

            # Bônus por movimento real com estabilidade
            if distance > 1.0 and stability > 0.7:
                quality = min(quality + 0.2, 1.0)  # Bônus fixo
            elif distance > 0.5 and stability > 0.5:
                quality = min(quality + 0.1, 1.0)

            if success:
                quality = 1.0

            # Garante que qualidade nunca seja 0 se há movimento
            if distance > 0.1 and quality == 0:
                quality = 0.1  # Mínimo garantido

            return float(quality)

        except Exception as e:
            self.logger.error(f"❌ ERRO CRÍTICO no cálculo de qualidade: {e}")
            return 0.0
    
    def _generate_state_fingerprint(self, state: np.ndarray) -> str:
        """Gera fingerprint rápido do estado para cache"""
        try:
            if len(state) > 5:
                # Usa primeiras 3 e últimas 2 dimensões para fingerprint
                essential = np.concatenate([state[:3], state[-2:]])
                return "_".join(f"{x:.2f}" for x in essential)
            return "_".join(f"{x:.2f}" for x in state[:5])
        except:
            return "unknown"
    
    def _analyze_skills(self, metrics: Dict) -> Dict[str, float]:
        """Análise simplificada de habilidades """
        distance = metrics.get("distance", 0)
        speed = metrics.get("speed", 0)
        roll = abs(metrics.get("roll", 0))
        pitch = abs(metrics.get("pitch", 0))

        # Foco ABSOLUTO em movimento positivo
        movimento_positivo = 0.0
        if distance > 0:
            movimento_positivo = min(distance / 1.5, 1.0)  # Meta realista

        # Estabilidade baseada em thresholds REALISTAS
        estabilidade = 1.0 - min((roll + pitch) / 0.8, 1.0)  # Mais tolerante

        # Progresso baseado em movimento REAL
        progresso_basico = 1.0 if distance > 0.3 else min(distance / 0.3, 0.5)

        return {
            "movimento_positivo": movimento_positivo,
            "velocidade_eficiente": min(speed / 0.8, 1.0) if speed > 0 else 0.0,
            "estabilidade": estabilidade,
            "progresso_basico": progresso_basico,
            "coordenação": 0.7 if metrics.get("alternating", False) else 0.3,
            "controle_postural": 1.0 - min(pitch * 2.0, 1.0),
        }
    
    def get_training_batch(self, batch_size=32):
        """Retorna batch OTIMIZADO para treinamento"""
        if not self.current_group_buffer:
            return None

        # Sampling inteligente com múltiplas fontes
        batch = []
        
        # 70% do grupo atual (prioritário)
        group_samples = self.group_buffers[self.current_group].sample(
            int(batch_size * 0.7)
        )
        batch.extend(group_samples)
        
        # 20% do core buffer (alta qualidade)
        core_samples = self.core_buffer.sample(int(batch_size * 0.2))
        batch.extend(core_samples)
        
        # 10% de transferência entre grupos
        transfer_samples = self._get_transfer_experiences(int(batch_size * 0.1))
        batch.extend(transfer_samples)
        
        # Garante tamanho e qualidade mínima
        final_batch = [exp for exp in batch if exp.quality > 0.2]
        
        if len(final_batch) < batch_size // 2:
            # Fallback: pega melhores disponíveis
            high_quality = self.group_buffers[self.current_group].get_high_quality(batch_size)
            return high_quality[:batch_size]
        
        return final_batch[:batch_size]
    
    def _get_transfer_experiences(self, count: int) -> List[Experience]:
        """Obtém experiências transferíveis de outros grupos"""
        transfer_experiences = []
        
        for group_id, buffer in self.group_buffers.items():
            if group_id == self.current_group:
                continue
                
            # Pega experiências relevantes para grupo atual
            for exp in buffer.buffer[:100]:  # Amostra das melhores
                relevance = self.skill_map.calculate_skill_relevance(exp, self.current_group)
                if relevance > 0.6 and exp.quality > 0.5:
                    transfer_experiences.append(exp)
                    if len(transfer_experiences) >= count:
                        return transfer_experiences
        
        return transfer_experiences[:count]
    
    def _cleanup_low_quality(self):
        """Limpeza periódica de experiências de baixa qualidade"""
        for group_id, buffer in self.group_buffers.items():
            if len(buffer.buffer) > 1500:
                # Mantém apenas as top 80% por qualidade
                buffer.buffer.sort(key=lambda x: x.quality, reverse=True)
                buffer.buffer = buffer.buffer[:1200]
                buffer.priorities = buffer.priorities[:1200]
        
        # Limpa core buffer também
        if len(self.core_buffer.buffer) > 800:
            self.core_buffer.buffer.sort(key=lambda x: x.quality, reverse=True)
            self.core_buffer.buffer = self.core_buffer.buffer[:600]
            self.core_buffer.priorities = self.core_buffer.priorities[:600]
        
        self.logger.debug("✅ Limpeza de buffer concluída")
    
    def get_current_group(self) -> int:
        """Retorna grupo atual de forma robusta"""
        if hasattr(self, 'current_group'):
            return self.current_group

        if hasattr(self, '_dpg_manager') and self._dpg_manager:
            return getattr(self._dpg_manager, 'current_group', 1)

        return 1 

    def transition_with_preservation(self, old_group: int, new_group: int, adaptive_config: Dict):
        """Transição OTIMIZADA com preservação inteligente"""
        self.current_group = new_group

        # Garante que o buffer do grupo existe
        if new_group not in self.group_buffers:
            self.group_buffers[new_group] = PrioritizedBuffer(capacity=1500)

        self.current_group_buffer = self.group_buffers[new_group].buffer

        if old_group == new_group:
            return

        # Verifica se o grupo antigo existe e tem experiências
        if old_group not in self.group_buffers:
            self.logger.warning(f"Grupo antigo {old_group} não encontrado nos buffers")
            return

        old_buffer = self.group_buffers[old_group]

        # Verifica se há experiências no buffer antigo
        if not old_buffer.buffer:
            self.logger.info(f"ℹ️  Buffer do grupo {old_group} vazio - nada para preservar")
            return

        # Preserva experiências relevantes do grupo antigo
        relevant_experiences = []

        for exp in old_buffer.buffer:
            metrics = exp.info.get("metrics", {})
            distance = metrics.get("distance", 0)

            # PRIORIDADE: Experiências com movimento significativo
            movement_bonus = min(distance / 1.0, 1.0) 

            relevance = self.skill_map.calculate_skill_relevance(exp, new_group)
            adjusted_relevance = relevance * (1.0 + movement_bonus * 0.5)

            if adjusted_relevance > 0.3 or exp.quality > 0.5:
                relevant_experiences.append((exp, adjusted_relevance))

        # Ordena por relevância + qualidade
        relevant_experiences.sort(key=lambda x: (x[1] * 0.7 + x[0].quality * 0.3), reverse=True)

        # Limita o número de experiências transferidas
        transfer_limit = min(len(relevant_experiences), 200)  

        preserved_count = 0
        for exp, relevance in relevant_experiences[:transfer_limit]:
            priority = self._calculate_experience_priority(exp) * (1.0 + relevance)
            success = self.group_buffers[new_group].add(exp, priority)
            if success:
                preserved_count += 1

    def get_status(self):
        """Status OTIMIZADO com métricas de eficiência"""
        try:
            # Calcula qualidade REAL de todas as experiências
            all_experiences = []

            for group_id, buffer in self.group_buffers.items():
                all_experiences.extend(buffer.buffer)

            all_experiences.extend(self.core_buffer.buffer)

            total_quality = 0.0
            total_distance = 0.0
            count = len(all_experiences)

            if count > 0:
                for exp in all_experiences:
                    total_quality += exp.quality
                    metrics = exp.info.get("metrics", {})
                    total_distance += metrics.get("distance", 0)

                avg_quality = total_quality / count
                avg_distance = total_distance / count
            else:
                avg_quality = 0.0
                avg_distance = 0.0

            # ALERTA CRÍTICO se qualidade é 0 mas há experiências
            if count > 10 and avg_quality == 0:
                self.logger.error(f"🚨🚨 ALERTA CRÍTICO: {count} experiências mas qualidade média 0.00!")
                # DEBUG DETALHADO
                sample_exps = all_experiences[:5]  # Primeiras 5
                for i, exp in enumerate(sample_exps):
                    metrics = exp.info.get("metrics", {})
                    distance = metrics.get("distance", 0)
                    self.logger.error(f"   🧪 Amostra {i+1}: Quality={exp.quality:.2f}, Distance={distance:.2f}m, Reward={exp.reward:.1f}")

            cache_hits = self.performance_stats["cache_hits"]
            cache_misses = self.performance_stats["cache_misses"]
            cache_total = cache_hits + cache_misses

            return {
                "total_experiences": self.stored_count,
                "stored_count": self.stored_count,
                "rejected_count": self.rejected_count,
                "rejection_rate": self.rejected_count / max(self.episode_count, 1),
                "cache_hit_rate": cache_hits / max(cache_total, 1),
                "current_group_size": len(self.current_group_buffer),
                "core_buffer_size": len(self.core_buffer.buffer),
                "avg_quality": avg_quality,
                "avg_distance": avg_distance,
                "total_calculated": count,
                "quality_calculation_working": avg_quality > 0  # INDICADOR CRÍTICO
            }

        except Exception as e:
            self.logger.error(f"❌ ERRO RADICAL no cálculo de status: {e}")
            return {"error": str(e), "critical": True}
    
    def _calculate_avg_distance(self) -> float:
        """Calcula distância média das experiências"""
        total_distance = 0.0
        count = 0

        for buffer in self.group_buffers.values():
            if buffer.buffer:
                for exp in buffer.buffer[:50]:  
                    distance = exp.info.get("metrics", {}).get("distance", 0)
                    total_distance += max(distance, 0)  
                    count += 1

        return total_distance / max(count, 1)
    
    def _calculate_avg_quality(self) -> float:
        """Calcula qualidade média de forma eficiente"""
        total_quality = 0.0
        count = 0
        
        for buffer in self.group_buffers.values():
            if buffer.buffer:
                total_quality += sum(exp.quality for exp in buffer.buffer[:50])  # Amostra
                count += min(len(buffer.buffer), 50)
        
        if count > 0:
            return total_quality / count
        return 0.0

    def get_metrics(self) -> Dict:
        """Métricas focadas em performance - OTIMIZADAS"""
        base_metrics = self.get_status()

        if not self.current_group_buffer:
            return {**base_metrics, "movement_efficiency": 0, "positive_movement_rate": 0}

        # Cálculos eficientes
        recent_experiences = self.current_group_buffer[-100:]  # Amostra recente
        positive_exps = [exp for exp in recent_experiences 
                        if exp.info.get("metrics", {}).get("distance", 0) > 0.1]

        positive_rate = len(positive_exps) / len(recent_experiences) if recent_experiences else 0

        optimization_metrics = {
            "cache_hit_rate": base_metrics["cache_hit_rate"],
            "positive_movement_rate": positive_rate,
            "movement_efficiency": min(positive_rate * 2.0, 1.0),
            "rejection_rate": base_metrics["rejection_rate"],
            "buffer_efficiency": len(self.current_group_buffer) / self.max_experiences
        }

        return {**base_metrics, **optimization_metrics}