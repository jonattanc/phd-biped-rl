# dpg_buffer.py
import numpy as np
from typing import Dict, List, Any
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
    phase: int
    quality: float


class BufferManager:
    """
    Gerenciador inteligente de buffers com preservação de aprendizado
    """
    
    def __init__(self, logger, config, max_core_experiences=100):
        self.logger = logger
        self.config = config
        self.max_core_experiences = max_core_experiences
        
        # Buffers organizados
        self.phase_buffers = {}  # Buffer por fase
        self.core_buffer = deque(maxlen=max_core_experiences)  # Experiências fundamentais
        self.current_phase_buffer = []  # Buffer da fase atual
        
        # Estatísticas
        self.experience_count = 0
        self.phase_transitions = 0
    
    def store_experience(self, sim, action, reward, phase_info):
        """Armazena experiência nos buffers apropriados"""
        experience = self._create_experience(sim, action, reward, phase_info)
        
        # Armazenar na fase atual
        current_phase = phase_info['phase']
        if current_phase not in self.phase_buffers:
            self.phase_buffers[current_phase] = []
        
        self.phase_buffers[current_phase].append(experience)
        self.current_phase_buffer.append(experience)
        
        # Se for experiência fundamental, armazenar no core
        if self._is_fundamental_experience(experience):
            self.core_buffer.append(experience)
        
        self.experience_count += 1
        
        # Limitar tamanho do buffer atual
        if len(self.current_phase_buffer) > 1000:
            self.current_phase_buffer = self.current_phase_buffer[-500:]
    
    def transition_phase(self, old_phase, new_phase):
        """Executa transição suave entre fases preservando aprendizado"""
        self.phase_transitions += 1
        
        # 1. Extrair experiências fundamentais da fase antiga
        fundamental_experiences = self._extract_fundamental_experiences(old_phase)
        
        # 2. Filtrar experiências relevantes para nova fase
        relevant_experiences = self._filter_relevant_experiences(old_phase, new_phase)
        
        # 3. Combinar: relevantes + fundamentais do core
        new_phase_experiences = relevant_experiences + list(self.core_buffer)
        
        # 4. Inicializar buffer da nova fase
        self.phase_buffers[new_phase] = new_phase_experiences
        self.current_phase_buffer = new_phase_experiences
        
        self.logger.info(f"🔄 Transição buffer: Fase {old_phase}→{new_phase}, "
                        f"Experiências: {len(new_phase_experiences)}")
    
    def get_training_batch(self, batch_size=32):
        """Retorna batch para treinamento da fase atual"""
        if not self.current_phase_buffer:
            return None
        
        # Amostrar do buffer atual + algumas do core
        available_experiences = self.current_phase_buffer + list(self.core_buffer)
        
        if len(available_experiences) < batch_size:
            batch_size = len(available_experiences)
        
        # Amostrar com prioridade para experiências de alta qualidade
        qualities = [exp.quality for exp in available_experiences]
        probabilities = np.array(qualities) / sum(qualities)
        
        indices = np.random.choice(
            len(available_experiences), 
            size=batch_size, 
            p=probabilities,
            replace=False
        )
        
        return [available_experiences[i] for i in indices]
    
    def _create_experience(self, sim, action, reward, phase_info):
        """Cria objeto de experiência com métricas de qualidade"""
        # Extrair estado (simplificado)
        state = self._extract_state(sim)
        next_state = state  # Simplificado - na prática seria próximo estado
        
        # Calcular qualidade da experiência
        quality = self._calculate_experience_quality(sim, action, reward, phase_info)
        
        return Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,  # Simplificado
            info=phase_info,
            phase=phase_info['phase'],
            quality=quality
        )
    
    def _extract_state(self, sim):
        """Extrai representação do estado do simulador"""
        state_features = []
        
        # Adicionar features básicas
        try:
            state_features.extend([
                getattr(sim, "robot_x_velocity", 0),
                getattr(sim, "robot_y_velocity", 0),
                getattr(sim, "robot_roll", 0),
                getattr(sim, "robot_pitch", 0),
                getattr(sim, "robot_left_foot_contact", 0),
                getattr(sim, "robot_right_foot_contact", 0),
            ])
        except Exception:
            pass
        
        return np.array(state_features, dtype=np.float32)
    
    def _calculate_experience_quality(self, sim, action, reward, phase_info):
        """Calcula qualidade da experiência para priorização"""
        quality = 0.0
        
        # 1. Recompensa alta
        quality += min(abs(reward) * 0.2, 1.0)
        
        # 2. Progresso positivo
        distance = getattr(sim, "episode_distance", 0)
        if distance > 0:
            quality += min(distance * 3.0, 1.0)
        
        # 3. Estabilidade
        roll = abs(getattr(sim, "robot_roll", 0))
        pitch = abs(getattr(sim, "robot_pitch", 0))
        stability = 1.0 - min(roll + pitch, 2.0) / 2.0
        quality += stability * 0.3
        
        # 4. Ação suave
        if hasattr(action, '__len__') and len(action) > 1:
            smoothness = 1.0 - min(np.std(action) * 3.0, 1.0)
            quality += smoothness * 0.2
        
        return min(quality, 1.0)
    
    def _is_fundamental_experience(self, experience):
        """Verifica se experiência é fundamental (útil em qualquer fase)"""
        return (experience.reward > 0.5 and 
                experience.quality > 0.7 and
                not experience.done)
    
    def _extract_fundamental_experiences(self, phase):
        """Extrai experiências fundamentais de uma fase específica"""
        if phase not in self.phase_buffers:
            return []
        
        fundamental = [
            exp for exp in self.phase_buffers[phase] 
            if self._is_fundamental_experience(exp)
        ]
        
        # Manter apenas as melhores
        fundamental.sort(key=lambda x: x.quality, reverse=True)
        return fundamental[:50]
    
    def _filter_relevant_experiences(self, old_phase, new_phase):
        """Filtra experiências relevantes para nova fase"""
        if old_phase not in self.phase_buffers:
            return []
        
        old_experiences = self.phase_buffers[old_phase]
        relevant = []
        
        for exp in old_experiences:
            # Manter experiências com boa qualidade e recompensa positiva
            if exp.quality > 0.6 and exp.reward > 0:
                relevant.append(exp)
        
        # Limitar quantidade
        return relevant[:100]
    
    def get_status(self):
        """Retorna status do gerenciador de buffers"""
        total_phase_experiences = sum(len(buffer) for buffer in self.phase_buffers.values())
        
        return {
            "total_experiences": self.experience_count,
            "core_experiences": len(self.core_buffer),
            "current_phase_experiences": len(self.current_phase_buffer),
            "total_phase_experiences": total_phase_experiences,
            "phase_transitions": self.phase_transitions,
            "phases_with_buffer": list(self.phase_buffers.keys())
        }
    
    def get_metrics(self):
        """Retorna métricas para monitoramento"""
        avg_quality = np.mean([exp.quality for exp in self.current_phase_buffer]) if self.current_phase_buffer else 0
        avg_reward = np.mean([exp.reward for exp in self.current_phase_buffer]) if self.current_phase_buffer else 0
        
        return {
            "buffer_avg_quality": avg_quality,
            "buffer_avg_reward": avg_reward,
            "core_buffer_size": len(self.core_buffer),
            "current_buffer_size": len(self.current_phase_buffer)
        }
    
    def clear_phase_buffer(self, phase):
        """Limpa buffer de uma fase específica"""
        if phase in self.phase_buffers:
            del self.phase_buffers[phase]
    
    def clear_all_buffers(self):
        """Limpa todos os buffers (usar com cuidado!)"""
        self.phase_buffers.clear()
        self.core_buffer.clear()
        self.current_phase_buffer.clear()
        self.experience_count = 0
        self.logger.warning("🧹 Todos os buffers DPG foram limpos")