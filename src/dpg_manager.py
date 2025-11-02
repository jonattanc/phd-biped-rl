# dpg_manager.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import torch
import torch.nn as nn
from dpg_phase import PhaseManager, PhaseTransitionResult
from dpg_reward import RewardCalculator
from dpg_buffer import SmartBufferManager


class LearningMode(Enum):
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    TRANSITION = "transition"
    RECOVERY = "recovery"


@dataclass
class DPGConfig:
    """Configuração centralizada do DPG com suporte a grupos"""
    enabled: bool = False
    learning_mode: LearningMode = LearningMode.STANDARD
    adaptive_learning: bool = True
    irl_enabled: bool = True
    hdpg_enabled: bool = True
    
    def __post_init__(self):
        self.phase_weights = {
            "velocity": 2.0, "phase_angles": 1.0, "propulsion": 0.5,
            "clearance": 0.2, "stability": 3.0, "symmetry": 0.3,
            "effort_torque": 1e-4, "effort_power": 1e-5, 
            "action_smoothness": 1e-3, "lateral_penalty": 1.0, 
            "slip_penalty": 0.5,
        }

   
class AdaptiveCritic(nn.Module):
    """Crítico com Arquitetura Adaptável"""
    
    def __init__(self, input_dim, architecture_mode="basic"):
        super(AdaptiveCritic, self).__init__()
        self.architecture_mode = architecture_mode
        self.architectures = {
            "basic": {"hidden_dims": [64, 32], "num_heads": 2},
            "standard": {"hidden_dims": [128, 64, 32], "num_heads": 4},
            "advanced": {"hidden_dims": [256, 128, 64, 32], "num_heads": 6}
        }
        
        self.config = self.architectures[architecture_mode]
        self.layers = self._build_layers(input_dim)
        self.attention_heads = self._build_attention_heads()
        
    def _build_layers(self, input_dim):
        """Constrói camadas baseado na arquitetura"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.config["hidden_dims"]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_attention_heads(self):
        """Constrói heads de atenção para multi-head critic"""
        heads = []
        output_dim = self.config["hidden_dims"][-1]
        
        for _ in range(self.config["num_heads"]):
            heads.append(nn.Sequential(
                nn.Linear(output_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ))
        
        return nn.ModuleList(heads)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.layers(x)
        
        # Multi-head attention
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(features))
        
        # Combinação simples (média)
        final_output = torch.mean(torch.stack(head_outputs), dim=0)
        
        return final_output, head_outputs
    
    def migrate_architecture(self, new_mode: str, input_dim: int):
        """Migra para nova arquitetura preservando pesos"""
        if new_mode == self.architecture_mode:
            return
        
        old_weights = self._get_current_weights()
        self.architecture_mode = new_mode
        self.config = self.architectures[new_mode]
        
        # Reconstruir layers
        self.layers = self._build_layers(input_dim)
        self.attention_heads = self._build_attention_heads()
        
        # Migrar pesos compatíveis
        self._migrate_weights(old_weights)
    
    def _get_current_weights(self):
        """Obtém pesos atuais para migração"""
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.clone()
        return weights
    
    def _migrate_weights(self, old_weights):
        """Migra pesos para nova arquitetura"""
        for name, param in self.named_parameters():
            if name in old_weights:
                old_param = old_weights[name]
                if old_param.shape == param.shape:
                    param.data.copy_(old_param)
                elif len(old_param.shape) == 2 and len(param.shape) == 2:
                    # Tentar migração parcial para camadas lineares
                    min_rows = min(old_param.shape[0], param.shape[0])
                    min_cols = min(old_param.shape[1], param.shape[1])
                    param.data[:min_rows, :min_cols] = old_param[:min_rows, :min_cols]


class CriticManager:
    """Gerenciador do Critic Adaptável"""
    
    def __init__(self, logger, state_dim, action_dim):
        self.logger = logger
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        
        self.critic = AdaptiveCritic(self.input_dim, "basic")
        self.architecture_history = []
        
    def adapt_architecture(self, group_level: int, performance: float, experience_count: int):
        """Adapta arquitetura do critic baseado no contexto"""
        target_architecture = self._select_target_architecture(group_level, performance, experience_count)
        
        if target_architecture != self.critic.architecture_mode:
            self.logger.info(f"🔄 Migrando critic: {self.critic.architecture_mode} → {target_architecture}")
            self.critic.migrate_architecture(target_architecture, self.input_dim)
            
            self.architecture_history.append({
                'old_architecture': self.critic.architecture_mode,
                'new_architecture': target_architecture,
                'group_level': group_level,
                'performance': performance,
                'timestamp': np.datetime64('now')
            })
    
    def _select_target_architecture(self, group_level: int, performance: float, experience_count: int) -> str:
        """Seleciona arquitetura alvo baseado no contexto"""
        
        # Grupo 1: Arquitetura básica
        if group_level == 1:
            return "basic"
        
        # Grupo 2: Standard se performance boa e experiências suficientes
        elif group_level == 2:
            if performance > 0.7 and experience_count > 1000:
                return "standard"
            return "basic"
        
        # Grupo 3: Advanced para alta performance
        elif group_level == 3:
            if performance > 0.8 and experience_count > 2000:
                return "advanced"
            elif performance > 0.6:
                return "standard"
            return "basic"
        
        return "basic"
    
    def get_critic_status(self) -> Dict:
        """Retorna status do critic"""
        return {
            "current_architecture": self.critic.architecture_mode,
            "architecture_history": len(self.architecture_history),
            "input_dim": self.input_dim,
            "hidden_dims": self.critic.config["hidden_dims"],
            "num_heads": self.critic.config["num_heads"]
        }
    

class DPGManager:
    """
    CÉREBRO DO SISTEMA com todos os componentes adaptáveis
    """
    
    def __init__(self, logger, robot, reward_system, state_dim=10, action_dim=6):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system

        self.config = type('Config', (), {})()
        self.config.enabled = True
        
        # Componentes especializados
        self.phase_manager = PhaseManager(logger, {})
        self.reward_calculator = RewardCalculator(logger, {})
        self.buffer_manager = SmartBufferManager(logger, {})
        self.critic_manager = CriticManager(logger, state_dim, action_dim)
        
        # Estado do sistema
        self.enabled = False
        self.learning_progress = 0.0
        self.performance_trend = 0.0
    
    def enable(self, enabled=True):
        """Ativa o sistema completo"""
        self.enabled = enabled
        if enabled:
            self.logger.info("🧠 Sistema DPG Adaptável ativado")
    
    def calculate_reward(self, sim, action) -> float:
        """Calcula recompensa com todos os sistemas"""
        if not self.enabled:
            return 0.0
        
        # Obter contexto atual
        phase_info = self.phase_manager.get_current_phase_info()
        
        # Calcular recompensa
        reward = self.reward_calculator.calculate(sim, action, phase_info)
        
        # Armazenar experiência
        experience_data = {
            "state": self._extract_state(sim),
            "action": action,
            "reward": reward,
            "phase_info": phase_info,
            "metrics": self._extract_metrics(sim),
            "group_level": phase_info.get('group_level', 1)
        }
        self.buffer_manager.store_experience(experience_data)
        
        # Atualizar critic
        self._update_critic_architecture()
        
        return reward
    
    def get_brain_status(self) -> Dict:
        """Retorna status completo do sistema DPG"""
        if not self.enabled:
            return {"status": "disabled"}

        status = {
            "status": "active",
            "enabled": self.enabled,
            "learning_progress": self.learning_progress,
            "performance_trend": self.performance_trend,
        }

        # Coletar status dos componentes se disponíveis
        if hasattr(self, 'phase_manager') and self.phase_manager:
            try:
                phase_status = self.phase_manager.get_status()
                status.update({
                    "current_group": phase_status.get("current_group", 0),
                    "current_sub_phase": phase_status.get("current_sub_phase", 0),
                    "group_name": phase_status.get("group_name", "unknown"),
                    "episodes_in_sub_phase": phase_status.get("episodes_in_sub_phase", 0),
                    "success_rate": phase_status.get("success_rate", 0.0),
                    "avg_distance": phase_status.get("avg_distance", 0.0),
                    "consecutive_failures": phase_status.get("consecutive_failures", 0),
                    "validation_required": phase_status.get("validation_required", False),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter status do phase_manager: {e}")

        if hasattr(self, 'buffer_manager') and self.buffer_manager:
            try:
                buffer_status = self.buffer_manager.get_status()
                status.update({
                    "total_experiences": buffer_status.get("total_experiences", 0),
                    "current_group_experiences": buffer_status.get("current_group_experiences", 0),
                    "group_transitions": buffer_status.get("group_transitions", 0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter status do buffer_manager: {e}")

        if hasattr(self, 'reward_calculator') and self.reward_calculator:
            try:
                reward_status = self.reward_calculator.get_reward_status()
                status.update({
                    "components_enabled": reward_status.get("components_enabled", 0),
                    "demonstration_count": reward_status.get("demonstration_count", 0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter status do reward_calculator: {e}")

        return status
    
    def get_advanced_metrics(self) -> Dict:
        """Retorna métricas avançadas para monitoramento"""
        if not self.enabled:
            return {}
        
        metrics = {
            "system_health": 1.0,
            "learning_progress": self.learning_progress,
            "performance_trend": self.performance_trend,
            "dass_samples": 0,
            "irl_confidence": 0.0,
            "hdpg_convergence": 0.0,
            "hdpg_active": False,
        }
        
        # Coletar métricas dos componentes se disponíveis
        if hasattr(self, 'phase_manager') and self.phase_manager:
            try:
                phase_status = self.phase_manager.get_status()
                metrics.update({
                    "current_phase": phase_status.get("current_group", 0),
                    "phase_name": phase_status.get("group_name", "unknown"),
                    "success_rate": phase_status.get("success_rate", 0.0),
                    "avg_distance": phase_status.get("avg_distance", 0.0),
                    "consecutive_failures": phase_status.get("consecutive_failures", 0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter métricas do phase_manager: {e}")
        
        if hasattr(self, 'buffer_manager') and self.buffer_manager:
            try:
                buffer_metrics = self.buffer_manager.get_metrics()
                metrics.update({
                    "buffer_avg_quality": buffer_metrics.get("buffer_avg_quality", 0),
                    "buffer_avg_reward": buffer_metrics.get("buffer_avg_reward", 0),
                    "current_buffer_size": buffer_metrics.get("current_buffer_size", 0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter métricas do buffer_manager: {e}")
        
        if hasattr(self, 'reward_calculator') and self.reward_calculator:
            try:
                reward_status = self.reward_calculator.get_reward_status()
                metrics.update({
                    "demonstration_count": reward_status.get("demonstration_count", 0),
                    "irl_confidence": reward_status.get("model_confidences", {}).get("group_1", 0.0),
                })
            except Exception as e:
                self.logger.warning(f"Erro ao obter métricas do reward_calculator: {e}")
        
        return metrics
    
    def _update_critic_architecture(self):
        """Atualiza arquitetura do critic baseado no progresso"""
        if not self.phase_manager or not self.buffer_manager:
            return
        
        group_level = self.phase_manager.current_group_config.group_level
        performance = self.phase_manager._calculate_success_rate()
        experience_count = self.buffer_manager.experience_count
        
        self.critic_manager.adapt_architecture(group_level, performance, experience_count)
    
    def update_phase_progression(self, episode_results):
        """Atualiza progressão com validação"""
        if not self.enabled:
            return
        
        result = self.phase_manager.update_phase(episode_results)
        
        # Executar validação se necessária
        if result == PhaseTransitionResult.VALIDATION_REQUIRED:
            validation_results = self.phase_manager.execute_validation()
            self.logger.info(f"🔍 Validação executada: {validation_results}")
        
        # Atualizar transições de buffer se grupo mudou
        if (hasattr(self, 'last_group') and 
            self.last_group != self.phase_manager.current_group):
            
            self.buffer_manager.transition_with_preservation(
                self.last_group,
                self.phase_manager.current_group,
                self.phase_manager.current_group_config.adaptive_config
            )
        
        self.last_group = self.phase_manager.current_group
        return result
    
    def _extract_state(self, sim):
        """Extrai estado do simulador"""
        return np.array([
            getattr(sim, "robot_x_velocity", 0),
            getattr(sim, "robot_roll", 0),
            getattr(sim, "robot_pitch", 0),
        ], dtype=np.float32)
    
    def _extract_metrics(self, sim):
        """Extrai métricas do simulador"""
        return {
            "distance": getattr(sim, "episode_distance", 0),
            "speed": getattr(sim, "robot_x_velocity", 0),
            "roll": abs(getattr(sim, "robot_roll", 0)),
            "pitch": abs(getattr(sim, "robot_pitch", 0)),
        }
    
    def get_system_status(self) -> Dict:
        """Retorna status completo do sistema"""
        status = {
            "enabled": self.enabled,
            "learning_progress": self.learning_progress,
            "performance_trend": self.performance_trend,
        }
        
        # Coletar status de todos os componentes
        if self.phase_manager:
            status.update(self.phase_manager.get_status())
        
        if self.reward_calculator:
            status.update(self.reward_calculator.get_reward_status())
        
        if self.buffer_manager:
            status.update(self.buffer_manager.get_status())
        
        if self.critic_manager:
            status.update(self.critic_manager.get_critic_status())
        
        return status