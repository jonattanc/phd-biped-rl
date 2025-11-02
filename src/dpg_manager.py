# dpg_manager.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from dpg_phase import PhaseManager
from dpg_reward import RewardCalculator
from dpg_buffer import BufferManager


@dataclass
class DPGConfig:
    """Configuração centralizada do DPG"""
    enabled: bool = False
    initial_posture: Dict = None
    phase_targets: Dict = None
    phase_weights: Dict = None
    
    def __post_init__(self):
        if self.initial_posture is None:
            self.initial_posture = {
                "hip_frontal": 0.0, "hip_lateral": 0.0, "knee": 0.4,
                "ankle_frontal": -0.15, "ankle_lateral": 0.0, "body_pitch": 0.1
            }
        
        if self.phase_weights is None:
            self.phase_weights = {
                "velocity": 2.0, "phase_angles": 1.0, "propulsion": 0.5,
                "clearance": 0.2, "stability": 3.0, "symmetry": 0.3,
                "effort_torque": 1e-4, "effort_power": 1e-5, 
                "action_smoothness": 1e-3, "lateral_penalty": 1.0, 
                "slip_penalty": 0.5,
            }


class DPGManager:
    """
    Gerenciador principal - orquestra todos os componentes DPG
    """
    
    def __init__(self, logger, robot, reward_system):
        self.logger = logger
        self.robot = robot
        self.reward_system = reward_system
        self.config = DPGConfig()
        
        # Componentes especializados
        self.phase_manager = None
        self.reward_calculator = None
        self.buffer_manager = None
        
        # Estado
        self.enabled = False
        self.stagnation_counter = 0
    
    def enable(self, enabled=True):
        """Ativa/desativa o sistema DPG completo"""
        self.enabled = enabled
        
        if enabled:
            self._setup_components()
            self.logger.info("Sistema DPG modular ativado")
        else:
            self._cleanup_components()
            self.logger.info("Sistema DPG desativado")
    
    def _setup_components(self):
        """Configura todos os componentes especializados"""
        try:
            self.phase_manager = PhaseManager(self.logger, self.config)
            self.reward_calculator = RewardCalculator(self.logger, self.config)
            self.buffer_manager = BufferManager(self.logger, self.config)
            
            # Conectar ao sistema de recompensa
            self.reward_system.dpg_manager = self
            self.reward_system.phase_manager = self.phase_manager
            
            self.logger.info("✅ Todos os componentes DPG inicializados")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao configurar componentes DPG: {e}")
            raise
    
    def _cleanup_components(self):
        """Limpa todos os componentes"""
        self.phase_manager = None
        self.reward_calculator = None
        self.buffer_manager = None
        
        if hasattr(self.reward_system, 'dpg_manager'):
            self.reward_system.dpg_manager = None
        if hasattr(self.reward_system, 'phase_manager'):
            self.reward_system.phase_manager = None
    
    def calculate_reward(self, sim, action):
        """Calcula recompensa usando o pipeline DPG"""
        if not self.enabled:
            return 0.0
        
        try:
            # 1. Obter informação da fase atual
            phase_info = self.phase_manager.get_current_phase_info()
            
            # 2. Calcular recompensa
            reward = self.reward_calculator.calculate(sim, action, phase_info)
            
            # 3. Gerenciar experiência no buffer
            self.buffer_manager.store_experience(sim, action, reward, phase_info)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"❌ Erro no cálculo de recompensa DPG: {e}")
            return 0.0
    
    def update_phase_progression(self, episode_results):
        """Atualiza progressão de fases baseada nos resultados"""
        if self.enabled and self.phase_manager:
            self.phase_manager.update_phase(episode_results)
    
    def apply_initial_posture(self):
        """Aplica postura inicial otimizada"""
        if not self.enabled:
            return
        
        try:
            posture = self.config.initial_posture
            self.logger.info(f"🎯 Aplicando postura inicial DPG: {posture}")
            # Implementação específica do robô aqui
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao aplicar postura DPG: {e}")
    
    def get_status(self):
        """Retorna status completo do sistema"""
        if not self.enabled:
            return {"enabled": False}
        
        status = {
            "enabled": True,
            "components": {
                "phase_manager": self.phase_manager is not None,
                "reward_calculator": self.reward_calculator is not None,
                "buffer_manager": self.buffer_manager is not None,
            }
        }
        
        # Adicionar status específico dos componentes
        if self.phase_manager:
            status.update(self.phase_manager.get_status())
        
        if self.buffer_manager:
            status["buffer"] = self.buffer_manager.get_status()
        
        return status
    
    def get_advanced_metrics(self):
        """Retorna métricas avançadas para monitoramento"""
        if not self.enabled:
            return {}
        
        metrics = {
            "current_phase": self.phase_manager.current_phase if self.phase_manager else -1,
            "phase_name": self.phase_manager.get_phase_name() if self.phase_manager else "disabled",
            "stagnation_counter": self.stagnation_counter,
        }
        
        # Métricas do buffer
        if self.buffer_manager:
            buffer_metrics = self.buffer_manager.get_metrics()
            metrics.update(buffer_metrics)
        
        return metrics
    
    def get_detailed_status(self):
        """Retorna status detalhado para compatibilidade (similar ao gait_phase_dpg antigo)"""
        if not self.enabled or not self.phase_manager:
            return {
                "current_phase": -1,
                "phase_index": -1,
                "target_speed": 0.0,
                "episodes_in_phase": 0,
                "performance_metrics": {
                    "success_rate": 0.0,
                    "avg_distance": 0.0,
                    "avg_roll": 0.0,
                    "avg_speed": 0.0,
                    "positive_movement_rate": 0.0
                }
            }
        
        phase_info = self.phase_manager.get_current_phase_info()
        status = self.phase_manager.get_status()
        
        return {
            "current_phase": phase_info['phase'],
            "phase_index": phase_info['phase'],
            "target_speed": phase_info['target_speed'],
            "episodes_in_phase": phase_info['episodes_in_phase'],
            "performance_metrics": {
                "success_rate": self.phase_manager._calculate_success_rate(),
                "avg_distance": self.phase_manager._calculate_avg_distance(),
                "avg_roll": self.phase_manager._calculate_avg_roll(),
                "avg_speed": self.phase_manager._calculate_avg_speed(),
                "positive_movement_rate": self._calculate_positive_movement_rate()
            }
        }
    
    def _calculate_positive_movement_rate(self):
        """Calcula taxa de movimento positivo para compatibilidade"""
        if not self.phase_manager.performance_history:
            return 0.0
        positive_movements = sum(1 for r in self.phase_manager.performance_history if r.get("distance", 0) > 0.1)
        return positive_movements / len(self.phase_manager.performance_history)