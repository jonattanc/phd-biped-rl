# utils.py
from collections import deque
import os
import logging
import multiprocessing
import queue
import json
from datetime import datetime
import time
from typing import Dict
import numpy as np


class FormattedQueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)

        except Exception:
            self.handleError(record)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_PATH = os.path.join(PROJECT_ROOT, "tmp")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")
ESPECIALISTAS_PATH = os.path.join(PROJECT_ROOT, "especialistas")
REWARD_CONFIGS_PATH = os.path.join(PROJECT_ROOT, "reward_configs")
ENVIRONMENT_PATH = os.path.join(PROJECT_ROOT, "environments")
ROBOTS_PATH = os.path.join(PROJECT_ROOT, "robots")
TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, "training_data")
TEMP_MODEL_SAVE_PATH = os.path.join(TMP_PATH, "improvement_models")
TEMP_EVALUATION_SAVE_PATH = os.path.join(TMP_PATH, "evaluation_data")


def get_logger(description=["main"], ipc_queue=None):
    proc = multiprocessing.current_process()
    proc_num = proc._identity[0] if proc._identity else os.getpid()
    log_name = "__".join([str(item) for item in description])

    logger = logging.getLogger(log_name)

    if not logger.handlers:
        if log_name == "main":
            log_filename = os.path.join(PROJECT_ROOT, "logs", "log__main.txt")

        else:
            log_filename = os.path.join(PROJECT_ROOT, "logs", f"log__{log_name}__proc{proc_num}.txt")

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if ipc_queue is not None:
            queue_handler = FormattedQueueHandler(ipc_queue)
            queue_handler.setFormatter(formatter)
            logger.addHandler(queue_handler)

    return logger


# Funções de Logging e IPC
def add_queue_handler_to_logger(logger, ipc_queue):
    """Configura logging para IPC entre processos"""
    if ipc_queue is not None:
        # Verificar se o logger já tem um handler de queue para evitar duplicação
        has_queue_handler = any(isinstance(handler, FormattedQueueHandler) for handler in logger.handlers)
        if not has_queue_handler:
            queue_handler = FormattedQueueHandler(ipc_queue)
            formatter = logging.Formatter("%(asctime)s %(message)s")
            queue_handler.setFormatter(formatter)
            logger.addHandler(queue_handler)


# Funções de Arquivo/IO:
def ensure_directory(path):
    """Garante que um diretório existe"""
    os.makedirs(path, exist_ok=True)
    return path


def find_model_files(directory):
    """Encontra arquivos de modelo em um diretório (busca flexível)"""
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                model_files.append(os.path.join(root, file))
    return model_files


def load_default_settings():
    """Carrega as configurações padrão do arquivo JSON"""
    default_settings_path = os.path.join(PROJECT_ROOT, "default_settings.json")

    default_settings = {
        "default_robot": "robot_stage5",
        "reward_config_file": "default",
        "enable_visualize_robot": False,
        "enable_real_time": True,
        "camera_index": 1,
    }

    if not os.path.exists(default_settings_path):
        return default_settings

    with open(default_settings_path, "r", encoding="utf-8") as f:
        loaded_settings = json.load(f)

    for key, value in loaded_settings.items():
        default_settings[key] = value

    return default_settings


def make_serializable(obj):
    """Converte objetos em tipos compatíveis com JSON."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        # self.logger.warning(f"Making {obj} of type np.ndarray serializable")
        return obj.tolist()
    elif isinstance(obj, np.generic):
        # self.logger.warning(f"Making {obj} of type np.generic serializable")
        return obj.item()
    elif isinstance(obj, (datetime,)):
        # self.logger.warning(f"Making {obj} of type datetime serializable")
        return obj.isoformat()
    else:
        return obj
    

class PhaseManager:
    def __init__(self):
        self.current_phase = 1
        self.phase_history = []
        self.metrics_buffer = []
        self.buffer_size = 50
        
        # Critérios de transição
        self.phase1_to_2_threshold = 4.0  # distância média > 4m
        self.phase2_to_3_threshold = 9.0  # primeiro sucesso de 9m
        self.success_achieved = False
        
        # AJUSTES de peso por fase (em relação ao default.json)
        self.phase_weight_adjustments = {
            1: {},  # Fase 1: usa 100% dos pesos do default.json
            2: {    # Fase 2: ajusta componentes de eficiência
                'efficiency_bonus': 25.0,  # 2500% - Eficiência avançada
                'progress': 2.0,           # 200% do peso original  
                'gait_state_change': 1.5,  # 150% do peso original
                'foot_clearance': 15.0,    # 1500% do peso original
                'y_axis_deviation_square_penalty': 5.0,  # 500% - Precisão lateral
            },
            3: {    # Fase 3: ajusta componentes de performance
                'efficiency_bonus': 15.0,  # 1500% - Eficiência avançada
                'progress': 2.5,           # 250% do peso original
                'gait_state_change': 2.0,  # 200% do peso original 
                'foot_clearance': 10.0,    # 1000% do peso original 
                'fall_penalty': 2.0,       # 200% - Penalidade máxima por queda
                'y_axis_deviation_square_penalty': 15.0, # 1500% - Precisão lateral
            }
        }
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas do episódio"""
        self.metrics_buffer.append(episode_metrics)
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer.pop(0)
            
        if episode_metrics.get('success', False):
            self.success_achieved = True
    
    def should_transition_phase(self):
        """Verifica se deve transicionar de fase"""
        if len(self.metrics_buffer) < 10:
            return False
            
        current_metrics = self.get_current_metrics()
        
        if self.current_phase == 1:
            # Fase 1 -> 2: distância média > 4m
            if current_metrics['avg_distance'] > self.phase1_to_2_threshold:
                return True
                
        elif self.current_phase == 2:
            # Fase 2 -> 3: primeiro sucesso de 9m alcançado
            if self.success_achieved:
                return True
                
        return False
    
    def get_current_metrics(self):
        """Calcula métricas atuais do buffer"""
        if not self.metrics_buffer:
            return {
                'avg_reward': 0, 
                'avg_distance': 0, 
                'success_rate': 0,
                'reward_per_step': 0,
                'distance_per_step': 0
            }

        avg_reward = np.mean([m.get('reward', 0) for m in self.metrics_buffer])
        avg_distance = np.mean([m.get('distances', 0) for m in self.metrics_buffer])
        success_rate = np.mean([m.get('success', False) for m in self.metrics_buffer])
        total_reward = sum([m.get('reward', 0) for m in self.metrics_buffer])
        total_distance = sum([m.get('distances', 0) for m in self.metrics_buffer])
        total_steps = sum([m.get('steps', 1) for m in self.metrics_buffer])

        if total_steps > 0:
            reward_per_step = total_reward / total_steps
            distance_per_step = total_distance / total_steps
        else:
            reward_per_step = 0
            distance_per_step = 0

        return {
            'avg_reward': avg_reward,
            'avg_distance': avg_distance, 
            'success_rate': success_rate,
            'reward_per_step': reward_per_step,
            'distance_per_step': distance_per_step
        }
    
    def get_phase_weight_adjustments(self):
        """Retorna ajustes de peso para a fase atual"""
        return self.phase_weight_adjustments.get(self.current_phase, {})
    
    def transition_to_next_phase(self):
        """Transiciona para próxima fase"""
        if self.current_phase < 3:
            self.current_phase += 1
            self.phase_history.append({
                'phase': self.current_phase,
                'timestamp': time.time(),
                'metrics': self.get_current_metrics()
            })
            return True
        return False
    
    def get_phase_info(self):
        """Retorna informações detalhadas da fase atual"""
        current_metrics = self.get_current_metrics()
        
        return {
            'phase': self.current_phase,
            'current_rps': current_metrics['reward_per_step'],
            'current_dps': current_metrics['distance_per_step'], 
            'current_success': current_metrics['success_rate'],
            'avg_distance': current_metrics['avg_distance'],
            'avg_reward': current_metrics['avg_reward'],
            'success_achieved': self.success_achieved,
            'weight_adjustments': self.get_phase_weight_adjustments()
        }