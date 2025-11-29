# utils.py
from collections import deque
import os
import logging
import multiprocessing
import queue
import json
from datetime import datetime
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
        # Metas por fase: [reward_per_step, distance_per_step, success_rate]
        self.phase_targets = {
            1: [1.0, 0.015, 0.01],   # Fase Iniciante
            2: [5.0, 0.020, 0.15],   # Fase Intermediária  
            3: [10.0, 0.025, 0.30]    # Fase Avançada
        }
        self.metric_history = {
            'rps': deque(maxlen=8),    # reward per step
            'dps': deque(maxlen=8),    # distance per step
            'success': deque(maxlen=6) # success (1 ou 0)
        }
        self.phase_transition_occurred = False
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas para decisão de fase"""
        steps = episode_metrics.get('steps', 1)
        reward = episode_metrics.get('reward', 0)
        distance = episode_metrics.get('distances', 0)
        success = episode_metrics.get('success', False)
        
        # Calcular métricas por passo
        rps = reward / steps if steps > 0 else 0
        dps = distance / steps if steps > 0 else 0
        success_value = 1.0 if success else 0.0
        
        self.metric_history['rps'].append(rps)
        self.metric_history['dps'].append(dps)
        self.metric_history['success'].append(success_value)
    
    def should_transition_phase(self):
        """Decide se deve transicionar para a próxima fase"""
        if (len(self.metric_history['rps']) < 5 or 
            len(self.metric_history['dps']) < 5 or 
            len(self.metric_history['success']) < 5):
            return False
            
        targets = self.phase_targets[self.current_phase]
        avg_rps = np.mean(self.metric_history['rps'])
        avg_dps = np.mean(self.metric_history['dps'])
        avg_success = np.mean(self.metric_history['success'])
        
        # Verificar se atingiu pelo menos 90% da meta em todos os aspectos
        transition = (avg_rps >= targets[0] * 0.9 and 
                     avg_dps >= targets[1] * 0.9 and 
                     avg_success >= targets[2] * 0.9)
        
        if transition and self.current_phase < 3:
            self.current_phase += 1
            self.phase_transition_occurred = True
            return True
            
        return False
    
    def get_phase_weight_multiplier(self):
        """Retorna multiplicador de pesos baseado no progresso na fase atual"""
        if (len(self.metric_history['rps']) == 0 or 
            len(self.metric_history['dps']) == 0 or 
            len(self.metric_history['success']) == 0):
            return 1.0
            
        targets = self.phase_targets[self.current_phase]
        avg_rps = np.mean(self.metric_history['rps'])
        avg_dps = np.mean(self.metric_history['dps'])
        avg_success = np.mean(self.metric_history['success'])
        
        progress_rps = min(1.0, avg_rps / targets[0])
        progress_dps = min(1.0, avg_dps / targets[1])
        progress_success = min(1.0, avg_success / targets[2])
        
        # Progresso geral (ponderado)
        overall_progress = (progress_rps * 0.4 + progress_dps * 0.4 + progress_success * 0.2)
        
        # Multiplicador: 1.0 (sem progresso) até 2.0 (atingiu a meta)
        multiplier = 1.0 + overall_progress
        
        return multiplier
    
    def get_phase_info(self):
        """Retorna informações da fase atual para logging"""
        targets = self.phase_targets[self.current_phase]
        current_rps = np.mean(self.metric_history['rps']) if self.metric_history['rps'] else 0
        current_dps = np.mean(self.metric_history['dps']) if self.metric_history['dps'] else 0
        current_success = np.mean(self.metric_history['success']) if self.metric_history['success'] else 0
        
        return {
            'phase': self.current_phase,
            'target_rps': targets[0],
            'target_dps': targets[1],
            'target_success': targets[2],
            'current_rps': current_rps,
            'current_dps': current_dps,
            'current_success': current_success,
            'progress_percentage': min(100, int(self.get_phase_weight_multiplier() * 50 - 50))
        }