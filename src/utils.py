# utils.py
import os
import logging
import multiprocessing
import queue
import json


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
        "enable_dynamic_policy_gradient": True,
        "enable_visualize_robot": True,
        "enable_real_time": True,
        "camera_index": 1,
    }

    if not os.path.exists(default_settings_path):
        return default_settings

    with open(default_settings_path, "r") as f:
        loaded_settings = json.load(f)

    for key, value in loaded_settings.items():
        default_settings[key] = value

    return default_settings
