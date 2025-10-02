# utils.py
import json
import os
import logging
import multiprocessing
import queue


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
ENVIRONMENT_PATH = os.path.join(PROJECT_ROOT, "environments")
ROBOTS_PATH = os.path.join(PROJECT_ROOT, "robots")
TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, "training_data")
TRAINING_CONTROL_PATH = os.path.join(PROJECT_ROOT, "training_control")


def get_logger(description=["main"], ipc_queue=None):
    proc = multiprocessing.current_process()
    proc_num = proc._identity[0] if proc._identity else os.getpid()
    log_name = "__".join(description)

    logger = logging.getLogger(log_name)

    if not logger.handlers:
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
def setup_ipc_logging(logger, ipc_queue):
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


# Funções de Validação:
def validate_episodes_count(episodes_str):
    """Valida e converte número de episódios"""
    try:
        episodes = int(episodes_str)
        if episodes <= 0:
            raise ValueError("Número de episódios deve ser positivo")
        return episodes
    except ValueError:
        raise ValueError("Número de episódios deve ser um inteiro positivo")
