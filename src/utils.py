# utils.py
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
