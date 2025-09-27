# main.py
import torch
import os
import shutil
from gui import TrainingGUI
import utils


def setup_folders():
    logs = []

    for folder in [utils.TMP_PATH, utils.LOGS_PATH]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except PermissionError:
                logs.append(f"Não foi possível deletar {folder} (arquivo em uso). Continuando...")

        os.makedirs(folder, exist_ok=True)

    return logs


if __name__ == "__main__":
    folder_logs = setup_folders()
    logger = utils.get_logger()
    logger.info("\n".join(folder_logs))

    logger.info(f"Executando em {utils.PROJECT_ROOT}")

    logger.info("Verificando GPU")
    logger.info(f"Cuda version: {torch.version.cuda}")
    is_gpu_available = torch.cuda.is_available()
    logger.info(f"GPU available: {is_gpu_available}")

    if is_gpu_available:
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        device = "cuda"

    else:
        device = "cpu"

    logger.info("Iniciando GUI...")
    app = TrainingGUI(device)
    app.start()
    logger.info("Programa finalizado.")
