# main.py
import os
import shutil
import logging
import multiprocessing
from gui import TrainingGUI
import utils


def setup_folders():
    for folder in [utils.TMP_PATH, utils.LOGS_PATH]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder, exist_ok=True)


if __name__ == "__main__":
    setup_folders()
    utils.get_logger()

    logging.info(f"Executando em {utils.PROJECT_ROOT}")

    logging.info("Iniciando GUI...")
    app = TrainingGUI()
    app.start()
    logging.info("Programa finalizado.")
