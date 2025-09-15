# main.py
import os
import shutil
import logging
from robot import Robot
from environment import Environment
from agent import Agent
from gui import TrainingGUI
import pybullet as p
import tkinter as tk
import time
from gui import TrainingGUI


def setup_folders():
    for folder in ["tmp", "logs", "logs/data"]:
        path = os.path.abspath(folder)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

def setup_logger():
    log_filename = os.path.join("logs", "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode="w"),
        ],
    )

if __name__ == "__main__":
    setup_folders()
    setup_logger()
    logging.info("Inicializando interface de treinamento...")
    root = tk.Tk()
    app = TrainingGUI(root)
    app.start()