# main.py
import os
import shutil
import logging
from gym_env import ExoskeletonPRst1  # <-- Importa o novo ambiente
from agent import Agent
from gui import TrainingGUI
import tkinter as tk

def setup_folders():
    for folder in ["tmp", "logs", "logs/data", "logs/tensorboard", "models"]:
        path = os.path.abspath(folder)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

def setup_logger():
    log_filename = os.path.join("logs", "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode="w", encoding='utf-8'),
        ],
    )

if __name__ == "__main__":
    setup_folders()
    setup_logger()
    logging.info("Inicializando interface de treinamento...")

    # Cria o ambiente Gym
    env = ExoskeletonPRst1(enable_gui=False)

    # Cria o agente PPO
    agent = Agent(env=env)

    # Inicia a GUI
    root = tk.Tk()
    app = TrainingGUI(root, agent)  # <-- Passa o agente para a GUI
    app.start()