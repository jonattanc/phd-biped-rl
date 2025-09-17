# main.py
import os
import shutil
import logging
import multiprocessing
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent
from gui import TrainingGUI
import tkinter as tk
import utils


def setup_folders():
    if os.path.exists(utils.TMP_PATH):
        shutil.rmtree(utils.TMP_PATH)
    os.makedirs(utils.TMP_PATH, exist_ok=True)


def setup_logger(description):
    proc = multiprocessing.current_process()
    proc_num = proc._identity[0] if proc._identity else os.getpid()

    log_name = "__".join(description)
    log_filename = os.path.join(utils.PROJECT_ROOT, "logs", f"log__{log_name}__proc{proc_num}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
        ],
    )

    return logging.getLogger(__name__)


def train_in_process(selected_environment, selected_robot):
    logger = setup_logger([selected_environment, selected_robot])

    environment = Environment(name=selected_environment)
    robot = Robot(name=selected_robot)
    agent = Agent()
    sim = Simulation(robot, environment, agent, enable_gui=True)

    sim.setup()
    sim.run()


if __name__ == "__main__":
    setup_folders()
    setup_logger(["main"])

    logging.info(f"Executando em {utils.PROJECT_ROOT}")

    selected_robot = "robot_stage1"
    environment_list = ["PR"]

    processes = []

    for selected_environment in environment_list:
        p = multiprocessing.Process(target=train_in_process, args=(selected_environment, selected_robot))
        p.start()
        processes.append(p)

    # # Inicia a GUI
    # root = tk.Tk()
    # app = TrainingGUI(root, agent)
    # app.start()

    for p in processes:
        p.join()
