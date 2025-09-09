import os
import shutil
import logging
import multiprocessing
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent


def setup_folder(path):
    path = os.path.abspath(path)

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)


def setup_folders():
    for folder in ["tmp", "logs"]:
        setup_folder(folder)


def setup_logger(env, selected_robot):
    proc = multiprocessing.current_process()
    proc_num = proc._identity[0] if proc._identity else os.getpid()
    log_filename = os.path.join("logs", f"log__{env}__{selected_robot}__proc{proc_num}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode="w"),
        ],
    )

    return logging.getLogger(__name__)


def train_in_process(selected_environment, selected_robot, initial_base_position, initial_base_orientation):
    logger = setup_logger(selected_environment, selected_robot)

    environment = Environment(name=selected_environment)
    robot = Robot(name=selected_robot, base_position=initial_base_position, base_orientation=initial_base_orientation)
    agent = Agent()
    sim = Simulation(robot, environment, agent, enable_gui=False)

    sim.setup()
    sim.run()


if __name__ == "__main__":
    setup_folders()

    initial_base_position = [0, 0, 0.7]  # x, y, z
    initial_base_orientation = [0, 0, 0]  # Euler angles: x roll, y pitch, z yaw

    selected_robot = "robot_stage1"

    environment_list = ["PR", "PBA"]

    processes = []

    for selected_environment in environment_list:
        p = multiprocessing.Process(target=train_in_process, args=(selected_environment, selected_robot, initial_base_position, initial_base_orientation))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
