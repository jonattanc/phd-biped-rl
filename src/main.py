import os
import shutil
import logging
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


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("logs", "log.txt"), mode="w"),
        ],
    )

    return logging.getLogger(__name__)


if __name__ == "__main__":
    setup_folders()
    logger = setup_logger()

    enable_gui = True

    initial_base_position = [0, 0, 0.7]  # x, y, z
    initial_base_orientation = [0, 0, 0]  # Euler angles: x roll, y pitch, z yaw

    selected_robot = "robot_stage1"
    robot = Robot(name=selected_robot, base_position=initial_base_position, base_orientation=initial_base_orientation)

    selected_environment = "PR"  # Plano Regular
    environment = Environment(name=selected_environment)

    agent = Agent()

    sim = Simulation(robot, environment, agent, enable_gui=enable_gui)
    sim.setup()
    sim.run()
