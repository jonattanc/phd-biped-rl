import os
import shutil
import logging
from robot import Robot
from simulation import Simulation


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

    enable_real_time = True
    enable_gui = True

    initial_base_position = [0, 0, 0.7]  # x, y, z
    initial_base_orientation = [0, 0, 0]  # Euler angles: x roll, y pitch, z yaw
    selected_robot = "robot_minimalist"

    robot = Robot(name=selected_robot, base_position=initial_base_position, base_orientation=initial_base_orientation)

    plane = "plane.urdf"

    sim = Simulation(robot, plane, enable_gui=enable_gui, enable_real_time=enable_real_time)
    sim.setup()
    sim.run()
