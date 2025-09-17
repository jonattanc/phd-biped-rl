# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent
import utils


def process_runner(selected_environment, selected_robot):
    logger = utils.setup_logger([selected_environment, selected_robot])

    environment = Environment(name=selected_environment)
    robot = Robot(name=selected_robot)
    agent = Agent()

    sim = Simulation(robot, environment, agent, enable_gui=True, num_episodes=200)

    sim.setup()
    sim.run()
