# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent
import utils


def process_runner(selected_environment, selected_robot, pause_value, exit_value, enable_real_time_value):
    logger = utils.setup_logger([selected_environment, selected_robot])

    environment = Environment(name=selected_environment)
    robot = Robot(name=selected_robot)
    sim = Simulation(robot, environment, pause_value, exit_value, enable_real_time_value, num_episodes=200)
    agent = Agent(sim)

    agent.train(total_timesteps=100_000)
