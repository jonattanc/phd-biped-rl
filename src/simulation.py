import pybullet as p
import pybullet_data
import time
import math
import logging


class Simulation:
    def __init__(self, robot, environment, agent, enable_gui=True):
        self.robot = robot
        self.environment = environment
        self.agent = agent
        self.enable_gui = enable_gui
        self.logger = logging.getLogger(__name__)
        self.physics_client = None

    def setup(self):
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)

        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.807)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Sets path to search for files

        self.environment.load_in_simulation(use_fixed_base=True)
        self.robot.load_in_simulation()
        self.agent.set_revolute_indices(self.robot.revolute_indices)

    def reset(self):
        self.robot.reset_base_position_and_orientation()

    def run(self, num_episodes=3, episode_time_limit_ms=5000):
        for episode in range(num_episodes):
            self.logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            self.reset()
            self.logger.info("Robot and environment reset.")
            self.run_episode(episode_time_limit_ms)

        p.disconnect()

    def run_episode(self, time_limit_ms):
        timestep_s = 1 / 240.0
        timestep_ms = timestep_s * 1000
        steps = int(time_limit_ms / timestep_ms)

        for i in range(steps):
            p.stepSimulation()

            if self.enable_gui:
                time.sleep(timestep_s)

            self.agent.set_state(i)
            velocities = self.agent.get_action()

            p.setJointMotorControlArray(
                self.robot.get_body_id(),
                jointIndices=self.robot.revolute_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=velocities,
            )

            if i % 100 == 0:
                state = p.getJointState(self.robot.get_body_id(), jointIndex=0)
                self.logger.info(f"Step {i}: position={state[0]:.4f}, velocity={state[1]:.4f}")

        body_position, body_orientation = p.getBasePositionAndOrientation(self.robot.get_body_id())
        self.logger.info(f"Final body_position: {body_position}")
        self.logger.info(f"Final body_orientation: {p.getEulerFromQuaternion(body_orientation)}")
