import pybullet as p
import pybullet_data
import time
import math
import logging


class Simulation:
    def __init__(self, robot, plane, enable_gui=True, enable_real_time=False):
        self.robot = robot
        self.plane = plane
        self.enable_gui = enable_gui
        self.enable_real_time = enable_real_time
        self.logger = logging.getLogger(__name__)
        self.physics_client = None
        self.plane_id = None

    def setup(self):
        if self.enable_gui:
            self.physics_client = p.connect(p.GUI)

        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.807)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Sets path to search for files
        self.plane_id = p.loadURDF(self.plane, useFixedBase=True)  # Alterei para fixar o plano
        self.robot.load_in_simulation()

    def run(self, steps=2000):
        if self.enable_real_time:
            p.setRealTimeSimulation(1)

        for i in range(steps):
            if not self.enable_real_time:
                p.stepSimulation()  # Default period 1/240 s. Check setTimeStep and setPhysicsEngineParameter

            time.sleep(1 / 240.0)

            velocity = -15 * math.sin(i / 50)

            # p.setJointMotorControl2(self.robot.get_body_id(), jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=velocity)
            # p.setJointMotorControl2(self.robot.get_body_id(), jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=velocity)

            p.setJointMotorControlArray(
                self.robot.get_body_id(),
                jointIndices=[0, 1],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[velocity, velocity],
            )

            # self.robot.reset_base_position_and_orientation()

            if i % 100 == 0:
                state = p.getJointState(self.robot.get_body_id(), jointIndex=0)
                self.logger.info(f"Step {i}: position={state[0]:.4f}, velocity={state[1]:.4f}")

        body_position, body_orientation = p.getBasePositionAndOrientation(self.robot.get_body_id())
        self.logger.info(f"Final body_position: {body_position}")
        self.logger.info(f"Final body_orientation: {p.getEulerFromQuaternion(body_orientation)}")
        p.disconnect()
