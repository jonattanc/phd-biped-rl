# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent
import utils


def process_runner(selected_environment, selected_robot, algorithm, ipc_queue, pause_value, exit_value,  enable_real_time_value, device="cpu"):
    """Função executada no processo separado para treinamento real"""

    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Iniciando treinamento real: {selected_environment} + {selected_robot} + {algorithm}")
    logger.info(f"Tempo Real: {enable_real_time_value.value}")

    try:
        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(logger, robot, environment, ipc_queue, pause_value, exit_value,  enable_real_time_value)
        agent = Agent(logger, env=sim, algorithm=algorithm, device=device)
        sim.set_agent(agent)

        # Iniciar treinamento
        logger.info(f"Iniciando treinamento {algorithm}...")

        agent.train(total_timesteps=10_000_000)
        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    ipc_queue.put({"type": "done"})
