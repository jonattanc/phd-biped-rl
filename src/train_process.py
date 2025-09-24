# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent
import utils


def process_runner(selected_environment, selected_robot, algorithm, ipc_queue, pause_value, exit_value, enable_real_time_value):
    """Função executada no processo separado para treinamento real"""

    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Iniciando treinamento real: {selected_environment} + {selected_robot} + {algorithm}")

    try:
        # Callback para enviar dados para a GUI
        class DataCallback:
            def __init__(self, ipc_queue):
                self.ipc_queue = ipc_queue
                self.episode_count = 0

            def on_episode_end(self, episode_info):
                self.episode_count += 1
                self.ipc_queue.put(
                    {
                        "type": "episode_data",
                        "episode": self.episode_count,
                        "reward": float(episode_info.get("reward", 0)),
                        "time": float(episode_info.get("time", 0)),
                        "distance": float(episode_info.get("distance", 0)),
                        "success": bool(episode_info.get("success", False)),
                    }
                )

                if self.episode_count % 10 == 0:
                    logger.info(f"Episódio {self.episode_count} concluído")

        data_callback = DataCallback(ipc_queue)

        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(logger, robot, environment, pause_value, exit_value, enable_real_time_value, enable_gui=False)
        agent = Agent(logger, env=sim, algorithm=algorithm, data_callback=data_callback)

        # Iniciar treinamento
        logger.info("Iniciando treinamento {algorithm}...")

        agent.train(total_timesteps=100_000)
        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    ipc_queue.put({"type": "done"})
