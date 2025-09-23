# train_process.py
import logging
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent
import utils


def process_runner(selected_environment, selected_robot, algorithm, ipc_queue, data_queue, pause_value, exit_value, enable_real_time_value):
    """Função executada no processo separado para treinamento real"""

    logger = utils.get_logger([selected_environment, selected_robot, algorithm])
    logger.info(f"Iniciando treinamento real: {selected_environment} + {selected_robot} + {algorithm}")

    try:
        # Callback para enviar dados para a GUI
        class DataCallback:
            def __init__(self, data_queue):
                self.data_queue = data_queue
                self.episode_count = 0

            def on_episode_end(self, episode_info):
                self.episode_count += 1
                self.data_queue.put(
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
                    self.data_queue.put({"type": "log", "message": f"Episódio {self.episode_count} concluído"})

        data_callback = DataCallback(data_queue)

        # Criar componentes
        environment = Environment(name=selected_environment)
        robot = Robot(name=selected_robot)
        sim = Simulation(robot, environment, pause_value, exit_value, enable_real_time_value, enable_gui=False)
        agent = Agent(env=sim, algorithm=algorithm, data_callback=data_callback)

        # Iniciar treinamento
        ipc_queue.put("training_started")
        data_queue.put({"type": "log", "message": "Iniciando treinamento PPO..."})

        agent.train(total_timesteps=100000)

        ipc_queue.put("done")
        data_queue.put({"type": "log", "message": "Treinamento concluído!"})

    except Exception as e:
        error_msg = f"Erro: {str(e)}"
        logger.error(error_msg)
        ipc_queue.put(f"error: {error_msg}")
