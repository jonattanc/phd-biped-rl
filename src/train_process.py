# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent, TrainingCallback
import metrics_saver
import utils
import numpy as np
import random
import torch
from datetime import datetime
import os
import json
from dataclasses import asdict


def process_runner(
    selected_environment,
    selected_robot,
    algorithm,
    ipc_queue,
    ipc_queue_main_to_process,
    reward_system,
    pause_value,
    exit_value,
    enable_visualization_value,
    enable_real_time_value,
    camera_selection_value,
    config_changed_value,
    seed,
    device="cpu",
    initial_episode=0,
    model_path=None,
    episodes=1000,
    deterministic=False,
):
    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Iniciando simulação: {selected_environment} + {selected_robot} + {algorithm}")
    logger.info(f"Visualização: {enable_visualization_value.value}")
    logger.info(f"Tempo Real: {enable_real_time_value.value}")
    logger.info(f"Câmera: {camera_selection_value.value}")
    logger.info(f"Episódio inicial: {initial_episode}")
    logger.info(f"Modelo carregado: {model_path}")

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Criar componentes
        robot = Robot(logger, name=selected_robot)
        environment = Environment(logger, name=selected_environment, robot=robot)
        sim = Simulation(
            logger,
            robot,
            environment,
            reward_system,
            ipc_queue,
            ipc_queue_main_to_process,
            pause_value,
            exit_value,
            enable_visualization_value,
            enable_real_time_value,
            camera_selection_value,
            config_changed_value,
            initial_episode=initial_episode,
        )

        agent = Agent(logger, env=sim, model_path=model_path, algorithm=algorithm, device=device, initial_episode=initial_episode, seed=seed)
        sim.set_agent(agent)

        callback = TrainingCallback(logger)

        if algorithm is None:
            logger.info("Modo de avaliação")
            metrics_data = sim.evaluate(episodes, deterministic)

            if exit_value.value:
                ipc_queue.put({"type": "done"})
                return

            metrics_data = metrics_saver.calculate_extra_metrics(metrics_data)

            metrics_data["hyperparameters"] = {
                "selected_environment": selected_environment,
                "selected_robot": selected_robot,
                "algorithm": algorithm,
                "reward_system_components": {key: asdict(value) for key, value in reward_system.components.items()},
                "seed": seed,
                "device": device,
                "initial_episode": initial_episode,
                "model_path": model_path,
                "episodes": episodes,
                "deterministic": deterministic,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            metrics_path = os.path.join(utils.TEMP_EVALUATION_SAVE_PATH, f"evaluation_metrics_{os.getpid()}.json")
            metrics_serializebla_data = utils.make_serializable(metrics_data)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_serializebla_data, f, indent=4, ensure_ascii=False)

            ipc_queue.put({"type": "evaluation_complete", "metrics_path": metrics_path})

        else:
            logger.info("Modo de treinamento")

            timesteps_completed = 0
            timesteps_batch_size = 1000

            sim.pre_fill_buffer()

            while not exit_value.value:
                timesteps_completed += timesteps_batch_size
                
                # Treinamento normal do agente
                agent.model.learn(total_timesteps=timesteps_batch_size, reset_num_timesteps=False, callback=callback)

                # Enviar progresso para GUI
                if timesteps_completed % 10000 == 0:
                    logger.info(f"Progresso: {timesteps_completed} timesteps com aprendizagem")

            logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    finally:
        ipc_queue.put({"type": "done"})
