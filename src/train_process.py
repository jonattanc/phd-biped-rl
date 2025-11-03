# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent, TrainingCallback
from dpg_manager import DPGManager
import utils
import numpy as np
import random
import torch


def process_runner(
    selected_environment,
    environment_settings,
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
    enable_dpg=True,
):
    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Iniciando treinamento: {selected_environment} + {selected_robot} + {algorithm}")
    logger.info(f"Visualização: {enable_visualization_value.value}")
    logger.info(f"Tempo Real: {enable_real_time_value.value}")
    logger.info(f"Câmera: {camera_selection_value.value}")
    logger.info(f"Episódio inicial: {initial_episode}")
    logger.info(f"Modelo carregado: {model_path}")
    logger.info(f"Dynamic Policy Gradient: {enable_dpg}")

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(
            logger,
            robot,
            environment,
            environment_settings,
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

        if enable_dpg:
            dpg_manager = DPGManager(logger, robot, reward_system)
            dpg_manager.enable(True)
            reward_system.set_dpg_manager(dpg_manager)
            logger.info("Sistema DPG configurado e ativado")
        else:
            logger.info("Usando sistema de recompensa padrão (sem DPG)")

        agent = Agent(logger, env=sim, model_path=model_path, algorithm=algorithm, device=device, initial_episode=initial_episode, seed=seed)
        sim.set_agent(agent)

        callback = TrainingCallback(logger)

        # Iniciar treinamento
        logger.info(f"Iniciando treinamento {algorithm} no episódio {initial_episode}...")

        # Loop principal do treinamento
        timesteps_completed = 0
        timesteps_batch_size = 1000

        sim.pre_fill_buffer()

        while not exit_value.value:
            timesteps_completed += timesteps_batch_size
            agent.model.learn(total_timesteps=timesteps_batch_size, reset_num_timesteps=False, callback=callback)

            # Enviar progresso para GUI
            if timesteps_completed % 10000 == 0:
                logger.info(f"Progresso: {timesteps_completed} timesteps com aprendizagem")

        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    ipc_queue.put({"type": "done"})
