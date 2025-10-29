# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent, TrainingCallback
import utils
import time
import os
import json


def verify_control_files(control_dir, logger, agent, ipc_queue, context):
    """Verifica arquivos de controle e para cada um: salva o modelo, confirma via IPC e remove o arquivo de controle"""
    try:
        control_files = [f for f in os.listdir(control_dir) if f.startswith("save_model_") and f.endswith(".json")]
        control_files.sort()  # Processar em ordem

        for control_file in control_files:
            control_path = os.path.join(control_dir, control_file)
            with open(control_path, "r") as f:
                control_data = json.load(f)

            model_path = control_data.get("model_path")
            if model_path:
                logger.info(f"COMANDO DE SALVAMENTO {context}: {model_path}")

                # SALVAR MODELO
                agent.save_model(model_path)
                logger.info(f"MODELO SALVO {context}: {model_path}")

                # Verificar se foi salvo
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    logger.info(f"ARQUIVO CONFIRMADO: {model_path} ({file_size} bytes)")

                    # Enviar confirmação via IPC
                    ipc_queue.put({"type": "model_saved", "model_path": model_path})

                else:
                    logger.error(f"FALHA: Arquivo não criado: {model_path}")

                # Remover arquivo de controle processado
                try:
                    os.remove(control_path)
                    logger.info(f"Arquivo de controle removido: {control_file}")
                except Exception as e:
                    logger.warning(f"Não foi possível remover: {control_file}")

    except Exception as e:
        logger.exception("Erro ao verificar arquivos de controle")


def process_runner(
    selected_environment,
    selected_robot,
    algorithm,
    ipc_queue,
    reward_system,
    pause_value,
    exit_value,
    enable_visualization_value,
    enable_real_time_value,
    camera_selection_value,
    config_changed_value,
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
        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(
            logger,
            robot,
            environment,
            reward_system,
            ipc_queue,
            pause_value,
            exit_value,
            enable_visualization_value,
            enable_real_time_value,
            camera_selection_value,
            config_changed_value,
            initial_episode=initial_episode,
        )
        
        sim.reward_system.enable_dpg_progression(enable_dpg)
        
        if enable_dpg and hasattr(sim.reward_system, 'gait_phase_dpg'):
            status = sim.reward_system.gait_phase_dpg.get_status()
            logger.info(f"DPG Fases da Marcha - Fase atual: {status['current_phase']}, Velocidade alvo: {status['target_speed']} m/s")
        else:
            logger.info("Usando Agent padrão (sem DPG)")
        
        agent = Agent(logger, env=sim, model_path=model_path, algorithm=algorithm, device=device, initial_episode=initial_episode)

        sim.set_agent(agent)
        callback = TrainingCallback(logger)
        ipc_queue.put_nowait({"type": "minimum_steps_to_save", "minimum_steps_to_save": agent.minimum_steps_to_save})

        # Iniciar treinamento
        logger.info(f"Iniciando treinamento {algorithm} no episódio {initial_episode}...")

        # Loop principal do treinamento
        timesteps_completed = 0
        timesteps_batch_size = 1000

        # Diretório para controle de salvamento
        control_dir = utils.TRAINING_CONTROL_PATH
        os.makedirs(control_dir, exist_ok=True)

        sim.pre_fill_buffer()

        while not exit_value.value:
            verify_control_files(control_dir, logger, agent, ipc_queue, "VIA ARQUIVO")

            while pause_value.value and not exit_value.value:
                time.sleep(0.5)  # Verificar menos frequentemente durante pausa
                verify_control_files(control_dir, logger, agent, ipc_queue, "DURANTE PAUSA")

            if exit_value.value:
                break

            timesteps_completed += timesteps_batch_size
            if enable_dpg:
                agent.learn(total_timesteps=timesteps_batch_size, reset_num_timesteps=False, callback=callback)
            else:
                agent.model.learn(total_timesteps=timesteps_batch_size, reset_num_timesteps=False, callback=callback)

            # Log dos pesos DPG apenas se estiver ativado
            if enable_dpg and timesteps_completed % 5000 == 0:
                weights = agent.get_dpg_weights()
                if weights is not None:
                    logger.info(f"DPG - Pesos atuais: {weights}")

            # Enviar progresso para GUI
            try:
                ipc_queue.put_nowait({"type": "training_progress", "steps_completed": timesteps_completed})
            except Exception as e:
                logger.exception("Erro ao enviar progresso via IPC")

            if timesteps_completed % 10000 == 0:
                logger.info(f"Progresso: {timesteps_completed} timesteps")

        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    ipc_queue.put({"type": "done"})
