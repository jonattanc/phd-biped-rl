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
    selected_environment, selected_robot, algorithm, ipc_queue, pause_value, exit_value, enable_visualization_value, enable_real_time_value, device="cpu", initial_episode=0, reward_config=None
):
    """Função executada no processo separado para treinamento real"""

    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Iniciando treinamento real: {selected_environment} + {selected_robot} + {algorithm}")
    logger.info(f"Visualização: {enable_visualization_value.value}")
    logger.info(f"Tempo Real: {enable_real_time_value.value}")
    logger.info(f"Episódio inicial: {initial_episode}")

    try:
        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(logger, robot, environment, ipc_queue, pause_value, exit_value, enable_visualization_value, enable_real_time_value)
        agent = Agent(logger, env=sim, algorithm=algorithm, device=device)
        sim.set_agent(agent)

        # Iniciar treinamento
        logger.info(f"Iniciando treinamento {algorithm}...")

        # Loop principal do treinamento
        timesteps_completed = 0

        # Diretório para controle de salvamento
        control_dir = utils.TRAINING_CONTROL_PATH
        os.makedirs(control_dir, exist_ok=True)

        while not exit_value.value:
            verify_control_files(control_dir, logger, agent, ipc_queue, "VIA ARQUIVO")

            while pause_value.value and not exit_value.value:
                time.sleep(0.5)  # Verificar menos frequentemente durante pausa
                verify_control_files(control_dir, logger, agent, ipc_queue, "DURANTE PAUSA")

            if exit_value.value:
                break

            try:
                # Verificar se o ambiente está configurado
                if agent.model.get_env() is None:
                    logger.error("Ambiente não configurado! Configurando...")
                    agent.set_env(sim)

                # Usar agent.model.learn diretamente com parâmetros corretos
                callback = TrainingCallback(logger)
                agent.model.learn(total_timesteps=1000, reset_num_timesteps=False, callback=callback)
                timesteps_completed = agent.model.num_timesteps

                # Enviar progresso para GUI
                try:
                    ipc_queue.put_nowait({"type": "training_progress", "steps_completed": timesteps_completed})
                except Exception as e:
                    logger.exception("Erro ao enviar progresso via IPC")

                if timesteps_completed % 10000 == 0:
                    logger.info(f"Progresso: {timesteps_completed} timesteps")

            except Exception as e:
                logger.exception("Erro durante aprendizado")
                break

        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    # CONFIGURAR SISTEMA DE RECOMPENSAS
    if reward_config is not None:
        sim.reward_system.load_configuration(reward_config)
        logger.info("Configuração de recompensas carregada")
    else:
        # Fallback para padrão
        sim.reward_system.load_active_configuration()
        logger.info("Usando configuração padrão de recompensas")

    ipc_queue.put({"type": "done"})


def process_runner_resume(
    selected_environment, selected_robot, algorithm, ipc_queue, pause_value, exit_value, enable_visualization_value, enable_real_time_value, device="cpu", model_path=None, initial_episode=0
):
    """Função executada no processo separado para retomar treinamento"""

    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Retomando treinamento: {selected_environment} + {selected_robot} + {algorithm}")
    logger.info(f"Modelo carregado: {model_path}")
    logger.info(f"Episódio inicial recebido do GUI: {initial_episode}")
    logger.info(f"Tempo Real: {enable_real_time_value.value}")

    try:
        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(logger, robot, environment, ipc_queue, pause_value, exit_value, enable_visualization_value, enable_real_time_value, initial_episode=initial_episode)
        agent = Agent(logger, model_path=model_path, device=device, initial_episode=initial_episode)

        # CONFIGURAR O AMBIENTE NO MODELO CARREGADO
        logger.info("Configurando ambiente no modelo carregado...")
        agent.set_env(sim)
        sim.set_agent(agent)
        logger.info(f"Retomando treinamento {algorithm} do episódio {initial_episode}...")

        # Loop principal do treinamento
        timesteps_completed = 0

        # Diretório para controle de salvamento
        control_dir = utils.TRAINING_CONTROL_PATH
        os.makedirs(control_dir, exist_ok=True)

        while not exit_value.value:
            # VERIFICAÇÃO DE COMANDOS - PROCESSAR IMEDIATAMENTE
            verify_control_files(control_dir, logger, agent, ipc_queue, "VIA ARQUIVO EM RESUME")

            # Verificar pausa
            while pause_value.value and not exit_value.value:
                time.sleep(0.5)
                verify_control_files(control_dir, logger, agent, ipc_queue, "DURANTE PAUSA EM RESUME")

            if exit_value.value:
                break

            try:
                # Verificar se o ambiente está configurado
                if agent.model.get_env() is None:
                    logger.error("Ambiente não configurado na retomada! Configurando...")
                    agent.set_env(sim)

                # Usar agent.model.learn diretamente
                callback = TrainingCallback(logger)
                agent.model.learn(total_timesteps=1000, reset_num_timesteps=False, callback=callback)
                timesteps_completed += 1000
                if timesteps_completed % 10000 == 0:
                    logger.info(f"Progresso: {timesteps_completed} timesteps")
            except Exception as e:
                logger.exception("Erro durante aprendizado")
                break

        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner_resume")

    ipc_queue.put({"type": "done"})
