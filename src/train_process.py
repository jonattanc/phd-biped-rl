# train_process.py
from robot import Robot
from simulation import Simulation
from environment import Environment
from agent import Agent, TrainingCallback
import utils
import time
import queue
import os
import json


def process_runner(selected_environment, selected_robot, algorithm, ipc_queue, pause_value, exit_value, enable_real_time_value, device="cpu", initial_episode=0):
    """Função executada no processo separado para treinamento real"""

    logger = utils.get_logger([selected_environment, selected_robot, algorithm], ipc_queue)
    logger.info(f"Iniciando treinamento real: {selected_environment} + {selected_robot} + {algorithm}")
    logger.info(f"Tempo Real: {enable_real_time_value.value}")
    logger.info(f"Episódio inicial: {initial_episode}")

    try:
        # Criar componentes
        environment = Environment(logger, name=selected_environment)
        robot = Robot(logger, name=selected_robot)
        sim = Simulation(logger, robot, environment, ipc_queue, pause_value, exit_value, enable_real_time_value)
        agent = Agent(logger, env=sim, algorithm=algorithm, device=device)
        sim.set_agent(agent)
        sim.set_initial_episode(initial_episode)

        # Iniciar treinamento
        logger.info(f"Iniciando treinamento {algorithm}...")

        # Loop principal do treinamento
        total_timesteps = 10_000_000
        timesteps_completed = 0

        # Diretório para controle de salvamento
        control_dir = "training_control"
        os.makedirs(control_dir, exist_ok=True)

        while timesteps_completed < total_timesteps and not exit_value.value:
            # VERIFICAÇÃO DE COMANDOS VIA ARQUIVO
            try:
                # Verificar se há arquivos de controle de salvamento
                control_files = [f for f in os.listdir(control_dir) if f.startswith("save_model_") and f.endswith(".json")]

                for control_file in control_files:
                    control_path = os.path.join(control_dir, control_file)
                    try:
                        with open(control_path, "r") as f:
                            control_data = json.load(f)

                        model_path = control_data.get("model_path")
                        if model_path:
                            logger.info(f"COMANDO DE SALVAMENTO VIA ARQUIVO: {model_path}")

                            # SALVAR MODELO
                            agent.save_model(model_path)
                            logger.info(f"MODELO SALVO: {model_path}")

                            # Verificar se foi salvo
                            if os.path.exists(model_path):
                                file_size = os.path.getsize(model_path)
                                logger.info(f"ARQUIVO CONFIRMADO: {model_path} ({file_size} bytes)")

                                # Enviar confirmação via IPC
                                try:
                                    ipc_queue.put({"type": "model_saved", "model_path": model_path})
                                except:
                                    pass
                            else:
                                logger.error(f"FALHA: Arquivo não criado: {model_path}")

                            # Remover arquivo de controle processado
                            try:
                                os.remove(control_path)
                                logger.info(f"Arquivo de controle removido: {control_file}")
                            except:
                                logger.warning(f"Não foi possível remover: {control_file}")

                    except Exception as e:
                        logger.error(f"Erro ao processar arquivo de controle {control_file}: {e}")

            except Exception as e:
                logger.error(f"Erro ao verificar arquivos de controle: {e}")

            # Verificar pausa
            while pause_value.value and not exit_value.value:
                time.sleep(0.5)  # Verificar menos frequentemente durante pausa

                # Verificar arquivos de controle durante pausa também
                try:
                    control_files = [f for f in os.listdir(control_dir) if f.startswith("save_model_") and f.endswith(".json")]
                    for control_file in control_files:
                        control_path = os.path.join(control_dir, control_file)
                        try:
                            with open(control_path, "r") as f:
                                control_data = json.load(f)

                            model_path = control_data.get("model_path")
                            if model_path:
                                logger.info(f"COMANDO DE SALVAMENTO DURANTE PAUSA: {model_path}")
                                agent.save_model(model_path)
                                logger.info(f"MODELO SALVO DURANTE PAUSA: {model_path}")

                                if os.path.exists(model_path):
                                    try:
                                        ipc_queue.put({"type": "model_saved", "model_path": model_path})
                                    except:
                                        pass

                                try:
                                    os.remove(control_path)
                                except:
                                    pass

                        except Exception as e:
                            logger.error(f"Erro ao processar arquivo durante pausa: {e}")
                except Exception as e:
                    logger.error(f"Erro ao verificar controles durante pausa: {e}")

            if exit_value.value:
                break

            try:
                # Verificar se o ambiente está configurado
                if agent.model.get_env() is None:
                    logger.error("Ambiente não configurado! Configurando...")
                    agent.set_env(sim)

                # Usar agent.model.learn diretamente com parâmetros corretos
                callback = TrainingCallback()
                agent.model.learn(total_timesteps=1000, reset_num_timesteps=False, callback=callback)
                timesteps_completed += 1000
                if timesteps_completed % 10000 == 0:
                    logger.info(f"Progresso: {timesteps_completed}/{total_timesteps} timesteps")
            except Exception as e:
                logger.error(f"Erro durante aprendizado: {e}")
                break

        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner")

    ipc_queue.put({"type": "done"})


def process_runner_resume(selected_environment, selected_robot, algorithm, ipc_queue, pause_value, exit_value, enable_real_time_value, device="cpu", model_path=None, initial_episode=0):
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
        sim = Simulation(logger, robot, environment, ipc_queue, pause_value, exit_value, enable_real_time_value)

        # Carregar modelo existente
        agent = Agent(logger, model_path=model_path, device=device, initial_episode=initial_episode)

        # CONFIGURAR O AMBIENTE NO MODELO CARREGADO
        logger.info("Configurando ambiente no modelo carregado...")
        agent.set_env(sim)
        sim.set_initial_episode(initial_episode)
        sim.set_agent(agent)
        logger.info(f"Retomando treinamento {algorithm}...")

        # Loop principal do treinamento
        total_timesteps = 10_000_000
        timesteps_completed = 0

        # Diretório para controle de salvamento
        control_dir = "training_control"
        os.makedirs(control_dir, exist_ok=True)

        while timesteps_completed < total_timesteps and not exit_value.value:
            try:
                control_files = [f for f in os.listdir(control_dir) if f.startswith("save_model_") and f.endswith(".json")]

                for control_file in control_files:
                    control_path = os.path.join(control_dir, control_file)
                    try:
                        with open(control_path, "r") as f:
                            control_data = json.load(f)

                        save_model_path = control_data.get("model_path")
                        if save_model_path:
                            logger.info(f"COMANDO DE SALVAMENTO VIA ARQUIVO: {save_model_path}")

                            agent.save_model(save_model_path)
                            logger.info(f"MODELO SALVO: {save_model_path}")

                            if os.path.exists(save_model_path):
                                file_size = os.path.getsize(save_model_path)
                                logger.info(f"ARQUIVO CONFIRMADO: {save_model_path} ({file_size} bytes)")

                                try:
                                    ipc_queue.put({"type": "model_saved", "model_path": save_model_path})
                                except:
                                    pass
                            else:
                                logger.error(f"FALHA: Arquivo não criado: {save_model_path}")

                            try:
                                os.remove(control_path)
                                logger.info(f"Arquivo de controle removido: {control_file}")
                            except:
                                logger.warning(f"Não foi possível remover: {control_file}")

                    except Exception as e:
                        logger.error(f"Erro ao processar arquivo de controle {control_file}: {e}")

            except Exception as e:
                logger.error(f"Erro ao verificar arquivos de controle: {e}")

            # Verificar pausa
            while pause_value.value and not exit_value.value:
                time.sleep(0.5)

                try:
                    control_files = [f for f in os.listdir(control_dir) if f.startswith("save_model_") and f.endswith(".json")]
                    for control_file in control_files:
                        control_path = os.path.join(control_dir, control_file)
                        try:
                            with open(control_path, "r") as f:
                                control_data = json.load(f)

                            save_model_path = control_data.get("model_path")
                            if save_model_path:
                                logger.info(f"COMANDO DE SALVAMENTO DURANTE PAUSA: {save_model_path}")
                                agent.save_model(save_model_path)
                                logger.info(f"MODELO SALVO DURANTE PAUSA: {save_model_path}")

                                if os.path.exists(save_model_path):
                                    try:
                                        ipc_queue.put({"type": "model_saved", "model_path": save_model_path})
                                    except:
                                        pass

                                try:
                                    os.remove(control_path)
                                except:
                                    pass

                        except Exception as e:
                            logger.error(f"Erro ao processar arquivo durante pausa: {e}")
                except Exception as e:
                    logger.error(f"Erro ao verificar controles durante pausa: {e}")

            if exit_value.value:
                break

            try:
                # Verificar se o ambiente está configurado
                if agent.model.get_env() is None:
                    logger.error("Ambiente não configurado na retomada! Configurando...")
                    agent.set_env(sim)

                # Usar agent.model.learn diretamente
                callback = TrainingCallback()
                agent.model.learn(total_timesteps=1000, reset_num_timesteps=False, callback=callback)
                timesteps_completed += 1000
                if timesteps_completed % 10000 == 0:
                    logger.info(f"Progresso: {timesteps_completed}/{total_timesteps} timesteps")
            except Exception as e:
                logger.error(f"Erro durante aprendizado: {e}")
                break

        logger.info("Treinamento concluído!")

    except Exception as e:
        logger.exception("Erro em process_runner_resume")

    ipc_queue.put({"type": "done"})


def send_episode_data(ipc_queue, episode_data, episode_offset=0):
    """Envia dados do episódio para a GUI com offset correto"""
    try:
        # Ajustar o número do episódio baseado no offset
        adjusted_episode = episode_data.get("episode", 0) + episode_offset

        ipc_queue.put(
            {
                "type": "episode_data",
                "episode": episode_data.get("episode", 0),  # Episódio relativo
                "adjusted_episode": adjusted_episode,  # Episódio absoluto
                "reward": episode_data.get("reward", 0),
                "time": episode_data.get("time", 0),
                "distance": episode_data.get("distance", 0),
                "imu_x": episode_data.get("imu_x", 0),
                "imu_y": episode_data.get("imu_y", 0),
                "imu_z": episode_data.get("imu_z", 0),
                "roll": episode_data.get("roll", 0),
                "pitch": episode_data.get("pitch", 0),
                "yaw": episode_data.get("yaw", 0),
            }
        )
    except Exception as e:
        logger = utils.get_logger()
        logger.error(f"Erro ao enviar dados do episódio: {e}")
