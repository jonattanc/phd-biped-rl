# evaluate_model.py
import os
import logging
from agent import Agent
import metrics_saver
import multiprocessing
import utils


def evaluate_and_save(model_path, seed, environment, robot, num_episodes, deterministic, enable_visualization):
    """Avalia um modelo e salva as métricas"""

    logging.info(f"Avaliando {robot} no circuito {environment}...")

    if not os.path.exists(model_path):
        logging.error(f"Arquivo do modelo não existe: {model_path}")
        return None

    from simulation import Simulation
    from robot import Robot
    from environment import Environment

    pause_val = multiprocessing.Value("b", 0)
    exit_val = multiprocessing.Value("b", 0)
    realtime_val = multiprocessing.Value("b", 0)
    enable_visualization_val = multiprocessing.Value("b", enable_visualization)
    camera_selection_val = multiprocessing.Value("i", 0)
    config_changed_value = multiprocessing.Value("i", 0)
    env = None
    ipc_queue = multiprocessing.Queue()

    try:
        # Criar ambiente de avaliação
        logger = utils.get_logger(["evaluation", environment, robot])
        robot = Robot(logger, name=robot)
        env_obj = Environment(logger, name=environment)
        env = Simulation(logger, robot, env_obj, None, None, ipc_queue, None, pause_val, exit_val, enable_visualization_val, realtime_val, camera_selection_val, config_changed_value, num_episodes)
        agent = Agent(logger, env=env, model_path=model_path)  # TODO: Passar algoritmo

        # Configurar o agente
        env.set_agent(agent)

        metrics = agent.evaluate(env, num_episodes=num_episodes)

        if metrics is None:
            logging.error("Falha ao gerar métricas de avaliação")
            return None

        if "total_times" not in metrics or not metrics["total_times"]:
            logging.error("Nenhum dado de episódio foi coletado")
            return None

        logging.info(f"Métricas obtidas: {len(metrics.get('total_times', []))} episódios")
        logging.info(f"Tempo médio: {metrics.get('avg_time', 0):.2f}s")
        logging.info(f"Taxa de sucesso: {metrics.get('success_rate', 0)*100:.1f}%")

        hyperparams = {"algorithm": agent.algorithm, "num_episodes": num_episodes, "seed": seed, "deterministic": deterministic, "model_path": model_path, "enable_visualization": enable_visualization}

        os.makedirs("logs/data", exist_ok=True)
        saved_files = metrics_saver.save_complexity_metrics(metrics=metrics, environment=environment, robot=robot, role=role, seed=seed, hyperparams=hyperparams)

        if saved_files and saved_files[0] is not None:
            logging.info(f"CSV salvo com sucesso: {saved_files[0]}")
        else:
            logging.error("Falha ao salvar o CSV")

        logging.info(f"Avaliação concluída com sucesso!")
        return metrics

    except Exception as e:
        logging.exception("Erro crítico na avaliação")
        return None
    finally:
        if env is not None:
            env.close()
