# evaluate_model.py (modificações)
import os
import logging
from agent import Agent
from metrics_saver import save_complexity_metrics, compile_results, generate_report
import multiprocessing
import utils


def evaluate_and_save(model_path, circuit_name="PR", avatar_name="robot_stage1", role="AE", num_episodes=5, seed=42, deterministic=True):
    """Avalia um modelo e salva as métricas"""

    logging.info(f"Avaliando {avatar_name} no circuito {circuit_name}...")
    logging.info(f"Modo determinístico: {deterministic}")

    # Importar aqui para evitar conflitos
    from simulation import Simulation
    from robot import Robot
    from environment import Environment

    pause_val = multiprocessing.Value("b", 0)
    exit_val = multiprocessing.Value("b", 0)
    realtime_val = multiprocessing.Value("b", 0)
    env = None

    try:
        # Criar ambiente de avaliação
        logger = utils.get_logger(["evaluation", circuit_name, avatar_name])
        robot = Robot(logger, name=avatar_name)
        env_obj = Environment(logger, name=circuit_name)
        env = Simulation(logger, robot, env_obj, None, pause_val, exit_val, realtime_val, num_episodes=num_episodes, seed=seed)
        agent = Agent(logger, model_path=model_path)

        # Configurar ambiente no agente
        agent.set_env(env)

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

        hyperparams = {"algorithm": agent.algorithm, "num_episodes": num_episodes, "seed": seed, "deterministic": deterministic, "model_path": model_path}

        # SALVAR MÉTRICAS - garantir que o diretório existe
        os.makedirs("logs/data", exist_ok=True)
        if not os.path.exists("logs/data"):
            logging.error("Não foi possível criar o diretório logs/data")
            return None

        saved_files = save_complexity_metrics(metrics=metrics, circuit_name=circuit_name, avatar_name=avatar_name, role=role, seed=seed, hyperparams=hyperparams)

        if saved_files and saved_files[0] is not None:
            logging.info(f"CSV salvo com sucesso: {saved_files[0]}")
            if os.path.exists(saved_files[0]):
                logging.info(f"Arquivo confirmado: {saved_files[0]} (tamanho: {os.path.getsize(saved_files[0])} bytes)")
            else:
                logging.error(f"Arquivo não foi criado: {saved_files[0]}")
        else:
            logging.error("Falha ao salvar o CSV")

        logging.info(f"Avaliação concluída: Sucesso: {metrics['success_rate']*100:.1f}% | Tempo: {metrics['avg_time']:.2f}s")
        return metrics

    except Exception as e:
        logging.error(f"Erro na avaliação: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass


def evaluate_single_model_directly():
    """Função para testar a avaliação diretamente, sem interferência da GUI"""
    logging.info("=== INICIANDO AVALIAÇÃO DIRETA ===")

    # Verificar se existe algum modelo
    models_dir = utils.TRAINING_DATA_PATH
    if not os.path.exists(models_dir):
        logging.error(f"Diretório de modelos não encontrado: {models_dir}")
        logging.info("Criando diretório...")
        os.makedirs(models_dir, exist_ok=True)
        logging.info("Por favor, coloque um modelo .zip no diretório e execute novamente")
        return

    # Buscar modelos recursivamente
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".zip"):
                model_files.append(os.path.join(root, file))

    if not model_files:
        logging.error(f"Nenhum modelo .zip encontrado em {models_dir}")
        logging.info("Execute um treinamento primeiro para gerar um modelo.")
        return

    # Avaliar o primeiro modelo encontrado
    model_path = model_files[0]
    logging.info(f"Avaliando modelo: {os.path.basename(model_path)}")

    metrics = evaluate_and_save(model_path, num_episodes=3)

    if metrics:
        logging.info("=== AVALIAÇÃO CONCLUÍDA COM SUCESSO ===")

        # Compilar resultados
        logging.info("Compilando resultados...")
        compiled_df = compile_results()

        if compiled_df is not None:
            generate_report()
            logging.info("Relatório gerado com sucesso!")
        else:
            logging.warning("Nenhum dado para compilar")
    else:
        logging.error("=== FALHA NA AVALIAÇÃO ===")


def main():
    """Avalia todos os modelos e compila resultados automaticamente"""
    utils.get_logger()

    evaluate_single_model_directly()


if __name__ == "__main__":
    main()
