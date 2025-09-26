# evaluate_model.py
import os
import logging
from agent import Agent
from metrics_saver import save_complexity_metrics, compile_results, generate_report
import multiprocessing
import utils


def evaluate_and_save(model_path, circuit_name="PR", avatar_name="robot_stage1", role="AE", num_episodes=5, seed=42):
    """Avalia um modelo e salva as métricas"""

    logging.info(f"Avaliando {avatar_name} no circuito {circuit_name}...")

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
        robot = Robot(avatar_name)
        env_obj = Environment(circuit_name)
        env = Simulation(robot, env_obj, pause_val, exit_val, realtime_val, num_episodes=num_episodes, seed=seed)
        agent = Agent(model_path=model_path)
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

        hyperparams = {"algorithm": "PPO", "num_episodes": num_episodes, "seed": seed}

        # SALVAR MÉTRICAS - garantir que o diretório existe
        os.makedirs("logs/data", exist_ok=True)
        if not os.path.exists("logs/data"):
            logging.error("Não foi possível criar o diretório logs/data")
            return None

        saved_files = save_complexity_metrics(metrics=metrics, circuit_name=circuit_name, avatar_name=avatar_name, role=role, seed=seed, hyperparams=hyperparams)

        if saved_files and saved_files[0] is not None:
            logging.info(f"CSV salvo com sucesso: {saved_files[0]}")
            # Verificar se o arquivo realmente existe
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
    models_dir = "logs/data/models"
    if not os.path.exists(models_dir):
        logging.error(f"Diretório de modelos não encontrado: {models_dir}")
        logging.info("Criando diretório...")
        os.makedirs(models_dir, exist_ok=True)
        logging.info("Por favor, coloque um modelo .zip no diretório e execute novamente")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    if not model_files:
        logging.error(f"Nenhum modelo .zip encontrado em {models_dir}")
        logging.info("Criando um modelo de teste...")

        # Criar um modelo simples de teste se não existir
        try:
            from simulation import Simulation
            from robot import Robot
            from environment import Environment
            from agent import Agent

            robot = Robot("robot_stage1")
            env_obj = Environment("PR")

            pause_val = multiprocessing.Value("b", 0)
            exit_val = multiprocessing.Value("b", 0)
            realtime_val = multiprocessing.Value("b", 0)

            env = Simulation(robot, env_obj, pause_val, exit_val, realtime_val)
            agent = Agent(env=env, algorithm="PPO")

            # Salvar modelo de teste
            test_model_path = os.path.join(models_dir, "test_model.zip")
            agent.model.save(test_model_path)
            logging.info(f"Modelo de teste criado: {test_model_path}")
            model_files = ["test_model.zip"]

        except Exception as e:
            logging.error(f"Erro ao criar modelo de teste: {e}")
            return

    # Avaliar o primeiro modelo encontrado
    model_path = os.path.join(models_dir, model_files[0])
    logging.info(f"Avaliando modelo: {model_files[0]}")

    metrics = evaluate_and_save(model_path, num_episodes=3)  # Reduzir para teste rápido

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
