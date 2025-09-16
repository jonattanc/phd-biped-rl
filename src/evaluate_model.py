# evaluate_model.py
import os
import logging
from gym_env import ExoskeletonPRst1
from agent import Agent
from metrics_saver import save_complexity_metrics, compile_results, generate_report

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def evaluate_and_save(model_path, circuit_name="PR", avatar_name="robot_stage1", 
                     role="AE", num_episodes=20, seed=42):
    """Avalia um modelo e salva as métricas"""
    
    logging.info(f"Avaliando {avatar_name} no circuito {circuit_name}...")
    
    env = ExoskeletonPRst1(enable_gui=False, seed=seed)
    
    try:
        agent = Agent(model_path=model_path)
        metrics = agent.evaluate(env, num_episodes=num_episodes)
        
        hyperparams = {
            "algorithm": "PPO"
        }
        
        save_complexity_metrics(
            metrics=metrics,
            circuit_name=circuit_name,
            avatar_name=avatar_name,
            role=role,
            seed=seed,
            hyperparams=hyperparams
        )
        
        logging.info(f"Sucesso: {metrics['success_rate']*100:.1f}% | Tempo: {metrics['avg_time']:.2f}s")
        return metrics
        
    except Exception as e:
        logging.error(f"Erro: {e}")
        return None
    finally:
        env.close()

def main():
    """Avalia todos os modelos e compila resultados automaticamente"""
    setup_logger()
    
    models_dir = "logs/data/models"
    os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(models_dir):
        logging.error("Diretório 'models-dir' não encontrado!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        logging.error("Nenhum modelo .zip encontrado em 'models_dir'")
        return
    
    # Avaliar todos os modelos
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        evaluate_and_save(model_path)
    
    # Compilar resultados automaticamente
    logging.info("Compilando resultados...")
    compiled_df = compile_results()
    
    if compiled_df is not None:
        generate_report()
        logging.info("Relatório gerado com sucesso!")
    else:
        logging.warning("Nenhum dado para compilar")

if __name__ == "__main__":
    main()