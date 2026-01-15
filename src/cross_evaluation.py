# cross_evaluation.py
import multiprocessing
import os
import json
import time
import pandas as pd
from datetime import datetime
import numpy as np
import utils


class CrossEvaluation:
    def __init__(self, logger, models_directory):
        self.logger = logger
        self.models_directory = models_directory
        self.complexity_order = ["PR", "PBA", "PRA", "PRD", "PG", "PRB"]
        self.session_output_dir = utils.CRUZADA_DATA_PATH
        os.makedirs(self.session_output_dir, exist_ok=True)
        os.makedirs(utils.TEMP_EVALUATION_SAVE_PATH, exist_ok=True)

    def run_complete_evaluation(self, num_episodes=100, deterministic=True):
        """Executa avaliação cruzada focando em generalização e direcionalidade"""
        self.logger.info("=" * 50)
        self.logger.info("AVALIAÇÃO CRUZADA - GENERALIZAÇÃO")

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_generalization_results = []

        # Para cada modelo de origem
        for origin_circuit in self.complexity_order:
            self.logger.info(f"MODELO ORIGEM: {origin_circuit}")

            # Dados deste modelo
            model_data = {
                "model_name": origin_circuit,
                "transfer_data": {}
            }

            # Encontrar modelo
            model_path = self._find_specialist_model(origin_circuit)
            if not model_path:
                self.logger.warning(f"Modelo {origin_circuit} não encontrado")
                continue

            # Avaliar em TODOS os outros circuitos
            for target_circuit in self.complexity_order:
                if target_circuit == origin_circuit:
                    continue  # Pular auto-avaliação

                self.logger.info(f"  → {target_circuit}")

                # Avaliar transferência
                transfer_metrics = self._evaluate_for_cross_analysis(
                    model_path, target_circuit, "robot_stage5", num_episodes, deterministic
                )

                # Armazenar dados
                model_data["transfer_data"][target_circuit] = {
                    "metrics": transfer_metrics,
                    "temp_data_path": transfer_metrics.get("temp_data_path")
                }

                # Adicionar aos resultados
                all_generalization_results.append({
                    "origin_circuit": origin_circuit,
                    "target_circuit": target_circuit,
                    "model_path": model_path,
                    "avg_time_target": transfer_metrics["avg_time"],
                    "success_rate_target": transfer_metrics["success_rate"],
                    "avg_distance_target": transfer_metrics["avg_distance"],
                    "avg_velocity_target": transfer_metrics["avg_velocity"]
                })

            # Gerar matrizes para este modelo IMEDIATAMENTE
            self.logger.info(f"\nGerando matrizes para {origin_circuit}...")
            self._generate_generalization_matrices(model_data, timestamp)

        # Classificação Direcional (RF-11)
        self.logger.info("\nClassificando direcionalidade das transferências...")
        directional_results = self._classify_directional_transfers(all_generalization_results)

        # Gerar relatório simplificado
        self.logger.info("\nGerando relatório consolidado...")
        report = self._generate_simplified_report(all_generalization_results, directional_results, num_episodes)

        # Exportar relatórios essenciais
        self._export_essential_reports(report)

        elapsed_time = time.time() - start_time
        self.logger.info("AVALIAÇÃO DE GENERALIZAÇÃO CONCLUÍDA!")
        self.logger.info(f"Tempo total: {elapsed_time:.1f}s")
        self.logger.info(f"Total de transferências avaliadas: {len(all_generalization_results)}")
        self.logger.info("=" * 60)

        return report

    def _classify_directional_transfers(self, generalization_results):
        """RF-11: Classifica transferências como ascendentes/descendentes"""
        directional_results = []

        for transfer in generalization_results:
            origin_idx = self.complexity_order.index(transfer["origin_circuit"])
            target_idx = self.complexity_order.index(transfer["target_circuit"])

            if target_idx > origin_idx:
                direction = "ascendente"
                difficulty = "mais complexo"
            else:
                direction = "descendente"
                difficulty = "menos complexo"

            transfer["direction"] = direction
            transfer["difficulty"] = difficulty
            directional_results.append(transfer)

        return directional_results

    def _generate_simplified_report(self, generalization_results, directional_results, num_episodes):
        """Gera relatório focado apenas em generalização e direcionalidade"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            "metadata": {
                "timestamp": timestamp,
                "num_episodes": num_episodes,
                "total_transfers": len(generalization_results),
                "circuits_order": self.complexity_order
            },
            "generalization_results": generalization_results,
            "directional_analysis": directional_results,
            "summary": {
                "ascendente_count": sum(1 for r in directional_results if r.get("direction") == "ascendente"),
                "descendente_count": sum(1 for r in directional_results if r.get("direction") == "descendente")
            }
        }

    def _export_essential_reports(self, report):
        """Exporta apenas os relatórios essenciais"""
        timestamp = report["metadata"]["timestamp"]
        
        # CSV de todas as transferências (generalização)
        df_all = pd.DataFrame(report["generalization_results"])
        all_path = os.path.join(self.session_output_dir, f"generalizacao_{timestamp}.csv")
        df_all.to_csv(all_path, index=False, encoding='utf-8-sig')
        
        # CSV de direcionalidade
        df_dir = pd.DataFrame(report["directional_analysis"])
        dir_path = os.path.join(self.session_output_dir, f"direcionalidade_{timestamp}.csv")
        df_dir.to_csv(dir_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"Relatórios exportados: {all_path}, {dir_path}")
    
    def _find_specialist_model(self, circuit_name):
        """Encontra modelo especialista para um circuito"""
        # Padrão de nome esperado: PR_model.zip, etc.
        possible_patterns = [f"{circuit_name}_model.zip"]

        for pattern in possible_patterns:
            model_path = os.path.join(self.models_directory, pattern)
            if os.path.exists(model_path):
                return model_path

        # Buscar recursivamente
        for root, dirs, files in os.walk(self.models_directory):
            for file in files:
                if circuit_name in file and file.endswith(".zip"):
                    return os.path.join(root, file)

        return None

    def _evaluate_for_cross_analysis(self, model_path, circuit_name, robot_name, num_episodes, deterministic):
        """Função específica para avaliação cruzada"""
        try:
            # Configurar valores para avaliação
            pause_val = multiprocessing.Value("b", 0)
            exit_val = multiprocessing.Value("b", 0)
            enable_visualization_val = multiprocessing.Value("b", 0)  # Sem visualização
            enable_real_time_val = multiprocessing.Value("b", 0)
            camera_selection_val = multiprocessing.Value("i", 0)
            config_changed_value = multiprocessing.Value("i", 0)

            # Criar filas IPC
            ipc_queue = multiprocessing.Queue()
            ipc_queue_main_to_process = multiprocessing.Queue()

            # Importar aqui para evitar dependências circulares
            from reward_system import RewardSystem
            reward_system = RewardSystem(self.logger)

            eval_logger = self.logger

            # Criar e executar processo de avaliação
            from train_process import process_runner

            # Configurar processo
            eval_process = multiprocessing.Process(
                target=process_runner,
                args=(
                    circuit_name,  # selected_environment
                    robot_name,    # selected_robot
                    False,         # is_fast_td3
                    None,          # algorithm (None para avaliação)
                    ipc_queue,
                    ipc_queue_main_to_process,
                    reward_system,
                    pause_val,
                    exit_val,
                    enable_visualization_val,
                    enable_real_time_val,
                    camera_selection_val,
                    config_changed_value,
                    42,            # seed
                    "cpu",         # device
                    0,             # initial_episode
                    model_path,    # model_path
                    num_episodes,  # episodes
                    deterministic,  # deterministic
                    True,          # cross_evaluation
                ),
                daemon=True
            )

            # Iniciar processo
            eval_logger.info(f"Iniciando avaliação cruzada: {circuit_name} com {robot_name}")
            eval_process.start()

            # Coletar resultados
            metrics_data = None
            metrics_path = None

            # Aguardar conclusão (timeout de 2 minutos)
            timeout = time.time() + (num_episodes * 30)
            while eval_process.is_alive() and time.time() < timeout:
                try:
                    msg = ipc_queue.get(timeout=0.5)
                    if msg.get("type") == "evaluation_complete":
                        metrics_path = msg.get("metrics_path")
                        break
                except:
                    continue
                
            # Forçar término se demorar muito
            if eval_process.is_alive():
                eval_logger.warning(f"Avaliação de {circuit_name} excedeu o timeout, terminando...")
                exit_val.value = True
                eval_process.join(timeout=5)

            # Aguardar processo terminar normalmente
            eval_process.join(timeout=10)

            temp_data_path = None
            if metrics_path and os.path.exists(metrics_path):
                # Copiar arquivo de métricas para o diretório temporário
                temp_data_path = metrics_path  

                # Carregar métricas para processamento
                with open(temp_data_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)

                # Extrair métricas relevantes
                if "episodes" in metrics_data:
                    total_times = []
                    success_count = 0
                    total_distance = 0
                    total_velocity = 0

                    for episode_num, episode_data in metrics_data["episodes"].items():
                        if "episode_data" in episode_data:
                            ep_data = episode_data["episode_data"]
                            episode_time = ep_data.get("time", ep_data.get("times", 0))
                            episode_distance = ep_data.get("distance", ep_data.get("distances", 0))

                            # Calcular velocidade
                            episode_velocity = episode_distance / episode_time if episode_time > 0 else 0

                            if episode_time > 0:  # Filtrar episódios com tempo zero
                                total_times.append(episode_time)
                                total_distance += episode_distance
                                total_velocity += episode_velocity
                                if ep_data.get("success", False):
                                    success_count += 1

                    if total_times:
                        avg_time = np.mean(total_times)
                        std_time = np.std(total_times)
                        success_rate = success_count / len(total_times)
                        avg_distance = total_distance / len(total_times)
                        avg_velocity = total_velocity / len(total_times)
                    else:
                        avg_time = 0
                        std_time = 0
                        success_rate = 0
                        avg_distance = 0
                        avg_velocity = 0

                    eval_logger.info(f"Avaliação concluída: Tempo médio={avg_time:.2f}s, Distância={avg_distance:.2f}m, Sucesso={success_rate:.1%}")

                    return {
                        "avg_time": avg_time,
                        "std_time": std_time,
                        "success_rate": success_rate,
                        "success_count": success_count,
                        "avg_distance": avg_distance,
                        "avg_velocity": avg_velocity,
                        "num_episodes": len(total_times),
                        "total_times": total_times,
                        "temp_data_path": temp_data_path,  
                        "expected_episodes": num_episodes
                    }

            eval_logger.warning(f"Não foi possível obter métricas para {circuit_name}")
            return self._create_default_metrics()

        except Exception as e:
            self.logger.error(f"Erro na avaliação cruzada: {e}")
            return self._create_default_metrics()
         
    def _create_default_metrics(self):
        """Métricas padrão em caso de erro"""
        return {"avg_time": 0, "std_time": 0, "success_rate": 0, "success_count": 0, "avg_distance": 0, "avg_velocity": 0,"num_episodes": 0, "total_times": [], "temp_data_path": None }

    def _generate_generalization_matrices(self, model_data, timestamp):
        """Gera matrizes de Tempo, Distância e Velocidade para um modelo"""
        try:
            model_name = model_data["model_name"]
            transfer_data = model_data["transfer_data"]

            # Coletar episódios de cada circuito alvo
            episodes_by_target = {}

            for target_circuit, data in transfer_data.items():
                temp_path = data.get("temp_data_path")
                if temp_path and os.path.exists(temp_path):
                    episodes = self._extract_episodes_for_matrix(temp_path, target_circuit)
                    episodes_by_target[target_circuit] = episodes

            if not episodes_by_target:
                self.logger.warning(f"Nenhum dado para matrizes do modelo {model_name}")
                return

            # Circuitos alvo na ordem de complexidade
            targets_ordered = [c for c in self.complexity_order if c != model_name and c in episodes_by_target]

            # Número máximo de episódios
            max_episodes = max(len(eps) for eps in episodes_by_target.values())

            # Construir matrizes
            tempo_matrix = []
            distancia_matrix = []
            velocidade_matrix = []

            for ep_idx in range(max_episodes):
                tempo_row = []
                distancia_row = []
                velocidade_row = []

                for target in targets_ordered:
                    episodes = episodes_by_target[target]

                    if ep_idx < len(episodes):
                        ep = episodes[ep_idx]
                        tempo_row.append(ep["Tempo_s"])
                        distancia_row.append(ep["Distância_m"])
                        velocidade_row.append(ep["Velocidade_ms"])
                    else:
                        tempo_row.append(np.nan)
                        distancia_row.append(np.nan)
                        velocidade_row.append(np.nan)

                tempo_matrix.append(tempo_row)
                distancia_matrix.append(distancia_row)
                velocidade_matrix.append(velocidade_row)

            # Criar DataFrames
            df_tempo = pd.DataFrame(tempo_matrix, columns=targets_ordered)
            df_distancia = pd.DataFrame(distancia_matrix, columns=targets_ordered)
            df_velocidade = pd.DataFrame(velocidade_matrix, columns=targets_ordered)

            # Adicionar coluna de episódio
            df_tempo.insert(0, "Episódio", range(1, len(df_tempo) + 1))
            df_distancia.insert(0, "Episódio", range(1, len(df_distancia) + 1))
            df_velocidade.insert(0, "Episódio", range(1, len(df_velocidade) + 1))

            # Salvar
            model_dir = os.path.join(self.session_output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Nomes descritivos
            tempo_path = os.path.join(model_dir, f"Tempo_{model_name}_{timestamp}.csv")
            distancia_path = os.path.join(model_dir, f"Distancia_{model_name}_{timestamp}.csv")
            velocidade_path = os.path.join(model_dir, f"Velocidade_{model_name}_{timestamp}.csv")

            df_tempo.to_csv(tempo_path, index=False, encoding='utf-8-sig', na_rep='-')
            df_distancia.to_csv(distancia_path, index=False, encoding='utf-8-sig', na_rep='-')
            df_velocidade.to_csv(velocidade_path, index=False, encoding='utf-8-sig', na_rep='-')

            self.logger.info(f"  ✓ Matrizes salvas em: {model_dir}")

        except Exception as e:
            self.logger.error(f"Erro ao gerar matrizes para {model_name}: {e}")

    def _extract_episodes_for_matrix(self, file_path, target_circuit):
        """Extrai episódios de um arquivo para a matriz"""
        episodes = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "episodes" in data:
                for ep_num, ep_info in data["episodes"].items():
                    if "episode_data" in ep_info:
                        ep_data = ep_info["episode_data"]
                        tempo = ep_data.get("time", ep_data.get("times", 0))
                        distancia = ep_data.get("distance", ep_data.get("distances", 0))

                        if tempo > 0:  # Só processar episódios válidos
                            velocidade = distancia / tempo if tempo > 0 else 0
                            episodes.append({
                                "Tempo_s": round(tempo, 2),
                                "Distância_m": round(distancia, 2),
                                "Velocidade_ms": round(velocidade, 2)
                            })
        except Exception as e:
            self.logger.warning(f"Erro ao extrair episódios de {file_path}: {e}")
        
        return episodes
    
    def export_report(self, report, output_dir=None):
        """Método de compatibilidade para exportar relatório"""
        if output_dir is None:
            output_dir = self.session_output_dir

        timestamp = report["metadata"]["timestamp"]

        # CSV já foram exportados em _export_essential_reports
        generalization_csv = os.path.join(output_dir, f"generalizacao_{timestamp}.csv")
        directionality_csv = os.path.join(output_dir, f"direcionalidade_{timestamp}.csv")

        # Criar um JSON resumido também
        json_path = os.path.join(output_dir, f"resumo_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": report["metadata"],
                "summary": report["summary"],
                "csv_files": {
                    "generalization": generalization_csv,
                    "directionality": directionality_csv
                }
            }, f, indent=2, ensure_ascii=False)

        return {
            "generalization_csv": generalization_csv,
            "directionality_csv": directionality_csv,
            "json_path": json_path,
            "output_dir": output_dir,
            "timestamp": timestamp
        }