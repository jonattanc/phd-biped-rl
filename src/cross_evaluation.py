# cross_evaluation.py
from email import utils
import multiprocessing
import os
import json
import time
import pandas as pd
from datetime import datetime
import numpy as np
import shutil
import utils


class CrossEvaluation:
    def __init__(self, logger, models_directory):
        self.logger = logger
        self.models_directory = models_directory
        self.complexity_order = ["PR", "PBA", "PRA", "PRD", "PG", "PRB"]
        self.session_output_dir = utils.CRUZADA_DATA_PATH

    def run_complete_evaluation(self, num_episodes=100, deterministic=True):
        """Executa toda a avaliação cruzada automaticamente"""
        self.logger.info("Iniciando avaliação cruzada completa")

        # 1. Avaliação de Complexidade (RF-08)
        complexity_results = self._run_complexity_analysis(num_episodes, deterministic)

        # 2. Avaliação de Generalização (RF-09)
        generalization_results = self._run_generalization_analysis(num_episodes, deterministic)

        # 3. Avaliação de Especificidade (RF-10)
        specificity_results = self._run_specificity_analysis(num_episodes, deterministic)

        # 4. Classificação Direcional (RF-11)
        directional_analysis = self._classify_directional_transfers(generalization_results)

        # 5. Gerar relatório completo
        comprehensive_report = self._generate_comprehensive_report(complexity_results, generalization_results, specificity_results, directional_analysis, num_episodes)

        return comprehensive_report

    def _run_complexity_analysis(self, num_episodes, deterministic):
        """RF-08: Avaliação de Complexidade - 6 circuitos × 100 repetições"""
        complexity_results = []

        for circuit in self.complexity_order:
            model_path = self._find_specialist_model(circuit)
            if not model_path:
                self.logger.warning(f"Modelo especialista não encontrado para {circuit}")
                continue

            self.logger.info(f"Avaliando complexidade: {circuit}")

            # Executar avaliação (integrar com evaluate_model existente)
            metrics = self._evaluate_for_cross_analysis(model_path, circuit, "robot_stage5", num_episodes, deterministic)

            complexity_results.append(
                {"circuit": circuit, "model_path": model_path, "metrics": metrics, "avg_time": metrics["avg_time"], "std_time": metrics["std_time"], "success_rate": metrics["success_rate"]}
            )

        return complexity_results

    def _run_generalization_analysis(self, num_episodes, deterministic):
        """RF-09: Avaliação de Generalização - 6×5×100 execuções"""
        generalization_results = []

        for origin_circuit in self.complexity_order:
            origin_model = self._find_specialist_model(origin_circuit)
            if not origin_model:
                continue

            for target_circuit in self.complexity_order:
                if origin_circuit == target_circuit:
                    continue  # Pular auto-avaliação

                self.logger.info(f"Generalização: {origin_circuit} → {target_circuit}")

                # Avaliar modelo origem no circuito destino
                metrics = self._evaluate_for_cross_analysis(origin_model, target_circuit, "robot_stage5", num_episodes, deterministic)

                generalization_results.append(
                    {
                        "origin_circuit": origin_circuit,
                        "target_circuit": target_circuit,
                        "model_path": origin_model,
                        "metrics": metrics,
                        "avg_time_target": metrics["avg_time"],
                        "success_rate_target": metrics["success_rate"],
                    }
                )

        return generalization_results

    def _run_specificity_analysis(self, num_episodes, deterministic):
        """RF-10: Avaliação de Especificidade - 6×6×100 execuções"""
        specificity_results = []

        for target_circuit in self.complexity_order:
            # Encontrar AE (especialista) deste circuito
            ae_model = self._find_specialist_model(target_circuit)
            if not ae_model:
                continue

            # Avaliar o AE em seu próprio circuito
            ae_metrics = self._evaluate_for_cross_analysis(ae_model, target_circuit, "robot_stage5", num_episodes, deterministic)

            # Avaliar todos os AGs (generalistas) neste circuito
            for origin_circuit in self.complexity_order:
                if origin_circuit == target_circuit:
                    continue  # AE já avaliado

                ag_model = self._find_specialist_model(origin_circuit)
                if not ag_model:
                    continue

                ag_metrics = self._evaluate_for_cross_analysis(ag_model, target_circuit, "robot_stage5", num_episodes, deterministic)

                # Calcular ΔTm_espec
                delta_tm = ag_metrics["avg_time"] - ae_metrics["avg_time"]

                specificity_results.append(
                    {
                        "target_circuit": target_circuit,
                        "origin_circuit": origin_circuit,
                        "ae_avg_time": ae_metrics["avg_time"],
                        "ag_avg_time": ag_metrics["avg_time"],
                        "delta_tm": delta_tm,
                        "ae_success_rate": ae_metrics["success_rate"],
                        "ag_success_rate": ag_metrics["success_rate"],
                    }
                )

        return specificity_results

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
            
            # Salvar cópia dos dados detalhados
            if self.session_output_dir:
                model_name = os.path.basename(model_path).replace('.zip', '')
                detailed_filename = f"{model_name}_{circuit_name}{int(time.time())}.json"
                detailed_path = os.path.join(self.session_output_dir, detailed_filename)

                with open(detailed_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2, ensure_ascii=False)

                self.logger.info(f"Dados detalhados salvos em: {detailed_path}")

                # Carregar métricas
                if metrics_path and os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
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
                                episode_time = ep_data.get("time", 0)
                                episode_distance = ep_data.get("distance", 0)
                                episode_velocity = ep_data.get("velocity", 0)

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

                        eval_logger.info(f"Avaliação concluída: Tempo médio={avg_time:.2f}s, Distância={avg_distance:.2f}m, Velocidade={avg_velocity:.2f}m/s, Sucesso={success_rate:.1%}")

                        return {
                            "avg_time": avg_time,
                            "std_time": std_time,
                            "success_rate": success_rate,
                            "success_count": success_count,
                            "avg_distance": avg_distance,
                            "avg_velocity": avg_velocity,
                            "num_episodes": len(total_times),
                            "total_times": total_times,
                            "raw_data_path": detailed_path if self.session_output_dir else None,
                            "expected_episodes": num_episodes
                        }
            
            eval_logger.warning(f"Não foi possível obter métricas para {circuit_name}")
            return self._create_default_metrics()
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação cruzada: {e}")
            return self._create_default_metrics()
        
    def _create_default_metrics(self):
        """Métricas padrão em caso de erro"""
        return {"avg_time": 0, "std_time": 0, "success_rate": 0, "success_count": 0, "avg_distance": 0, "avg_velocity": 0,"num_episodes": 0, "total_times": []}

    def _generate_comprehensive_report(self, complexity, generalization, specificity, directional, num_episodes):
        """Gera relatório completo da avaliação cruzada"""

        # Calcular ranking de complexidade
        complexity_ranking = self._rank_circuit_complexity(complexity)

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(complexity) + len(generalization) + len(specificity),
                "circuits_tested": self.complexity_order,
                "num_episodes": num_episodes,
            },
            "complexity_ranking": complexity_ranking,
            "best_specialists": self._identify_best_specialists(complexity),
            "generalization_analysis": self._analyze_generalization_patterns(generalization),
            "directional_insights": self._extract_directional_insights(directional),
            "specificity_gaps": self._calculate_specificity_gaps(specificity),
            "raw_data": {"complexity": complexity, "generalization": generalization, "specificity": specificity, "directional": directional},
        }

        return report

    def _rank_circuit_complexity(self, complexity_results):
        """Ranking de complexidade baseado no tempo médio dos AEs"""
        ranked = sorted(complexity_results, key=lambda x: x["avg_time"])
        return [{"circuit": item["circuit"], "avg_time": item["avg_time"]} for item in ranked]

    def _identify_best_specialists(self, complexity_results):
        """Identifica os melhores especialistas por circuito"""
        best_specialists = {}
        for result in complexity_results:
            circuit = result["circuit"]
            best_specialists[circuit] = {"avg_time": result["avg_time"], "success_rate": result["success_rate"], "model_path": result["model_path"]}
        return best_specialists

    def _analyze_generalization_patterns(self, generalization_results):
        """Analisa padrões de generalização"""
        patterns = {}
        for result in generalization_results:
            origin = result["origin_circuit"]
            target = result["target_circuit"]

            if origin not in patterns:
                patterns[origin] = {}

            patterns[origin][target] = {"avg_time": result["avg_time_target"], "success_rate": result["success_rate_target"]}

        return patterns

    def _extract_directional_insights(self, directional_results):
        """Extrai insights das transferências direcionais"""
        ascendente_stats = [r for r in directional_results if r["direction"] == "ascendente"]
        descendente_stats = [r for r in directional_results if r["direction"] == "descendente"]

        return {
            "ascendente_count": len(ascendente_stats),
            "descendente_count": len(descendente_stats),
            "ascendente_avg_success": np.mean([r["success_rate_target"] for r in ascendente_stats]) if ascendente_stats else 0,
            "descendente_avg_success": np.mean([r["success_rate_target"] for r in descendente_stats]) if descendente_stats else 0,
        }

    def _calculate_specificity_gaps(self, specificity_results):
        """Calcula gaps de especificidade"""
        gaps = {}
        for result in specificity_results:
            target = result["target_circuit"]
            if target not in gaps:
                gaps[target] = []

            gaps[target].append(result["delta_tm"])

        # Calcular estatísticas por circuito
        gap_stats = {}
        for circuit, gap_list in gaps.items():
            gap_stats[circuit] = {"mean_gap": np.mean(gap_list), "std_gap": np.std(gap_list) if len(gap_list) > 1 else 0, "max_gap": max(gap_list), "min_gap": min(gap_list)}

        return gap_stats

    def _export_episode_details(self, report, output_dir, timestamp):
        """Exporta detalhes de todos os episódios"""
        try:
            episodes_dir = os.path.join(output_dir, "episode_details")
            os.makedirs(episodes_dir, exist_ok=True)

            # Coletar todos os caminhos de dados brutos
            all_raw_paths = []

            # Adicionar caminhos da complexidade
            for item in report["raw_data"]["complexity"]:
                if "raw_data_path" in item and item["raw_data_path"]:
                    all_raw_paths.append(item["raw_data_path"])

            # Adicionar caminhos da generalização
            for item in report["raw_data"]["generalization"]:
                if "raw_data_path" in item and item["raw_data_path"]:
                    all_raw_paths.append(item["raw_data_path"])

            # Consolidar todos os episódios
            all_episodes = []

            for raw_path in all_raw_paths:
                if os.path.exists(raw_path):
                    try:
                        with open(raw_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if "episodes" in data:
                            for ep_num, ep_data in data["episodes"].items():
                                if isinstance(ep_data, dict) and "episode_data" in ep_data:
                                    episode_info = ep_data["episode_data"].copy()
                                    episode_info["evaluation_file"] = os.path.basename(raw_path)
                                    all_episodes.append(episode_info)
                    except Exception as e:
                        self.logger.warning(f"Erro ao processar {raw_path}: {e}")

            # Salvar todos os episódios consolidados
            if all_episodes:
                episodes_df = pd.DataFrame(all_episodes)
                episodes_csv_path = os.path.join(episodes_dir, f"all_episodes_{timestamp}.csv")
                episodes_df.to_csv(episodes_csv_path, index=False)

                self.logger.info(f"Detalhes de {len(all_episodes)} episódios salvos em: {episodes_csv_path}")

        except Exception as e:
            self.logger.error(f"Erro ao exportar detalhes dos episódios: {e}")

    def export_report(self, report, output_dir="logs/cross_evaluation"):
        """Exporta relatório completo"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salvar JSON completo
        json_path = os.path.join(output_dir, f"cross_evaluation_report_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Salvar CSV resumido
        self._export_csv_report(report, output_dir, timestamp)

        self.logger.info(f"Relatório exportado: {json_path}")
        return json_path

    def _export_csv_report(self, report, output_dir, timestamp):
        """Exporta relatório em formato CSV para análise"""

        # CSV de complexidade
        complexity_df = pd.DataFrame(report["raw_data"]["complexity"])
        complexity_df.to_csv(os.path.join(output_dir, f"complexity_{timestamp}.csv"), index=False)

        # CSV de generalização
        generalization_df = pd.DataFrame(report["raw_data"]["generalization"])
        generalization_df.to_csv(os.path.join(output_dir, f"generalization_{timestamp}.csv"), index=False)

        # CSV de especificidade
        specificity_df = pd.DataFrame(report["raw_data"]["specificity"])
        specificity_df.to_csv(os.path.join(output_dir, f"specificity_{timestamp}.csv"), index=False)
