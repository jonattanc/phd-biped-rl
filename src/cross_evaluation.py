# cross_evaluation.py
import os
import json
import pandas as pd
from datetime import datetime
import numpy as np


class CrossEvaluation:
    def __init__(self, logger, models_directory):
        self.logger = logger
        self.models_directory = models_directory
        self.complexity_order = ["PR", "PBA", "PRA", "PRD", "PG", "PRB"]

    def run_complete_evaluation(self, num_episodes=20, deterministic=True):
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
        comprehensive_report = self._generate_comprehensive_report(complexity_results, generalization_results, specificity_results, directional_analysis)

        return comprehensive_report

    def _run_complexity_analysis(self, num_episodes, deterministic):
        """RF-08: Avaliação de Complexidade - 6 circuitos × 20 repetições"""
        complexity_results = []

        for circuit in self.complexity_order:
            model_path = self._find_specialist_model(circuit)
            if not model_path:
                self.logger.warning(f"Modelo especialista não encontrado para {circuit}")
                continue

            self.logger.info(f"Avaliando complexidade: {circuit}")

            # Executar avaliação (integrar com evaluate_model existente)
            metrics = self._evaluate_model_on_circuit(model_path, circuit, "robot_stage1", num_episodes, deterministic)

            complexity_results.append(
                {"circuit": circuit, "model_path": model_path, "metrics": metrics, "avg_time": metrics["avg_time"], "std_time": metrics["std_time"], "success_rate": metrics["success_rate"]}
            )

        return complexity_results

    def _run_generalization_analysis(self, num_episodes, deterministic):
        """RF-09: Avaliação de Generalização - 6×5×20 execuções"""
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
                metrics = self._evaluate_model_on_circuit(origin_model, target_circuit, "robot_stage1", num_episodes, deterministic)

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
        """RF-10: Avaliação de Especificidade - 6×6×20 execuções"""
        specificity_results = []

        for target_circuit in self.complexity_order:
            # Encontrar AE (especialista) deste circuito
            ae_model = self._find_specialist_model(target_circuit)
            if not ae_model:
                continue

            # Avaliar o AE em seu próprio circuito
            ae_metrics = self._evaluate_model_on_circuit(ae_model, target_circuit, "robot_stage1", num_episodes, deterministic)

            # Avaliar todos os AGs (generalistas) neste circuito
            for origin_circuit in self.complexity_order:
                if origin_circuit == target_circuit:
                    continue  # AE já avaliado

                ag_model = self._find_specialist_model(origin_circuit)
                if not ag_model:
                    continue

                ag_metrics = self._evaluate_model_on_circuit(ag_model, target_circuit, "robot_stage1", num_episodes, deterministic)

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
        # Padrão de nome esperado: model_PR.zip, model_PBA.zip, etc.
        possible_patterns = [f"model_{circuit_name}.zip", f"{circuit_name}_model.zip", f"specialist_{circuit_name}.zip"]

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

    def _evaluate_model_on_circuit(self, model_path, circuit_name, robot_name, num_episodes, deterministic):
        """Integração com evaluate_model existente"""
        try:
            from evaluate_model import evaluate_and_save

            metrics = evaluate_and_save(
                model_path=model_path,
                circuit_name=circuit_name,
                avatar_name=robot_name,
                num_episodes=num_episodes,
                deterministic=deterministic,
                seed=42,
                enable_visualization=False,  # Desativar visualização para performance
            )

            return metrics if metrics else self._create_default_metrics()

        except ImportError:
            self.logger.error("Módulo evaluate_model não encontrado")
            return self._create_default_metrics()

    def _create_default_metrics(self):
        """Métricas padrão em caso de erro"""
        return {"avg_time": 0, "std_time": 0, "success_rate": 0, "success_count": 0, "num_episodes": 0, "total_times": []}

    def _generate_comprehensive_report(self, complexity, generalization, specificity, directional):
        """Gera relatório completo da avaliação cruzada"""

        # Calcular ranking de complexidade
        complexity_ranking = self._rank_circuit_complexity(complexity)

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(complexity) + len(generalization) + len(specificity),
                "circuits_tested": self.complexity_order,
                "num_episodes": 20,
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
