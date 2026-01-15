# tab_comparison.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import queue
import json
from datetime import datetime
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from cross_evaluation import CrossEvaluation


class ComparisonTab:
    def __init__(self, parent, device, logger):
        self.frame = ttk.Frame(parent)
        self.device = device
        self.logger = logger

        # IPC Queue para comunicação
        self.ipc_queue = queue.Queue()

        # Dados de avaliação cruzada
        self.cross_evaluation_results = None
        self.cross_evaluator = None

        # Configurar IPC logging
        utils.add_queue_handler_to_logger(self.logger, self.ipc_queue)

        self.setup_ui()

    def setup_ui(self):
        """Configura a interface unificada para avaliação cruzada"""
        # Frame principal
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=1)

        # Controles de avaliação cruzada
        control_frame = ttk.LabelFrame(main_frame, text="Avaliação Cruzada de Modelos", padding="10")
        control_frame.pack(fill=tk.X, pady=1)

        # Configurações
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=1)

        # Diretório de modelos
        ttk.Label(settings_frame, text="Diretório de Modelos:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.models_dir_var = tk.StringVar(value=utils.MODELS_DATA_PATH)
        ttk.Entry(settings_frame, textvariable=self.models_dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(settings_frame, text="Procurar", command=self.browse_models_dir, width=10).grid(row=0, column=2, padx=5)

        # Parâmetros de avaliação
        ttk.Label(settings_frame, text="Episódios por avaliação:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.cross_episodes_var = tk.StringVar(value="100")
        ttk.Entry(settings_frame, textvariable=self.cross_episodes_var, width=8).grid(row=1, column=1, padx=5, sticky=tk.W)

        self.cross_deterministic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Modo Determinístico", variable=self.cross_deterministic_var).grid(row=1, column=2, padx=5)

        # Botão de execução
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=1)

        self.run_evaluation_btn = ttk.Button(button_frame, text="Executar Avaliação Cruzada", command=self.run_cross_evaluation, width=30)
        self.run_evaluation_btn.pack(pady=1)

        # Frame para resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados da Avaliação Cruzada", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=1)

        # Notebook para organizar resultados
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)

        # Aba de relatório textual
        report_frame = ttk.Frame(results_notebook)
        results_notebook.add(report_frame, text="Relatório Consolidado")

        self.results_text = tk.Text(report_frame, height=15, state=tk.DISABLED, wrap=tk.WORD)
        text_scrollbar = ttk.Scrollbar(report_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=text_scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Aba de visualizações
        viz_frame = ttk.Frame(results_notebook)
        results_notebook.add(viz_frame, text="Visualizações")

        # Gráficos de avaliação cruzada
        self.fig_cross, self.axs_cross = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_cross = FigureCanvasTkAgg(self.fig_cross, master=viz_frame)
        self.canvas_cross.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_cross_plots()

        # Botões de exportação
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, pady=5)

        self.export_report_btn = ttk.Button(export_frame, text="Exportar Relatório Completo", command=self.export_cross_evaluation_report, state=tk.DISABLED, width=20)
        self.export_report_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(export_frame, text="Exportar Gráficos", command=self.export_cross_plots, state=tk.DISABLED, width=15).pack(side=tk.LEFT, padx=5)

        # Inicializar avaliador
        self.cross_evaluator = CrossEvaluation(self.logger, self.models_dir_var.get())

    def _initialize_cross_plots(self):
        """Inicializa os gráficos para avaliação cruzada"""
        try:
            # Gráfico de tempo por transferência
            self.axs_cross[0, 0].set_title("Tempo por Transferência (RF-09)")
            self.axs_cross[0, 0].set_ylabel("Tempo Médio (s)")
            self.axs_cross[0, 0].grid(True, alpha=0.3)
    
            # Heatmap de sucesso
            self.axs_cross[0, 1].set_title("Matriz de Sucesso (RF-09)")
            self.axs_cross[0, 1].grid(False)
    
            # Gráfico de transferências direcionais
            self.axs_cross[1, 0].set_title("Transferências Direcionais (RF-11)")
            self.axs_cross[1, 0].set_ylabel("Quantidade")
            self.axs_cross[1, 0].grid(True, alpha=0.3)
    
            # Gráfico de sucesso por direção
            self.axs_cross[1, 1].set_title("Sucesso por Direção (RF-11)")
            self.axs_cross[1, 1].set_ylabel("Taxa de Sucesso (%)")
            self.axs_cross[1, 1].grid(True, alpha=0.3)
    
            self.canvas_cross.draw_idle()
    
        except Exception as e:
            self.logger.exception("Erro ao inicializar gráficos de avaliação cruzada")

    def browse_models_dir(self):
        """Seleciona diretório com modelos especialistas"""
        directory = filedialog.askdirectory(title="Selecionar Diretório de Modelos Especialistas", initialdir=utils.MODELS_DATA_PATH)
        if directory:
            self.models_dir_var.set(directory)
            self.cross_evaluator = CrossEvaluation(self.logger, directory)
            self.logger.info(f"Diretório de modelos atualizado: {directory}")

    def run_cross_evaluation(self):
        """Executa avaliação cruzada completa (RF-08, RF-09, RF-10, RF-11)"""
        try:
            episodes = int(self.cross_episodes_var.get())
            if episodes <= 0:
                raise ValueError("Número de episódios deve ser positivo")
        except ValueError:
            messagebox.showerror("Erro", "Número de episódios deve ser um número inteiro positivo.")
            return

        # Verificar se existem modelos no diretório
        if not self._check_models_exist():
            messagebox.showwarning("Aviso", f"Nenhum modelo especialista encontrado em:\n{self.models_dir_var.get()}\n" f"Certifique-se de que existem modelos para: PR, PBA, PRA, PRD, PG, PRB")
            return

        # Atualizar diretório do avaliador
        self.cross_evaluator.models_directory = self.models_dir_var.get()

        # Desabilitar botão durante execução
        self.run_evaluation_btn.config(state=tk.DISABLED, text="Executando Avaliação Cruzada...")

        # Executar em thread separada
        eval_thread = threading.Thread(target=self._run_cross_evaluation_thread, args=(episodes, self.cross_deterministic_var.get()), daemon=True)
        eval_thread.start()

    def _check_models_exist(self):
        """Verifica se existem modelos especialistas no diretório"""
        required_circuits = ["PR", "PBA", "PRA", "PRD", "PG", "PRB"]
        found_models = 0

        for circuit in required_circuits:
            if self.cross_evaluator._find_specialist_model(circuit):
                found_models += 1

        self.logger.info(f"Modelos encontrados: {found_models}/6")
        return found_models > 0  # Pelo menos um modelo

    def _run_cross_evaluation_thread(self, episodes, deterministic):
        """Executa avaliação cruzada em thread separada"""
        try:
            self.logger.info("Iniciando avaliação cruzada completa...")

            # Executar avaliação completa
            report = self.cross_evaluator.run_complete_evaluation(num_episodes=episodes, deterministic=deterministic)

            # Exportar relatório automaticamente
            report_path = self.cross_evaluator.export_report(report)

            # Atualizar interface
            self.root.after(0, lambda: self._display_cross_evaluation_results(report, report_path))

        except Exception as e:
            self.logger.exception("Erro na avaliação cruzada")
            error_msg = str(e)  # Salvar a mensagem de erro em uma variável
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na avaliação cruzada: {error_msg}"))

    def _display_cross_evaluation_results(self, report, report_path):
        """Exibe resultados da avaliação cruzada - VERSÃO SIMPLIFICADA"""
        try:
            self.cross_evaluation_results = report

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            results_text = "=== AVALIAÇÃO CRUZADA - GENERALIZAÇÃO ===\n\n"
            results_text += f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            results_text += f"Total de transferências: {report['metadata']['total_transfers']}\n"
            results_text += f"Episódios por avaliação: {report['metadata']['num_episodes']}\n"
            results_text += f"Ordem de complexidade: {', '.join(report['metadata']['circuits_order'])}\n\n"

            # Análise de Generalização (RF-09)
            results_text += "1. ANÁLISE DE GENERALIZAÇÃO (RF-09)\n"
            results_text += "=" * 50 + "\n\n"

            # Agrupar por modelo origem para melhor visualização
            from collections import defaultdict
            by_origin = defaultdict(list)

            for transfer in report["generalization_results"]:
                by_origin[transfer["origin_circuit"]].append(transfer)

            for origin, transfers in by_origin.items():
                results_text += f"Modelo {origin} →\n"
                for transfer in transfers:
                    target = transfer["target_circuit"]
                    time = transfer["avg_time_target"]
                    success = transfer["success_rate_target"] * 100
                    distance = transfer["avg_distance_target"]
                    velocity = transfer["avg_velocity_target"]

                    results_text += f"  • {target}: {time:.2f}s, {success:.1f}% sucesso, "
                    results_text += f"{distance:.1f}m, {velocity:.2f}m/s\n"
                results_text += "\n"

            # Transferências Direcionais (RF-11)
            results_text += "2. TRANSFERÊNCIAS DIRECIONAIS (RF-11)\n"
            results_text += "=" * 50 + "\n\n"

            ascendente = report["summary"]["ascendente_count"]
            descendente = report["summary"]["descendente_count"]
            total = ascendente + descendente

            results_text += f"• Transferências Ascendentes (→ mais complexo): {ascendente}\n"
            results_text += f"• Transferências Descendentes (→ menos complexo): {descendente}\n"
            results_text += f"• Total: {total}\n"
            results_text += f"• Proporção Ascendente/Descendente: {ascendente/descendente:.2f} (se >1, mais ascendentes)\n\n"

            # Análise por direção
            results_text += "Resumo por direção:\n"
            direction_data = {"ascendente": [], "descendente": []}

            for transfer in report["directional_analysis"]:
                direction = transfer["direction"]
                direction_data[direction].append({
                    "success": transfer["success_rate_target"],
                    "time": transfer["avg_time_target"]
                })

            for direction, data in direction_data.items():
                if data:
                    avg_success = sum(d["success"] for d in data) / len(data) * 100
                    avg_time = sum(d["time"] for d in data) / len(data)
                    results_text += f"  • {direction.capitalize()}: {avg_success:.1f}% sucesso médio, {avg_time:.2f}s tempo médio\n"

            # Caminhos dos arquivos
            results_text += f"\nArquivos gerados:\n"
            results_text += f"• Generalização: {report_path['generalization_csv']}\n"
            results_text += f"• Direcionalidade: {report_path['directionality_csv']}\n"
            results_text += f"• JSON resumo: {report_path['json_path']}\n"
            results_text += f"• Diretório: {report_path['output_dir']}\n"

            self.results_text.insert(1.0, results_text)
            self.results_text.config(state=tk.DISABLED)

            # Atualizar gráficos
            self._update_cross_plots(report)

            # Habilitar botões de exportação
            self.export_report_btn.config(state=tk.NORMAL)

            messagebox.showinfo(
                "Avaliação Cruzada Concluída",
                f"Análise de generalização concluída:\n"
                f"• RF-09 (Generalização): ✓\n"
                f"• RF-11 (Direcionalidade): ✓\n\n"
                f"Total de transferências: {report['metadata']['total_transfers']}\n"
                f"Arquivos CSV salvos em: {report_path['output_dir']}",
            )

        except Exception as e:
            self.logger.exception("Erro ao exibir resultados")
            messagebox.showerror("Erro", f"Erro ao exibir resultados: {e}")

    def _update_cross_plots(self, report):
        """Atualiza os gráficos com os resultados da avaliação cruzada SIMPLIFICADA"""
        try:
            # Limpar gráficos
            for ax in self.axs_cross.flat:
                ax.clear()

            # 1. Gráfico de tempo médio por transferência
            origins = []
            targets = []
            times = []
            successes = []

            for transfer in report["generalization_results"]:
                origins.append(transfer["origin_circuit"])
                targets.append(transfer["target_circuit"])
                times.append(transfer["avg_time_target"])
                successes.append(transfer["success_rate_target"] * 100)

            # Criar labels únicos para cada transferência
            transfer_labels = [f"{o}→{t}" for o, t in zip(origins, targets)]

            bars = self.axs_cross[0, 0].bar(transfer_labels, times, color="skyblue", alpha=0.7)
            self.axs_cross[0, 0].set_title("Tempo Médio por Transferência (RF-09)")
            self.axs_cross[0, 0].set_ylabel("Tempo Médio (s)")
            self.axs_cross[0, 0].tick_params(axis="x", rotation=45, labelsize=8)

            # 2. Heatmap de sucesso por origem→destino
            circuits = report["metadata"]["circuits_order"]
            success_matrix = np.zeros((len(circuits), len(circuits)))

            for transfer in report["generalization_results"]:
                i = circuits.index(transfer["origin_circuit"])
                j = circuits.index(transfer["target_circuit"])
                success_matrix[i, j] = transfer["success_rate_target"] * 100

            im = self.axs_cross[0, 1].imshow(success_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
            self.axs_cross[0, 1].set_title("Taxa de Sucesso - Matriz (RF-09)")
            self.axs_cross[0, 1].set_xticks(range(len(circuits)))
            self.axs_cross[0, 1].set_yticks(range(len(circuits)))
            self.axs_cross[0, 1].set_xticklabels(circuits)
            self.axs_cross[0, 1].set_yticklabels(circuits)
            self.axs_cross[0, 1].set_xlabel("Circuito Destino")
            self.axs_cross[0, 1].set_ylabel("Circuito Origem")

            # Adicionar valores na matriz
            for i in range(len(circuits)):
                for j in range(len(circuits)):
                    if i != j:  # Não mostrar auto-transferências
                        value = success_matrix[i, j]
                        if value > 0:
                            self.axs_cross[0, 1].text(j, i, f"{value:.1f}%", 
                                                     ha="center", va="center", 
                                                     color="black" if value < 50 else "white",
                                                     fontsize=8)

            plt.colorbar(im, ax=self.axs_cross[0, 1], label="Taxa de Sucesso (%)")

            # 3. Gráfico de transferências direcionais
            directions = ["Ascendente", "Descendente"]
            counts = [
                report["summary"]["ascendente_count"],
                report["summary"]["descendente_count"]
            ]
            colors = ["lightcoral", "lightblue"]

            bars = self.axs_cross[1, 0].bar(directions, counts, color=colors, alpha=0.7)
            self.axs_cross[1, 0].set_title("Transferências Direcionais (RF-11)")
            self.axs_cross[1, 0].set_ylabel("Quantidade")

            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.axs_cross[1, 0].text(bar.get_x() + bar.get_width()/2, height, 
                                         f"{count}", ha="center", va="bottom")

            # 4. Comparação de sucesso por direção
            direction_data = {"Ascendente": [], "Descendente": []}

            for transfer in report["directional_analysis"]:
                direction = "Ascendente" if transfer["direction"] == "ascendente" else "Descendente"
                direction_data[direction].append(transfer["success_rate_target"] * 100)

            direction_names = []
            avg_success = []

            for direction, values in direction_data.items():
                if values:
                    direction_names.append(direction)
                    avg_success.append(np.mean(values))

            if direction_names:
                bars = self.axs_cross[1, 1].bar(direction_names, avg_success, 
                                              color=["lightcoral", "lightblue"], alpha=0.7)
                self.axs_cross[1, 1].set_title("Sucesso por Direção (RF-11)")
                self.axs_cross[1, 1].set_ylabel("Taxa de Sucesso Média (%)")
                self.axs_cross[1, 1].axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50%")
                self.axs_cross[1, 1].legend()

                for bar, success in zip(bars, avg_success):
                    height = bar.get_height()
                    self.axs_cross[1, 1].text(bar.get_x() + bar.get_width()/2, height, 
                                             f"{success:.1f}%", ha="center", va="bottom")

            self.fig_cross.tight_layout()
            self.canvas_cross.draw_idle()

        except Exception as e:
            self.logger.exception("Erro ao atualizar gráficos de avaliação cruzada")

    def export_cross_evaluation_report(self):
        """Exporta relatório da avaliação cruzada"""
        if not self.cross_evaluation_results:
            messagebox.showwarning("Aviso", "Nenhum resultado de avaliação cruzada para exportar.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                title="Salvar Relatório de Avaliação Cruzada",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"cross_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(self.cross_evaluation_results, f, indent=2, ensure_ascii=False)

                messagebox.showinfo("Sucesso", f"Relatório exportado para:\n{filename}")
                self.logger.info(f"Relatório de avaliação cruzada exportado: {filename}")

        except Exception as e:
            self.logger.exception("Erro ao exportar relatório de avaliação cruzada")
            messagebox.showerror("Erro", f"Erro ao exportar relatório: {e}")

    def export_cross_plots(self):
        """Exporta os gráficos da avaliação cruzada"""
        if not self.cross_evaluation_results:
            messagebox.showwarning("Aviso", "Nenhum gráfico para exportar.")
            return

        try:
            directory = filedialog.askdirectory(title="Selecione onde salvar os gráficos")
            if not directory:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Salvar gráfico combinado
            self.fig_cross.savefig(os.path.join(directory, f"cross_evaluation_plots_{timestamp}.png"), dpi=300, bbox_inches="tight")

            messagebox.showinfo("Sucesso", f"Gráficos exportados para:\n{directory}")
            self.logger.info(f"Gráficos de avaliação cruzada exportados: {directory}")

        except Exception as e:
            self.logger.exception("Erro ao exportar gráficos de avaliação cruzada")
            messagebox.showerror("Erro", f"Erro ao exportar gráficos: {e}")

    def start(self):
        """Inicializa a aba de comparação"""
        self.logger.info("Aba de avaliação cruzada inicializada")

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()
