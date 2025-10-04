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
        utils.setup_ipc_logging(self.logger, self.ipc_queue)

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
        ttk.Label(settings_frame, text="Diretório de Modelos Especialistas:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.models_dir_var = tk.StringVar(value=utils.TRAINING_DATA_PATH)
        ttk.Entry(settings_frame, textvariable=self.models_dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(settings_frame, text="Procurar", command=self.browse_models_dir, width=10).grid(row=0, column=2, padx=5)

        # Parâmetros de avaliação
        ttk.Label(settings_frame, text="Episódios por avaliação:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.cross_episodes_var = tk.StringVar(value="20")
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
            # Gráfico de ranking de complexidade
            self.axs_cross[0, 0].set_title("Ranking de Complexidade (RF-08)")
            self.axs_cross[0, 0].set_ylabel("Tempo Médio (s)")
            self.axs_cross[0, 0].grid(True, alpha=0.3)

            # Gráfico de generalização
            self.axs_cross[0, 1].set_title("Performance de Generalização (RF-09)")
            self.axs_cross[0, 1].set_ylabel("Taxa de Sucesso (%)")
            self.axs_cross[0, 1].grid(True, alpha=0.3)

            # Gráfico de transferências direcionais
            self.axs_cross[1, 0].set_title("Transferências Direcionais (RF-11)")
            self.axs_cross[1, 0].set_ylabel("Quantidade")
            self.axs_cross[1, 0].grid(True, alpha=0.3)

            # Gráfico de gaps de especificidade
            self.axs_cross[1, 1].set_title("Gaps de Especificidade (RF-10)")
            self.axs_cross[1, 1].set_ylabel("ΔTm (s)")
            self.axs_cross[1, 1].grid(True, alpha=0.3)

            self.canvas_cross.draw_idle()

        except Exception as e:
            self.logger.exception("Erro ao inicializar gráficos de avaliação cruzada")

    def browse_models_dir(self):
        """Seleciona diretório com modelos especialistas"""
        directory = filedialog.askdirectory(title="Selecionar Diretório de Modelos Especialistas", initialdir=utils.TRAINING_DATA_PATH)
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
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na avaliação cruzada: {e}"))
        finally:
            self.root.after(0, lambda: self.run_evaluation_btn.config(state=tk.NORMAL, text="Executar Avaliação Cruzada Completa"))

    def _display_cross_evaluation_results(self, report, report_path):
        """Exibe resultados da avaliação cruzada"""
        try:
            self.cross_evaluation_results = report

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            results_text = "=== AVALIAÇÃO CRUZADA COMPLETA ===\n\n"
            results_text += f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            results_text += f"Total de avaliações: {report['metadata']['total_evaluations']}\n"
            results_text += f"Episódios por avaliação: {report['metadata']['num_episodes']}\n\n"

            # 1. Ranking de Complexidade (RF-08)
            results_text += "1. RANKING DE COMPLEXIDADE (RF-08)\n"
            results_text += "=" * 50 + "\n"
            for i, circuit in enumerate(report["complexity_ranking"], 1):
                specialist = report["best_specialists"][circuit["circuit"]]
                results_text += f"{i}º - {circuit['circuit']}: {circuit['avg_time']:.2f}s "
                results_text += f"({specialist['success_rate']*100:.1f}% sucesso)\n"

            # 2. Análise de Generalização (RF-09)
            results_text += "\n2. ANÁLISE DE GENERALIZAÇÃO (RF-09)\n"
            results_text += "=" * 50 + "\n"

            gen_patterns = report["generalization_analysis"]
            for origin, targets in gen_patterns.items():
                origin_time = report["best_specialists"][origin]["avg_time"]
                results_text += f"\n{origin} (Tm={origin_time:.2f}s) → \n"
                for target, metrics in targets.items():
                    delta = (metrics["avg_time"] - origin_time) / origin_time * 100
                    results_text += f"  • {target}: {metrics['avg_time']:.2f}s "
                    results_text += f"({metrics['success_rate']*100:.1f}%, Δ={delta:+.1f}%)\n"

            # 3. Transferências Direcionais (RF-11)
            results_text += "\n3. TRANSFERÊNCIAS DIRECIONAIS (RF-11)\n"
            results_text += "=" * 50 + "\n"
            directional = report["directional_insights"]
            results_text += f"• Transferências Ascendentes: {directional['ascendente_count']}\n"
            results_text += f"• Transferências Descendentes: {directional['descendente_count']}\n"
            results_text += f"• Sucesso Médio Ascendente: {directional['ascendente_avg_success']*100:.1f}%\n"
            results_text += f"• Sucesso Médio Descendente: {directional['descendente_avg_success']*100:.1f}%\n"

            # 4. Gaps de Especificidade (RF-10)
            results_text += "\n4. GAPS DE ESPECIFICIDADE (RF-10)\n"
            results_text += "=" * 50 + "\n"
            gaps = report["specificity_gaps"]
            for circuit, gap_stats in gaps.items():
                ae_time = report["best_specialists"][circuit]["avg_time"]
                results_text += f"• {circuit} (AE: {ae_time:.2f}s): "
                results_text += f"ΔTm = {gap_stats['mean_gap']:+.2f}s "
                results_text += f"(±{gap_stats['std_gap']:.2f}s)\n"

            results_text += f"\nRelatório completo salvo em:\n{report_path}"

            self.results_text.insert(1.0, results_text)
            self.results_text.config(state=tk.DISABLED)

            # Atualizar gráficos
            self._update_cross_plots(report)

            # Habilitar botões de exportação
            self.export_report_btn.config(state=tk.NORMAL)

            messagebox.showinfo(
                "Avaliação Cruzada Concluída",
                f"Todos os requisitos atendidos:\n"
                f"• RF-08 (Complexidade): ✓\n"
                f"• RF-09 (Generalização): ✓\n"
                f"• RF-10 (Especificidade): ✓\n"
                f"• RF-11 (Direcionalidade): ✓\n\n"
                f"Total: {report['metadata']['total_evaluations']} avaliações",
            )

        except Exception as e:
            self.logger.exception("Erro ao exibir resultados")
            messagebox.showerror("Erro", f"Erro ao exibir resultados: {e}")

    def _update_cross_plots(self, report):
        """Atualiza os gráficos com os resultados da avaliação cruzada"""
        try:
            # Limpar gráficos
            for ax in self.axs_cross.flat:
                ax.clear()

            # 1. Ranking de Complexidade
            circuits = [item["circuit"] for item in report["complexity_ranking"]]
            times = [item["avg_time"] for item in report["complexity_ranking"]]

            bars = self.axs_cross[0, 0].bar(circuits, times, color="skyblue", alpha=0.7)
            self.axs_cross[0, 0].set_title("Ranking de Complexidade (RF-08)")
            self.axs_cross[0, 0].set_ylabel("Tempo Médio (s)")
            self.axs_cross[0, 0].tick_params(axis="x", rotation=45)

            # Adicionar valores nas barras
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                self.axs_cross[0, 0].text(bar.get_x() + bar.get_width() / 2.0, height, f"{time_val:.2f}s", ha="center", va="bottom")

            # 2. Performance de Generalização (heatmap simplificado)
            gen_data = report["generalization_analysis"]
            circuits_ordered = [item["circuit"] for item in report["complexity_ranking"]]
            success_rates = []

            for origin in circuits_ordered:
                row = []
                for target in circuits_ordered:
                    if origin == target:
                        row.append(report["best_specialists"][origin]["success_rate"] * 100)
                    else:
                        row.append(gen_data.get(origin, {}).get(target, {}).get("success_rate", 0) * 100)
                success_rates.append(row)

            im = self.axs_cross[0, 1].imshow(success_rates, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
            self.axs_cross[0, 1].set_title("Taxa de Sucesso por Transferência (RF-09)")
            self.axs_cross[0, 1].set_xticks(range(len(circuits_ordered)))
            self.axs_cross[0, 1].set_yticks(range(len(circuits_ordered)))
            self.axs_cross[0, 1].set_xticklabels(circuits_ordered, rotation=45)
            self.axs_cross[0, 1].set_yticklabels(circuits_ordered)
            self.axs_cross[0, 1].set_xlabel("Circuito Alvo")
            self.axs_cross[0, 1].set_ylabel("Circuito Origem")

            # Adicionar barra de cores
            plt.colorbar(im, ax=self.axs_cross[0, 1], label="Taxa de Sucesso (%)")

            # 3. Transferências Direcionais
            directional = report["directional_insights"]
            directions = ["Ascendente", "Descendente"]
            counts = [directional["ascendente_count"], directional["descendente_count"]]
            colors = ["lightcoral", "lightblue"]

            bars = self.axs_cross[1, 0].bar(directions, counts, color=colors, alpha=0.7)
            self.axs_cross[1, 0].set_title("Transferências Direcionais (RF-11)")
            self.axs_cross[1, 0].set_ylabel("Quantidade")

            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.axs_cross[1, 0].text(bar.get_x() + bar.get_width() / 2.0, height, f"{count}", ha="center", va="bottom")

            # 4. Gaps de Especificidade
            gaps = report["specificity_gaps"]
            circuits_gap = list(gaps.keys())
            mean_gaps = [gaps[circuit]["mean_gap"] for circuit in circuits_gap]

            colors_gap = ["red" if gap > 0 else "green" for gap in mean_gaps]
            bars = self.axs_cross[1, 1].bar(circuits_gap, mean_gaps, color=colors_gap, alpha=0.7)
            self.axs_cross[1, 1].set_title("Gaps de Especificidade (RF-10)")
            self.axs_cross[1, 1].set_ylabel("ΔTm (s)")
            self.axs_cross[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
            self.axs_cross[1, 1].tick_params(axis="x", rotation=45)

            for bar, gap in zip(bars, mean_gaps):
                height = bar.get_height()
                self.axs_cross[1, 1].text(bar.get_x() + bar.get_width() / 2.0, height, f"{gap:+.2f}s", ha="center", va="bottom" if gap >= 0 else "top")

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
