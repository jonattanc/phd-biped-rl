# tab_evaluation.py
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import json
from datetime import datetime
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from utils import setup_ipc_logging, validate_episodes_count, ensure_directory


class EvaluationTab:
    def __init__(self, parent, device, logger):
        self.frame = ttk.Frame(parent)
        self.device = device
        self.logger = logger

        # IPC Queue para comunicação
        self.ipc_queue = queue.Queue()

        # Dados de avaliação
        self.evaluation_data = {"current_evaluation": None, "evaluation_history": [], "comparison_data": []}

        # Componentes da UI
        self.eval_model_path = None
        self.eval_env_var = None
        self.eval_robot_var = None
        self.eval_episodes_var = None
        self.eval_deterministic_var = None
        self.eval_start_btn = None
        self.metrics_text = None
        self.fig_evaluation = None
        self.axs_evaluation = None
        self.canvas_evaluation = None

        # Configurar IPC logging
        setup_ipc_logging(self.logger, self.ipc_queue)

        self.setup_ui()

    def setup_ui(self):
        """Configura a interface da aba de avaliação"""
        # Frame principal
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controles de avaliação
        control_frame = ttk.LabelFrame(main_frame, text="Configuração de Avaliação", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Linha 1: Seleção de modelo
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row1_frame, text="Modelo para Avaliar:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.eval_model_path = tk.StringVar()
        ttk.Entry(row1_frame, textvariable=self.eval_model_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(row1_frame, text="Procurar", command=self.browse_evaluation_model).grid(row=0, column=2, padx=5)

        # Linha 2: Configurações de avaliação
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row2_frame, text="Ambiente:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.eval_env_var = tk.StringVar(value="PR")
        env_combo = ttk.Combobox(row2_frame, textvariable=self.eval_env_var, values=["PR", "Pmu", "RamA", "RamD", "PG", "PRB"], width=10)
        env_combo.grid(row=0, column=1, padx=5)

        ttk.Label(row2_frame, text="Robô:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.eval_robot_var = tk.StringVar(value="robot_stage1")
        robot_combo = ttk.Combobox(row2_frame, textvariable=self.eval_robot_var, values=["robot_stage1", "robot_stage2", "robot_stage3"], width=12)
        robot_combo.grid(row=0, column=3, padx=5)

        ttk.Label(row2_frame, text="Episódios:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.eval_episodes_var = tk.StringVar(value="20")
        ttk.Entry(row2_frame, textvariable=self.eval_episodes_var, width=8).grid(row=0, column=5, padx=5)

        # Linha 3: Botões de controle
        row3_frame = ttk.Frame(control_frame)
        row3_frame.pack(fill=tk.X, pady=5)

        self.eval_start_btn = ttk.Button(row3_frame, text="Executar Avaliação", command=self.start_evaluation, width=20)
        self.eval_start_btn.grid(row=0, column=0, padx=5)

        self.eval_deterministic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3_frame, text="Modo Determinístico", variable=self.eval_deterministic_var).grid(row=0, column=1, padx=5)

        self.record_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3_frame, text="Gravar Vídeos", variable=self.record_video_var).grid(row=0, column=2, padx=5)

        # Botões de exportação
        ttk.Button(row3_frame, text="Exportar Resultados", command=self.export_evaluation_results).grid(row=0, column=3, padx=5)
        ttk.Button(row3_frame, text="Salvar Gráficos", command=self.export_evaluation_plots).grid(row=0, column=4, padx=5)

        # Resultados da avaliação
        results_frame = ttk.LabelFrame(main_frame, text="Resultados da Avaliação", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Notebook para organizar resultados
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)

        # Aba de métricas textuais
        metrics_frame = ttk.Frame(results_notebook)
        results_notebook.add(metrics_frame, text="Métricas Detalhadas")

        # Métricas numéricas
        self.metrics_text = tk.Text(metrics_frame, height=12, state=tk.DISABLED, wrap=tk.WORD)
        metrics_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=metrics_scrollbar.set)
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metrics_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Aba de gráficos
        graph_frame = ttk.Frame(results_notebook)
        results_notebook.add(graph_frame, text="Visualizações")

        # Gráficos de avaliação
        self.fig_evaluation, self.axs_evaluation = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_evaluation = FigureCanvasTkAgg(self.fig_evaluation, master=graph_frame)
        self.canvas_evaluation.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_evaluation_plots()

        # Histórico de avaliações
        history_frame = ttk.LabelFrame(main_frame, text="Histórico de Avaliações", padding="10")
        history_frame.pack(fill=tk.X, pady=5)

        self.history_listbox = tk.Listbox(history_frame, height=4)
        history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Botões do histórico
        history_buttons = ttk.Frame(history_frame)
        history_buttons.pack(side=tk.RIGHT, padx=5)

        ttk.Button(history_buttons, text="Carregar", command=self.load_historical_evaluation, width=12).pack(pady=2)
        ttk.Button(history_buttons, text="Limpar", command=self.clear_history, width=12).pack(pady=2)

    def _initialize_evaluation_plots(self):
        """Inicializa os gráficos de avaliação usando utils"""
        try:
            # Gráfico de distribuição de tempos
            self.axs_evaluation[0, 0].set_title("Distribuição de Tempos")
            self.axs_evaluation[0, 0].set_ylabel("Frequência")
            self.axs_evaluation[0, 0].set_xlabel("Tempo (s)")
            self.axs_evaluation[0, 0].grid(True, alpha=0.3)

            # Gráfico de sucesso por episódio
            self.axs_evaluation[0, 1].set_title("Sucesso por Episódio")
            self.axs_evaluation[0, 1].set_ylabel("Sucesso")
            self.axs_evaluation[0, 1].set_xlabel("Episódio")
            self.axs_evaluation[0, 1].grid(True, alpha=0.3)

            # Gráfico de progressão temporal
            self.axs_evaluation[1, 0].set_title("Progressão de Tempos")
            self.axs_evaluation[1, 0].set_ylabel("Tempo (s)")
            self.axs_evaluation[1, 0].set_xlabel("Episódio")
            self.axs_evaluation[1, 0].grid(True, alpha=0.3)

            # Gráfico de métricas consolidadas
            self.axs_evaluation[1, 1].set_title("Métricas Consolidadas")
            self.axs_evaluation[1, 1].set_ylabel("Valor")
            self.axs_evaluation[1, 1].grid(True, alpha=0.3)

            self.canvas_evaluation.draw_idle()

        except Exception as e:
            self.logger.exception("Erro ao inicializar gráficos de avaliação")

    def browse_evaluation_model(self):
        """Abre diálogo para selecionar modelo para avaliação"""
        filename = filedialog.askopenfilename(title="Selecionar Modelo para Avaliação", filetypes=[("Zip files", "*.zip"), ("All files", "*.*")], initialdir=utils.TRAINING_DATA_PATH)
        if filename:
            self.eval_model_path.set(filename)
            self.logger.info(f"Modelo selecionado para avaliação: {os.path.basename(filename)}")

    def start_evaluation(self):
        """Inicia a avaliação do modelo selecionado usando validação do utils"""
        model_path = self.eval_model_path.get()

        # Validações usando funções do utils
        try:
            if not model_path or not os.path.exists(model_path):
                raise ValueError("Selecione um modelo válido para avaliação.")

            episodes = validate_episodes_count(self.eval_episodes_var.get())

        except ValueError as e:
            messagebox.showerror("Erro", str(e))
            return

        # Desabilitar botão durante avaliação
        self.eval_start_btn.config(state=tk.DISABLED, text="Avaliando...")

        # Executar avaliação em thread separada
        eval_thread = threading.Thread(target=self._run_evaluation, daemon=True)
        eval_thread.start()

    def _run_evaluation(self):
        """Executa a avaliação em thread separada"""
        try:
            # Importação dinâmica para evitar dependência circular
            from evaluate_model import evaluate_and_save

            model_path = self.eval_model_path.get()
            environment = self.eval_env_var.get()
            robot = self.eval_robot_var.get()
            episodes = int(self.eval_episodes_var.get())
            deterministic = self.eval_deterministic_var.get()
            record_video = self.record_video_var.get()

            self.logger.info(f"Iniciando avaliação: {model_path} no ambiente {environment}")
            self.logger.info(f"Configuração: {episodes} episódios, modo {'determinístico' if deterministic else 'estocástico'}")

            # Verificar se o arquivo do modelo existe
            if not os.path.exists(model_path):
                error_msg = f"Arquivo do modelo não encontrado: {model_path}"
                self.logger.error(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))
                return

            # Executar avaliação
            metrics = evaluate_and_save(
                model_path=model_path, 
                circuit_name=environment, 
                avatar_name=robot, 
                num_episodes=episodes, 
                deterministic=deterministic, 
                seed=42, 
                record_video=record_video
            )

            if metrics:
                # Atualizar interface com resultados
                self.root.after(0, lambda: self._display_evaluation_results(metrics))
            else:
                error_msg = "Falha na avaliação - o método evaluate_and_save retornou None"
                self.logger.error(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))

        except ImportError as e:
            error_msg = f"Módulo evaluate_model não encontrado: {e}"
            self.logger.error(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))
        except Exception as e:
            self.logger.exception("Erro na avaliação")
            self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))
        finally:
            self.root.after(0, lambda: self.eval_start_btn.config(state=tk.NORMAL, text="Executar Avaliação"))

    def _display_evaluation_results(self, metrics):
        """Exibe os resultados da avaliação na interface"""
        try:
            # Atualizar métricas textuais
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)

            success_rate = metrics.get("success_rate", 0) * 100
            avg_time = metrics.get("avg_time", 0)
            std_time = metrics.get("std_time", 0)
            success_count = metrics.get("success_count", 0)
            num_episodes = metrics.get("num_episodes", 0)
            total_rewards = metrics.get("total_rewards", [])

            results_text = f"""=== RESULTADOS DA AVALIAÇÃO ===

Estatísticas Gerais:
• Modelo: {os.path.basename(self.eval_model_path.get())}
• Ambiente: {self.eval_env_var.get()}
• Robô: {self.eval_robot_var.get()}
• Episódios executados: {num_episodes}
• Modo: {'Determinístico' if self.eval_deterministic_var.get() else 'Estocástico'}

Métricas de Performance:
• Taxa de sucesso: {success_rate:.1f}% ({success_count}/{num_episodes})
• Tempo médio: {avg_time:.2f} ± {std_time:.2f} segundos
• Melhor tempo: {min(metrics.get('total_times', [0])):.2f}s
• Pior tempo: {max(metrics.get('total_times', [0])):.2f}s
• Recompensa média: {sum(total_rewards)/len(total_rewards) if total_rewards else 0:.2f}

Distribuição de Tempos:
"""

            times = metrics.get("total_times", [])
            if times:
                for i, time_val in enumerate(times, 1):
                    status = "✓" if i <= success_count else "✗"
                    results_text += f"• Episódio {i}: {time_val:.2f}s {status}\n"

            # Análise de performance
            results_text += f"""
Análise:
• Performance: {'EXCELENTE' if success_rate >= 90 else 'BOA' if success_rate >= 70 else 'REGULAR' if success_rate >= 50 else 'RUIM'}
• Consistência: {'ALTA' if std_time < avg_time * 0.1 else 'MÉDIA' if std_time < avg_time * 0.2 else 'BAXA'}
"""

            self.metrics_text.insert(1.0, results_text)
            self.metrics_text.config(state=tk.DISABLED)

            # Atualizar gráficos
            self._update_evaluation_plots(metrics)

            # Salvar nos dados de avaliação
            evaluation_record = {
                "metrics": metrics,
                "timestamp": datetime.now(),
                "model_path": self.eval_model_path.get(),
                "environment": self.eval_env_var.get(),
                "robot": self.eval_robot_var.get(),
                "episodes": int(self.eval_episodes_var.get()),
                "deterministic": self.eval_deterministic_var.get(),
            }

            self.evaluation_data["current_evaluation"] = evaluation_record
            self.evaluation_data["evaluation_history"].append(evaluation_record)

            # Atualizar histórico
            self._update_history_listbox()

            messagebox.showinfo("Sucesso", "Avaliação concluída com sucesso!")

        except Exception as e:
            self.logger.exception("Erro ao exibir resultados")
            messagebox.showerror("Erro", f"Erro ao exibir resultados: {e}")

    def _update_evaluation_plots(self, metrics):
        """Atualiza os gráficos de avaliação com novos dados"""
        try:
            times = metrics.get("total_times", [])
            successes = [1] * metrics.get("success_count", 0) + [0] * (len(times) - metrics.get("success_count", 0))

            # Limpar gráficos
            for ax in self.axs_evaluation.flat:
                ax.clear()

            # Gráfico de distribuição de tempos
            if times:
                self.axs_evaluation[0, 0].hist(times, bins=min(10, len(times)), alpha=0.7, color="blue", edgecolor="black")
                self.axs_evaluation[0, 0].axvline(metrics.get("avg_time", 0), color="red", linestyle="--", label=f'Média: {metrics["avg_time"]:.2f}s')
            self.axs_evaluation[0, 0].set_title("Distribuição de Tempos")
            self.axs_evaluation[0, 0].set_ylabel("Frequência")
            self.axs_evaluation[0, 0].set_xlabel("Tempo (s)")
            self.axs_evaluation[0, 0].legend()
            self.axs_evaluation[0, 0].grid(True, alpha=0.3)

            # Gráfico de sucesso por episódio
            if successes:
                colors = ["green" if s == 1 else "red" for s in successes]
                bars = self.axs_evaluation[0, 1].bar(range(len(successes)), successes, color=colors, alpha=0.7)
                self.axs_evaluation[0, 1].set_title("Sucesso por Episódio")
                self.axs_evaluation[0, 1].set_ylabel("Sucesso (1=Sim, 0=Não)")
                self.axs_evaluation[0, 1].set_xlabel("Episódio")
                self.axs_evaluation[0, 1].set_yticks([0, 1])
                self.axs_evaluation[0, 1].set_yticklabels(["Falha", "Sucesso"])
                self.axs_evaluation[0, 1].grid(True, alpha=0.3)

            # Gráfico de progressão temporal
            if times:
                self.axs_evaluation[1, 0].plot(range(len(times)), times, "o-", color="orange", markersize=4)
                self.axs_evaluation[1, 0].axhline(y=metrics.get("avg_time", 0), color="red", linestyle="--", label=f'Média: {metrics["avg_time"]:.2f}s')
                self.axs_evaluation[1, 0].set_title("Progressão de Tempos")
                self.axs_evaluation[1, 0].set_ylabel("Tempo (s)")
                self.axs_evaluation[1, 0].set_xlabel("Episódio")
                self.axs_evaluation[1, 0].legend()
                self.axs_evaluation[1, 0].grid(True, alpha=0.3)

            # Gráfico de métricas consolidadas
            metrics_names = ["Taxa Sucesso", "Tempo Médio", "Desvio Padrão"]
            metrics_values = [metrics.get("success_rate", 0) * 100, metrics.get("avg_time", 0), metrics.get("std_time", 0)]
            colors = ["green", "blue", "orange"]
            bars = self.axs_evaluation[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
            self.axs_evaluation[1, 1].set_title("Métricas Consolidadas")
            self.axs_evaluation[1, 1].set_ylabel("Valor")
            self.axs_evaluation[1, 1].grid(True, alpha=0.3)

            # Adicionar valores nas barras
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                self.axs_evaluation[1, 1].text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.2f}", ha="center", va="bottom")

            self.fig_evaluation.tight_layout()
            self.canvas_evaluation.draw_idle()

        except Exception as e:
            self.logger.exception("Erro ao atualizar gráficos de avaliação")

    def _update_history_listbox(self):
        """Atualiza a lista de histórico de avaliações"""
        self.history_listbox.delete(0, tk.END)
        for i, evaluation in enumerate(self.evaluation_data["evaluation_history"]):
            timestamp = evaluation["timestamp"].strftime("%H:%M:%S")
            model_name = os.path.basename(evaluation["model_path"])
            success_rate = evaluation["metrics"].get("success_rate", 0) * 100
            display_text = f"{timestamp} - {model_name} - {success_rate:.1f}% sucesso"
            self.history_listbox.insert(tk.END, display_text)

    def load_historical_evaluation(self):
        """Carrega uma avaliação do histórico"""
        selection = self.history_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione uma avaliação do histórico.")
            return

        try:
            evaluation = self.evaluation_data["evaluation_history"][selection[0]]
            self._display_evaluation_results(evaluation["metrics"])
            messagebox.showinfo("Sucesso", "Avaliação histórica carregada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar avaliação: {e}")

    def clear_history(self):
        """Limpa o histórico de avaliações"""
        if self.evaluation_data["evaluation_history"]:
            self.evaluation_data["evaluation_history"].clear()
            self.history_listbox.delete(0, tk.END)
            self.logger.info("Histórico de avaliações limpo")

    def export_evaluation_results(self):
        """Exporta os resultados da avaliação para arquivo CSV"""
        if not self.evaluation_data["current_evaluation"]:
            messagebox.showwarning("Aviso", "Nenhum resultado de avaliação para exportar.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                title="Salvar Resultados da Avaliação",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )

            if filename:
                # Garantir que o diretório existe
                ensure_directory(os.path.dirname(filename))

                # Exportar para CSV
                evaluation = self.evaluation_data["current_evaluation"]
                metrics = evaluation["metrics"]
                
                # Criar DataFrame com dados completos
                data = {
                    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "model_path": [evaluation["model_path"]],
                    "environment": [evaluation["environment"]],
                    "robot": [evaluation["robot"]],
                    "episodes": [evaluation["episodes"]],
                    "deterministic": [evaluation["deterministic"]],
                    "avg_time": [metrics.get("avg_time", 0)],
                    "std_time": [metrics.get("std_time", 0)],
                    "success_rate": [metrics.get("success_rate", 0)],
                    "success_count": [metrics.get("success_count", 0)],
                    "num_episodes": [metrics.get("num_episodes", 0)],
                }
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding='utf-8')

                messagebox.showinfo("Sucesso", f"Resultados exportados para:\n{filename}")
                self.logger.info(f"Resultados de avaliação exportados: {filename}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar resultados: {e}")
            self.logger.exception("Erro ao exportar resultados")

    def export_evaluation_plots(self):
        """Exporta os gráficos de avaliação como imagens PNG separadas"""
        if not self.evaluation_data["current_evaluation"]:
            messagebox.showwarning("Aviso", "Nenhum gráfico para exportar.")
            return

        try:
            directory = filedialog.askdirectory(title="Selecione onde salvar os gráficos")
            if not directory:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(self.evaluation_data["current_evaluation"]["model_path"])
            environment = self.eval_env_var.get()

            metrics = self.evaluation_data["current_evaluation"]["metrics"]
            times = metrics.get("total_times", [])
            successes = [1] * metrics.get("success_count", 0) + [0] * (len(times) - metrics.get("success_count", 0))

            # Gráfico 1: Distribuição de Tempos
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            if times:
                ax1.hist(times, bins=min(10, len(times)), alpha=0.7, color="blue", edgecolor="black")
                ax1.axvline(metrics.get("avg_time", 0), color="red", linestyle="--", label=f'Média: {metrics["avg_time"]:.2f}s')
            ax1.set_title("Distribuição de Tempos")
            ax1.set_ylabel("Frequência")
            ax1.set_xlabel("Tempo (s)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            fig1.savefig(os.path.join(directory, f"distribuicao_tempos_{timestamp}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig1)

            # Gráfico 2: Sucesso por Episódio
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            if successes:
                colors = ["green" if s == 1 else "red" for s in successes]
                ax2.bar(range(len(successes)), successes, color=colors, alpha=0.7)
                ax2.set_title("Sucesso por Episódio")
                ax2.set_ylabel("Sucesso (1=Sim, 0=Não)")
                ax2.set_xlabel("Episódio")
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(["Falha", "Sucesso"])
                ax2.grid(True, alpha=0.3)
            fig2.savefig(os.path.join(directory, f"sucesso_episodio_{timestamp}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig2)

            # Gráfico 3: Progressão de Tempos
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            if times:
                ax3.plot(range(len(times)), times, "o-", color="orange", markersize=4)
                ax3.axhline(y=metrics.get("avg_time", 0), color="red", linestyle="--", label=f'Média: {metrics["avg_time"]:.2f}s')
                ax3.set_title("Progressão de Tempos")
                ax3.set_ylabel("Tempo (s)")
                ax3.set_xlabel("Episódio")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            fig3.savefig(os.path.join(directory, f"progressao_tempos_{timestamp}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig3)

            # Gráfico 4: Métricas Consolidadas
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            metrics_names = ["Taxa Sucesso", "Tempo Médio", "Desvio Padrão"]
            metrics_values = [metrics.get("success_rate", 0) * 100, metrics.get("avg_time", 0), metrics.get("std_time", 0)]
            colors = ["green", "blue", "orange"]
            bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
            ax4.set_title("Métricas Consolidadas")
            ax4.set_ylabel("Valor")
            ax4.grid(True, alpha=0.3)

            # Adicionar valores nas barras
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.2f}", ha="center", va="bottom")

            fig4.savefig(os.path.join(directory, f"metricas_consolidadas_{timestamp}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig4)

            messagebox.showinfo("Sucesso", f"4 gráficos exportados como PNG para:\n{directory}")
            self.logger.info(f"Gráficos de avaliação exportados: {directory}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar gráficos: {e}")
            self.logger.exception("Erro ao exportar gráficos")

    def _plot_to_export_figure(self, axs, metrics):
        """Plota dados nos eixos fornecidos para exportação"""
        # Replicar a lógica de plotagem do _update_evaluation_plots
        times = metrics.get("total_times", [])
        successes = [1] * metrics.get("success_count", 0) + [0] * (len(times) - metrics.get("success_count", 0))

        # Gráfico de distribuição de tempos
        if times:
            axs[0, 0].hist(times, bins=min(10, len(times)), alpha=0.7, color="blue", edgecolor="black")
            axs[0, 0].axvline(metrics.get("avg_time", 0), color="red", linestyle="--", label=f'Média: {metrics["avg_time"]:.2f}s')
        axs[0, 0].set_title("Distribuição de Tempos")
        axs[0, 0].set_ylabel("Frequência")
        axs[0, 0].set_xlabel("Tempo (s)")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

    def start(self):
        """Inicializa a aba de avaliação"""
        self.logger.info("Aba de avaliação inicializada")

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()
