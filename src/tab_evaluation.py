# tab_evaluation.py
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import sys
import pandas as pd
import multiprocessing
import shutil
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import common_tab
import train_process


class EvaluationTab(common_tab.GUITab):
    def __init__(self, gui, device, logger, reward_system, notebook):
        super().__init__(gui, device, logger, reward_system, notebook)

        self.frame = ttk.Frame(notebook)
        self.device = device

        # Componentes da UI
        self.metrics_text = None
        self.fig_evaluation = None
        self.axs_evaluation = None
        self.canvas_evaluation = None

        self.setup_ui()
        self.setup_ipc()

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
        row1_frame.pack(fill=tk.X)

        ttk.Label(row1_frame, text="Modelo para Avaliar:").grid(row=0, column=0, sticky=tk.W)
        self.eval_model_path = tk.StringVar()
        ttk.Entry(row1_frame, textvariable=self.eval_model_path, width=120).grid(row=0, column=1, padx=5)
        self.load_model_btn = ttk.Button(row1_frame, text="Carregar modelo", command=self.browse_evaluation_model)
        self.load_model_btn.grid(row=0, column=2, padx=5)

        # Linha 1.5: Descrição
        description_frame = ttk.Frame(control_frame)
        description_frame.pack(fill=tk.X)
        self.model_description_label = ttk.Label(description_frame, text="Nenhum modelo carregado")
        self.model_description_label.grid(row=0, column=0, sticky=tk.W)

        # Linha 2: Configurações de avaliação
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, pady=5)

        self.create_environment_selector(row2_frame, column=0)

        self.create_robot_selector(row2_frame, column=2, enabled=False)

        ttk.Label(row2_frame, text="Episódios:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.eval_episodes_var = tk.StringVar(value="20")
        self.episodes_spinbox = ttk.Spinbox(row2_frame, from_=0, to=1e7, textvariable=self.eval_episodes_var, width=8)
        self.episodes_spinbox.grid(row=0, column=5, padx=5)

        self.create_pause_btn(row2_frame, 6)
        self.create_stop_btn(row2_frame, 7)
        self.create_seed_selector(row2_frame, column=8)

        # Linha 3: Botões de controle
        row3_frame = ttk.Frame(control_frame)
        row3_frame.pack(fill=tk.X)

        self.eval_start_btn = ttk.Button(row3_frame, text="Executar Avaliação", command=self.start_evaluation, width=20, state=tk.DISABLED)
        self.eval_start_btn.grid(row=0, column=0)

        self.eval_deterministic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3_frame, text="Modo Determinístico", variable=self.eval_deterministic_var).grid(row=0, column=1, padx=5)

        self.create_dpg_selector(row3_frame, column=3)
        self.create_enable_visualization_selector(row3_frame, column=4)
        self.create_real_time_selector(row3_frame, column=5)
        self.create_camera_selector(row3_frame, column=6)

        # Botões de exportação
        self.save_results_btn = ttk.Button(row3_frame, text="Salvar Avaliação", command=self.save_evaluation_results, state=tk.DISABLED)
        self.save_results_btn.grid(row=0, column=8, padx=5)
        self.load_results_btn = ttk.Button(row3_frame, text="Carregar Avaliação", command=self.load_evaluation_results)
        self.load_results_btn.grid(row=0, column=9, padx=5)
        self.export_plot_btn = ttk.Button(row3_frame, text="Salvar Gráficos", command=self.export_evaluation_plots, state=tk.DISABLED)
        self.export_plot_btn.grid(row=0, column=10, padx=5)

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

        # Aba de gráfico dinâmico
        dynamic_graph_frame = ttk.Frame(results_notebook)
        results_notebook.add(dynamic_graph_frame, text="Gráfico Dinâmico")

        # Controles para o gráfico dinâmico
        controls_frame = ttk.Frame(dynamic_graph_frame)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(controls_frame, text="Episódio:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dynamic_episode_var = tk.IntVar(value=1)
        self.dynamic_episode_spinbox = ttk.Spinbox(controls_frame, from_=1, to=1e7, textvariable=self.dynamic_episode_var, width=8, state=tk.DISABLED, command=self.update_dynamic_plot)
        self.dynamic_episode_spinbox.grid(row=0, column=1, padx=5)
        self.dynamic_episode_spinbox.bind("<Return>", lambda event: self.update_dynamic_plot())

        ttk.Label(controls_frame, text="Dados para plotar:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.dynamic_data_var = tk.StringVar()
        self.dynamic_data_combobox = ttk.Combobox(controls_frame, textvariable=self.dynamic_data_var, state=tk.DISABLED)
        self.dynamic_data_combobox.grid(row=0, column=3, padx=5)
        self.dynamic_data_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_dynamic_plot())

        # Gráfico dinâmico
        self.fig_dynamic, self.ax_dynamic = plt.subplots(figsize=(10, 6))
        self.canvas_dynamic = FigureCanvasTkAgg(self.fig_dynamic, master=dynamic_graph_frame)
        self.canvas_dynamic.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_evaluation_plots()

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
        session_dir = filedialog.askdirectory(title="Selecione a pasta do treinamento", initialdir=utils.TRAINING_DATA_PATH)

        if not session_dir:
            return

        self.eval_model_path.set(session_dir)
        self.logger.info(f"Modelo selecionado para avaliação: {os.path.basename(session_dir)}")

        # Carregar dados do treinamento
        training_data = self._load_training_data_file(session_dir)

        # Restaurar dados do treinamento
        self._restore_training_data(training_data, session_dir)

        self.unlock_gui()

    def _restore_training_data(self, training_data, session_dir):
        """Restaura dados do treinamento carregado"""
        session_info = training_data["session_info"]
        self.logger.info(f"Treinamento carregado:\n{session_info}")

        self.current_robot = session_info["robot"]
        self.robot_var.set(self.current_robot)

        model_description = (
            f"Agente {session_info['algorithm']}"
            f" | Ambiente: {session_info['environment']}"
            f" | Steps: {session_info['total_steps']}"
            f" | Seed: {session_info['seed']}"
            f" | Episódios: {session_info['total_episodes']}"
            f" | Salvo em: {datetime.fromisoformat(session_info['save_time']).strftime("%d/%m/%Y %H:%M:%S")}"
        )

        self.model_description_label.config(text=model_description)

    def start_evaluation(self):
        """Inicia a avaliação do modelo selecionado usando validação do utils"""
        agent_model_folder = self.eval_model_path.get()

        try:
            if not agent_model_folder or not os.path.exists(agent_model_folder):
                raise ValueError("Selecione um modelo válido para avaliação.")

            agent_model_path = self._find_agent_model(agent_model_folder)

        except ValueError as e:
            self.logger.exception("Erro ao iniciar avaliação")
            messagebox.showerror("Erro", str(e))
            return

        self.lock_gui()
        self.save_results_btn.config(state=tk.DISABLED)
        self.export_plot_btn.config(state=tk.DISABLED)
        self._run_evaluation(agent_model_path)

    def _run_evaluation(self, agent_model_path):
        """Executa a avaliação em thread separada"""
        try:
            self.current_env = self.env_var.get()
            self.current_robot = self.robot_var.get()

            episodes = int(self.eval_episodes_var.get())
            deterministic = self.eval_deterministic_var.get()

            shutil.rmtree(utils.TEMP_EVALUATION_SAVE_PATH)
            utils.ensure_directory(utils.TEMP_EVALUATION_SAVE_PATH)

            pause_val = multiprocessing.Value("b", 0)
            exit_val = multiprocessing.Value("b", 0)
            enable_visualization_val = multiprocessing.Value("b", self.enable_visualization_var.get())
            realtime_val = multiprocessing.Value("b", self.real_time_var.get())
            camera_selection_val = multiprocessing.Value("i", self.camera_selection_int)
            config_changed_val = multiprocessing.Value("b", 0)

            self.pause_values.append(pause_val)
            self.exit_values.append(exit_val)
            self.enable_visualization_values.append(enable_visualization_val)
            self.enable_real_time_values.append(realtime_val)
            self.camera_selection_values.append(camera_selection_val)
            self.config_changed_values.append(config_changed_val)

            environment_settings = self.get_environment_settings(self.env_var.get())
            initial_episode = 0

            self.logger.info(f"Iniciando avaliação: {agent_model_path} no ambiente {self.current_env}")
            self.logger.info(f"Configuração: {episodes} episódios, modo {'determinístico' if deterministic else 'estocástico'}")

            p = multiprocessing.Process(
                target=train_process.process_runner,
                args=(
                    self.current_env,
                    environment_settings,
                    self.current_robot,
                    None,
                    self.ipc_queue,
                    self.ipc_queue_main_to_process,
                    self.reward_system,
                    pause_val,
                    exit_val,
                    enable_visualization_val,
                    realtime_val,
                    camera_selection_val,
                    config_changed_val,
                    self.seed_var.get(),
                    self.device,
                    initial_episode,
                    agent_model_path,
                    self.enable_dpg_var.get(),
                    episodes,
                    deterministic,
                ),
            )
            p.start()
            self.processes.append(p)

            self.logger.info(f"Processo de avaliação iniciado: {self.current_env} + {self.current_robot}")

        except Exception as e:
            self.logger.exception("Erro na avaliação")
            messagebox.showerror("Erro", f"Erro na avaliação: {e}")
            self.unlock_gui()

    def _display_evaluation_results(self):
        """Exibe os resultados da avaliação na interface"""
        try:
            metrics = self.metrics_data["extra_metrics"]

            # Atualizar métricas textuais
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)

            success_rate = metrics.get("success_rate", 0) * 100
            avg_time = metrics.get("avg_time", 0)
            std_time = metrics.get("std_time", 0)
            success_count = metrics.get("success_count", 0)
            num_episodes = metrics.get("num_episodes", 0)
            total_rewards = metrics.get("total_rewards", [])

            results_text = (
                "=== RESULTADOS DA AVALIAÇÃO ===\n\n"
                f"Estatísticas Gerais:\n"
                f"• Modelo: {self.eval_model_path.get()}\n"
                f"• Ambiente: {self.env_var.get()}\n"
                f"• Robô: {self.robot_var.get()}\n"
                f"• Episódios executados: {num_episodes}\n"
                f"• Modo: {'Determinístico' if self.eval_deterministic_var.get() else 'Estocástico'}\n\n"
                f"Métricas de Performance:\n"
                f"• Taxa de sucesso: {success_rate:.1f}% ({success_count}/{num_episodes})\n"
                f"• Tempo médio: {avg_time:.2f} ± {std_time:.2f} segundos\n"
                f"• Melhor tempo: {min(metrics.get('total_times', [0])):.2f}s\n"
                f"• Pior tempo: {max(metrics.get('total_times', [0])):.2f}s\n"
                f"• Recompensa média: {sum(total_rewards)/len(total_rewards) if total_rewards else 0:.2f}\n\n"
                f"Distribuição de Tempos:\n"
            )

            times = metrics.get("total_times", [])
            if times:
                for i, time_val in enumerate(times, 1):
                    status = "✓" if i <= success_count else "✗"
                    results_text += f"• Episódio {i}: {time_val:.2f}s {status}\n"

            # Análise de performance
            results_text += (
                "\nAnálise:\n"
                f"• Performance: "
                f"{'EXCELENTE' if success_rate >= 90 else 'BOA' if success_rate >= 70 else 'REGULAR' if success_rate >= 50 else 'RUIM'}\n"
                f"• Consistência: "
                f"{'ALTA' if std_time < avg_time * 0.1 else 'MÉDIA' if std_time < avg_time * 0.2 else 'BAIXA'}\n"
            )

            results_text += "\n=== DADOS BRUTOS ===\n\n"
            data_to_print = {k: v for k, v in self.metrics_data.items() if k != "episodes"}
            results_text += json.dumps(data_to_print, indent=4, ensure_ascii=False)

            self.metrics_text.insert(1.0, results_text)
            self.metrics_text.config(state=tk.DISABLED)

            # Atualizar gráficos
            self._update_evaluation_plots(metrics)

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

    def save_evaluation_results(self):
        try:
            self.logger.info("Salvando dados de avaliação")

            save_path = filedialog.asksaveasfilename(
                title="Selecione o arquivo JSON para salvar os dados",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=os.path.expanduser("~"),
                initialfile="evaluation_data.json",
            )

            shutil.copy2(self.metrics_path, save_path)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar dados de avaliação: {e}")
            self.logger.exception("Erro ao salvar dados de avaliação")

    def load_evaluation_results(self):
        try:
            self.logger.info("Carregando dados de avaliação")

            self.metrics_path = filedialog.askopenfilename(
                title="Selecione o arquivo JSON para carregar os dados", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=os.path.expanduser("~")
            )

            self.load_metrics()

            hyperparameters = self.metrics_data["hyperparameters"]

            self.eval_model_path.set(hyperparameters["model_path"])
            self.env_var.set(hyperparameters["selected_environment"])
            self.robot_var.set(hyperparameters["selected_robot"])
            self.eval_episodes_var.set(hyperparameters["episodes"])
            self.eval_deterministic_var.set(hyperparameters["deterministic"])
            self.seed_var.set(hyperparameters["seed"])
            self.enable_dpg_var.set(hyperparameters["enable_dpg"])

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados de avaliação: {e}")
            self.logger.exception("Erro ao carregar dados de avaliação")

    def export_evaluation_plots(self):
        """Exporta os gráficos de avaliação como imagens PNG separadas"""
        try:
            directory = filedialog.askdirectory(title="Selecione onde salvar os gráficos")
            if not directory:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            metrics = self.metrics_data["extra_metrics"]
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

    def update_dynamic_plot(self):
        try:
            self.logger.info("Atualizando gráfico dinâmico")
            episode = self.dynamic_episode_var.get()
            selected_data = self.dynamic_data_var.get()

            self.logger.info(f"episode: {episode}")
            self.logger.info(f"selected_data: {selected_data}")

        except Exception as e:
            self.logger.exception("Erro ao atualizar plot dinâmico")

    def pause_training(self, force_pause=False):
        """Pausa ou retoma o treinamento"""
        if not self.pause_values:
            self.logger.warning("Nenhum processo de treinamento ativo.")
            return

        try:
            if self.pause_values[-1].value and not force_pause:
                self.logger.info("Retomando treinamento.")
                self.pause_values[-1].value = 0
                self.pause_btn.config(text="Pausar")

            else:
                self.logger.info("Pausando treinamento.")
                self.pause_values[-1].value = 1
                self.pause_btn.config(text="Retomar")

            self.last_pause_value = self.pause_values[-1].value
            self.config_changed_values[-1].value = 1

        except Exception as e:
            self.logger.exception("Erro ao pausar/retomar treinamento")

    def ipc_runner(self):
        """Thread para monitorar a fila IPC e atualizar logs"""
        try:
            while True:
                try:
                    msg = self.ipc_queue.get(timeout=1.0)

                    if msg is None:
                        self.logger.info("ipc_runner finalizando")
                        break

                    if isinstance(msg, str):
                        continue

                    data_type = msg.pop("type")

                    if data_type == "done":
                        self.logger.info("Processo de avaliação finalizado.")
                        self.unlock_gui()

                    elif data_type == "evaluation_complete":
                        self.metrics_path = msg["metrics_path"]
                        self.load_metrics()

                except queue.Empty:
                    # Timeout normal, continuar loop
                    continue
                except EOFError:
                    self.logger.info("IPC queue fechada (EOFError)")
                    break
                except Exception as e:
                    self.logger.exception("Erro ao receber mensagem IPC")
                    continue
        except Exception as e:
            self.logger.exception("Erro em ipc_runner")

            if not self.gui_closed:
                self.on_closing()

    def load_metrics(self):
        with open(self.metrics_path, "r") as f:
            self.metrics_data = json.load(f)

        self._display_evaluation_results()
        self.save_results_btn.config(state=tk.NORMAL)
        self.export_plot_btn.config(state=tk.NORMAL)
        self.dynamic_episode_spinbox.config(state=tk.NORMAL)
        self.dynamic_data_combobox.config(state=tk.NORMAL)

        num_episodes = self.metrics_data["extra_metrics"]["num_episodes"]
        self.dynamic_episode_spinbox.config(from_=1, to=num_episodes)
        self.dynamic_episode_var.set(1)

        available_keys = list(self.metrics_data["episodes"]["1"]["step_data"].keys())
        self.logger.info(f"available_keys: {available_keys}")
        self.dynamic_data_combobox["values"] = available_keys
        self.dynamic_data_var.set(available_keys[0])

        self.update_dynamic_plot()

    def start(self):
        """Inicializa a aba de avaliação"""
        self.logger.info("Aba de avaliação inicializada")

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()
