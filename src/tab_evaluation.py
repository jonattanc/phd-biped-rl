# tab_evaluation.py
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import sys
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
        self.is_fast_td3 = False
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
        self.eval_model_path = tk.StringVar(value=utils.MODELS_DATA_PATH)
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

        ttk.Label(row2_frame, text="Executar até:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.eval_episodes_var = tk.StringVar(value="100 sucessos")
        ttk.Label(row2_frame, textvariable=self.eval_episodes_var, width=15).grid(row=0, column=5, padx=5)

        self.create_pause_btn(row2_frame, 6)
        self.create_stop_btn(row2_frame, 7)
        self.create_seed_selector(row2_frame, column=8)

        self.episode_count_label = ttk.Label(row2_frame, text="Episódios: 0")
        self.episode_count_label.grid(row=0, column=9, padx=5)

        # Linha 3: Botões de controle
        row3_frame = ttk.Frame(control_frame)
        row3_frame.pack(fill=tk.X)

        self.eval_start_btn = ttk.Button(row3_frame, text="Executar Avaliação", command=self.start_evaluation, width=20, state=tk.DISABLED)
        self.eval_start_btn.grid(row=0, column=0)
        # Verifica se há um modelo carregado inicialmente
        if self.eval_model_path.get() and self.eval_model_path.get().endswith('.zip'):
            self.eval_start_btn.config(state=tk.NORMAL)
        self.eval_deterministic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3_frame, text="Modo Determinístico", variable=self.eval_deterministic_var).grid(row=0, column=1, padx=5)

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
        self.dynamic_data_combobox = ttk.Combobox(controls_frame, textvariable=self.dynamic_data_var, state=tk.DISABLED, width=50)
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
        """Abre diálogo para selecionar modelo .zip para avaliação"""
        # Altere de askdirectory para askopenfilename
        model_file = filedialog.askopenfilename(
            title="Selecione o arquivo do modelo (.zip)",
            initialdir=utils.MODELS_DATA_PATH,
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")]
        )

        if not model_file:
            return

        # Agora salva o caminho do arquivo, não da pasta
        self.eval_model_path.set(model_file)

        # Tenta encontrar a pasta correspondente para carregar os dados de treinamento
        session_dir = os.path.dirname(model_file)

        # Carregar dados do treinamento se existirem - com tratamento de erro
        training_data = None
        try:
            training_data = self._load_training_data_file(session_dir)
        except FileNotFoundError:
            self.logger.info("Arquivo de dados de treinamento não encontrado. Apenas o modelo será carregado.")
        except Exception as e:
            self.logger.warning(f"Erro ao carregar dados de treinamento: {e}")

        if training_data:
            # Restaurar dados do treinamento
            self._restore_training_data(training_data, session_dir)
        else:
            # Se não encontrar dados de treinamento, apenas loga o modelo
            model_name = os.path.basename(model_file)
            self.model_description_label.config(text=f"Modelo: {model_name}")
            self.logger.info(f"Modelo carregado para avaliação: {model_name}")
            self.is_fast_td3 = False

        self.unlock_gui()

    def _find_agent_model(self, model_path):
        """Retorna o caminho do arquivo .zip"""
        # Se já for um arquivo .zip, retorna ele mesmo
        if model_path.endswith('.zip') and os.path.isfile(model_path):
            return model_path

        # Se for uma pasta, procura por .zip dentro
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.zip'):
                    return os.path.join(model_path, file)

        raise ValueError(f"Nenhum arquivo .zip encontrado em: {model_path}")
    
    def _restore_training_data(self, training_data, session_dir):
        """Restaura dados do treinamento carregado"""
        session_info = training_data["session_info"]
        self.logger.info(f"Treinamento carregado:\n{session_info}")

        self.current_robot = session_info["robot"]
        self.robot_var.set(self.current_robot)
        self.is_fast_td3 = "FASTTD3" in session_info["algorithm"].upper()

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
        """Inicia a avaliação do modelo .zip selecionado"""
        agent_model_file = self.eval_model_path.get()

        try:
            # Verifica se é um arquivo .zip válido
            if not agent_model_file or not os.path.exists(agent_model_file):
                raise ValueError("Selecione um arquivo .zip válido para avaliação.")

            if not agent_model_file.endswith('.zip'):
                raise ValueError("O arquivo selecionado deve ser um arquivo .zip")

            # Verifica se o arquivo de modelo existe diretamente
            if not os.path.isfile(agent_model_file):
                raise ValueError(f"Arquivo de modelo não encontrado: {agent_model_file}")

        except ValueError as e:
            self.logger.exception("Erro ao iniciar avaliação")
            messagebox.showerror("Erro", str(e))
            return

        self.lock_gui()
        self.save_results_btn.config(state=tk.DISABLED)
        self.export_plot_btn.config(state=tk.DISABLED)
        self.episode_count = self.update_episode_count(0)
        self.desired_successes = 100
        self._run_evaluation(agent_model_file)

    def _run_evaluation(self, agent_model_file):
        """Executa a avaliação em thread separada"""
        try:
            self.current_env = self.env_var.get()
            self.current_robot = self.robot_var.get()

            episodes = 10000
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

            initial_episode = 0

            self.logger.info(f"Iniciando avaliação: {agent_model_file} no ambiente {self.current_env}")
            self.logger.info(f"Configuração: {episodes} episódios, modo {'determinístico' if deterministic else 'estocástico'}")

            p = multiprocessing.Process(
                target=train_process.process_runner,
                args=(
                    self.current_env,
                    self.current_robot,
                    self.is_fast_td3,
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
                    agent_model_file,
                    episodes,
                    deterministic,
                    False,
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
        """Exibe os resultados da avaliação na interface e cria tabelas"""
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
            total_times = metrics.get("total_times", [])
            total_distances = []
            all_episodes_data = []

            # Coletar dados de todos os episódios
            for episode_num in range(1, num_episodes + 1):
                episode_key = str(episode_num)
                if episode_key in self.metrics_data["episodes"]:
                    episode_data = self.metrics_data["episodes"][episode_key]

                    if "episode_data" in episode_data:
                        distance = episode_data["episode_data"].get("distances", 0)
                        time_val = episode_data["episode_data"].get("times", 0)
                        success = False

                        if "episode_extra_data" in episode_data:
                            success = episode_data["episode_extra_data"].get("episode_success", False)
                        elif "episode_data" in episode_data:
                            success = episode_data["episode_data"].get("success", False)

                        total_distances.append(distance)

                        # Armazenar dados para tabelas
                        velocity = distance / time_val if time_val > 0 else 0
                        all_episodes_data.append({
                            "episode": episode_num,
                            "success": success,
                            "distance": distance,
                            "time": time_val,
                            "velocity": velocity
                        })

            # Criar tabela com os 100 primeiros sucessos
            first_100_successes = []
            for episode in all_episodes_data:
                if episode["success"] and len(first_100_successes) < 100:
                    first_100_successes.append(episode)

            # Criar tabela com todos os episódios (até alcançar 100 sucessos)
            # Para isso, vamos pegar todos os episódios até o episódio onde alcançamos 100 sucessos
            episodes_until_100_successes = []
            success_counter = 0
            for episode in all_episodes_data:
                episodes_until_100_successes.append(episode)
                if episode["success"]:
                    success_counter += 1
                    if success_counter >= 100:
                        break

            results_text = (
                "=== RESULTADOS DA AVALIAÇÃO ===\n\n"
                f"Estatísticas Gerais:\n"
                f"• Modelo: {self.eval_model_path.get()}\n"
                f"• Ambiente: {self.env_var.get()}\n"
                f"• Robô: {self.robot_var.get()}\n"
                f"• Episódios executados: {num_episodes}\n"
                f"• Sucessos alcançados: {success_count}\n"
                f"• Modo: {'Determinístico' if self.eval_deterministic_var.get() else 'Estocástico'}\n\n"
                f"Métricas de Performance:\n"
                f"• Taxa de sucesso: {success_rate:.1f}% ({success_count}/{num_episodes})\n"
                f"• Tempo médio: {avg_time:.2f} ± {std_time:.2f} segundos\n"
                f"• Melhor tempo: {min(total_times) if total_times else 0:.2f}s\n"
                f"• Pior tempo: {max(total_times) if total_times else 0:.2f}s\n"
                f"• Recompensa média: {sum(total_rewards)/len(total_rewards) if total_rewards else 0:.2f}\n\n"
            )

            # Adicionar tabela dos 100 primeiros sucessos
            if first_100_successes:
                results_text += "=== TABELA DOS 100 PRIMEIROS SUCESSOS ===\n"
                results_text += "Episódio | Distância (m) | Tempo (s) | Velocidade (m/s)\n"
                results_text += "-" * 60 + "\n"

                for episode_data in first_100_successes:
                    results_text += f"{episode_data['episode']:8d} | {episode_data['distance']:12.2f} | {episode_data['time']:10.2f} | {episode_data['velocity']:15.2f}\n"

                # Calcular médias para os 100 primeiros sucessos
                avg_distance = sum(ep["distance"] for ep in first_100_successes) / len(first_100_successes)
                avg_time_100 = sum(ep["time"] for ep in first_100_successes) / len(first_100_successes)
                avg_velocity = sum(ep["velocity"] for ep in first_100_successes) / len(first_100_successes)

                results_text += "-" * 60 + "\n"
                results_text += f"MÉDIA     | {avg_distance:12.2f} | {avg_time_100:10.2f} | {avg_velocity:15.2f}\n\n"

            # Adicionar tabela com todos os episódios até alcançar 100 sucessos
            if episodes_until_100_successes:
                results_text += "=== TABELA DE TODOS OS EPISÓDIOS (ATÉ 100 SUCESSOS) ===\n"
                results_text += "Episódio | Sucesso | Distância (m) | Tempo (s) | Velocidade (m/s)\n"
                results_text += "-" * 80 + "\n"

                for episode_data in episodes_until_100_successes:
                    success_symbol = "✓" if episode_data["success"] else "✗"
                    results_text += f"{episode_data['episode']:8d} | {success_symbol:8s} | {episode_data['distance']:12.2f} | {episode_data['time']:10.2f} | {episode_data['velocity']:15.2f}\n"

                # Calcular estatísticas
                successful_episodes = [ep for ep in episodes_until_100_successes if ep["success"]]
                failed_episodes = [ep for ep in episodes_until_100_successes if not ep["success"]]

                if successful_episodes:
                    avg_success_distance = sum(ep["distance"] for ep in successful_episodes) / len(successful_episodes)
                    avg_success_time = sum(ep["time"] for ep in successful_episodes) / len(successful_episodes)
                    avg_success_velocity = sum(ep["velocity"] for ep in successful_episodes) / len(successful_episodes)

                results_text += "-" * 80 + "\n"
                results_text += f"Total de episódios: {len(episodes_until_100_successes)}\n"
                results_text += f"Sucessos: {len(successful_episodes)} | Falhas: {len(failed_episodes)}\n"
                if successful_episodes:
                    results_text += f"Média dos sucessos - Distância: {avg_success_distance:.2f}m, Tempo: {avg_success_time:.2f}s, Velocidade: {avg_success_velocity:.2f}m/s\n\n"

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

            # Salvar as tabelas em arquivos separados
            self._save_tables_to_files(first_100_successes, episodes_until_100_successes)

        except Exception as e:
            self.logger.exception("Erro ao exibir resultados")
            messagebox.showerror("Erro", f"Erro ao exibir resultados: {e}")

    def _save_tables_to_files(self, first_100_successes, all_episodes):
        """Salva as tabelas em arquivos CSV para referência futura"""
        try:
            import csv
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            utils.ensure_directory(utils.ESPECIALISTA_DATA_PATH)
            current_env = self.env_var.get()

            # Salvar tabela dos 100 primeiros sucessos
            if first_100_successes:
                csv_filename_100 = f"{current_env}_100_sucessos_{timestamp}.csv"
                csv_path_100 = os.path.join(utils.ESPECIALISTA_DATA_PATH, csv_filename_100)
                with open(csv_path_100, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Episódio', 'Distância (m)', 'Tempo (s)', 'Velocidade (m/s)'])
                    for ep in first_100_successes:
                        writer.writerow([ep['episode'], f"{ep['distance']:.2f}", f"{ep['time']:.2f}", f"{ep['velocity']:.2f}"])
                self.logger.info(f"Tabela dos 100 primeiros sucessos salva em: {csv_path_100}")

            # Salvar tabela de todos os episódios
            if all_episodes:
                csv_filename_all = f"{current_env}_todos_episodios_{timestamp}.csv"
                csv_path_all = os.path.join(utils.ESPECIALISTA_DATA_PATH, csv_filename_all)
                with open(csv_path_all, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Episódio', 'Sucesso', 'Distância (m)', 'Tempo (s)', 'Velocidade (m/s)'])
                    for ep in all_episodes:
                        success_str = "Sim" if ep['success'] else "Não"
                        writer.writerow([ep['episode'], success_str, f"{ep['distance']:.2f}", f"{ep['time']:.2f}", f"{ep['velocity']:.2f}"])
                self.logger.info(f"Tabela de todos os episódios salva em: {csv_path_all}")

        except Exception as e:
            self.logger.error(f"Erro ao salvar tabelas: {e}")

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
                initialdir=os.path.join(os.path.expanduser("~"), "Desktop"),
                initialfile="evaluation_data.json",
            )

            if not save_path:
                return

            shutil.copy2(self.metrics_path, save_path)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar dados de avaliação: {e}")
            self.logger.exception("Erro ao salvar dados de avaliação")

    def load_evaluation_results(self):
        try:
            self.logger.info("Carregando dados de avaliação")

            self.metrics_path = filedialog.askopenfilename(
                title="Selecione o arquivo JSON para carregar os dados",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=utils.ESPECIALISTA_DATA_PATH,
            )

            if not self.metrics_path:
                return

            self.load_metrics()

            hyperparameters = self.metrics_data["hyperparameters"]

            self.eval_model_path.set(hyperparameters["model_path"])
            self.env_var.set(hyperparameters["selected_environment"])
            self.robot_var.set(hyperparameters["selected_robot"])
            self.eval_episodes_var.set(hyperparameters["episodes"])
            self.eval_deterministic_var.set(hyperparameters["deterministic"])
            self.seed_var.set(hyperparameters["seed"])

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

            if selected_data.startswith("episode_data_"):
                self.dynamic_episode_spinbox.config(state=tk.DISABLED)
                key = selected_data.replace("episode_data_", "")
                plot_raw_data = [value["episode_data"][key] for value in self.metrics_data["episodes"].values()]

            else:
                self.dynamic_episode_spinbox.config(state=tk.NORMAL)
                plot_raw_data = self.metrics_data["episodes"][str(episode)]["step_data"][selected_data]

            self.ax_dynamic.clear()

            if isinstance(plot_raw_data[0], list):
                plot_separated_data = list(map(list, zip(*plot_raw_data)))

                for i, series in enumerate(plot_separated_data):
                    x = range(1, len(series) + 1)
                    self.ax_dynamic.plot(x, series, label=f"{selected_data}[{i}]")

                self.ax_dynamic.legend()
                self.ax_dynamic.set_xlim(1, len(plot_separated_data[0]))

            else:
                plot_separated_data = plot_raw_data
                x = range(1, len(plot_separated_data) + 1)
                self.ax_dynamic.plot(x, plot_separated_data, color="blue")
                self.ax_dynamic.set_xlim(1, len(plot_separated_data))

            self.ax_dynamic.relim()
            self.ax_dynamic.autoscale_view()
            self.ax_dynamic.set_title(selected_data)
            self.ax_dynamic.set_ylabel(selected_data)
            self.ax_dynamic.grid(True, alpha=0.3)

            if selected_data.startswith("episode_data_"):
                self.ax_dynamic.set_xlabel("Episódio")

            else:
                self.ax_dynamic.set_xlabel("Step")

            self.fig_dynamic.tight_layout()
            self.canvas_dynamic.draw_idle()

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

                    elif data_type == "episode_data":
                        self.episode_count = self.update_episode_count(self.episode_count + 1)

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

    def update_episode_count(self, episode_count):
        self.episode_count_label.config(text=f"Episódios: {episode_count}/{self.eval_episodes_var.get()}")
        return episode_count

    def load_metrics(self):
        with open(self.metrics_path, "r", encoding="utf-8") as f:
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
        episode_data_keys = list(self.metrics_data["episodes"]["1"]["episode_data"].keys())
        available_keys += [f"episode_data_{key}" for key in episode_data_keys]
        self.logger.info(f"available_keys: {available_keys}")
        self.dynamic_data_combobox["values"] = available_keys
        self.dynamic_data_var.set(available_keys[0])

        self.update_dynamic_plot()

    def unlock_gui(self):
        """Desbloqueia a interface após carregar modelo"""
        super().unlock_gui()  # Chama o método da classe base

        # Habilita o botão de avaliação se um modelo .zip foi carregado
        model_path = self.eval_model_path.get()
        if model_path and model_path.endswith('.zip') and os.path.isfile(model_path):
            self.eval_start_btn.config(state=tk.NORMAL)
        else:
            self.eval_start_btn.config(state=tk.DISABLED)

    def start(self):
        """Inicializa a aba de avaliação"""
        self.logger.info("Aba de avaliação inicializada")
        # Verifica se há um modelo carregado ao iniciar
        if self.eval_model_path.get() and self.eval_model_path.get().endswith('.zip'):
            self.eval_start_btn.config(state=tk.NORMAL)
        else:
            self.eval_start_btn.config(state=tk.DISABLED)

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()
