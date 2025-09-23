# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import time
import utils
import train_process
import multiprocessing


class TrainingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cruzada Generalization - Training Dashboard")
        self.root.geometry("1200x800")

        self.processes = []
        self.pause_values = []
        self.exit_values = []
        self.enable_real_time_values = []
        self.ipc_queue = multiprocessing.Queue()
        self.training_data_queue = multiprocessing.Queue()
        self.ipc_thread = threading.Thread(target=self.ipc_runner, daemon=True)

        # Dados de treinamento:
        self.current_env = ""
        self.current_robot = ""
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": []}
        self.fig, self.axs = plt.subplots(3, figsize=(10, 8))
        self.canvas = None
        self.episode_logger = None
        self.hyperparams = {}
        self.logger = utils.get_logger()

        self.logger.info("Interface de treinamento inicializada.")
        self.setup_ui()

    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controle de Treinamento", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Seleção de algoritmo
        ttk.Label(control_frame, text="Algoritmo:").grid(row=0, column=0, sticky=tk.W)
        self.algorithm_var = tk.StringVar(value="PPO")
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var, values=["PPO", "TD3"])
        algorithm_combo.grid(row=0, column=1, padx=5)

        # Seleção de ambiente
        xacro_env_files = [file.replace(".xacro", "") for file in os.listdir(utils.ENVIRONMENT_PATH) if file.endswith(".xacro")]

        if len(xacro_env_files) == 0:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ENVIRONMENT_PATH}.")
            self.root.destroy()
            return

        ttk.Label(control_frame, text="Ambiente:").grid(row=0, column=0, sticky=tk.W)
        self.env_var = tk.StringVar(value=xacro_env_files[0])
        env_combo = ttk.Combobox(control_frame, textvariable=self.env_var, values=xacro_env_files)
        env_combo.grid(row=0, column=1, padx=5)

        # Seleção de robô
        xacro_robot_files = [file.replace(".xacro", "") for file in os.listdir(utils.ROBOTS_PATH) if file.endswith(".xacro")]

        if len(xacro_robot_files) == 0:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ROBOTS_PATH}.")
            self.root.destroy()
            return

        ttk.Label(control_frame, text="Robô:").grid(row=0, column=2, sticky=tk.W)
        self.robot_var = tk.StringVar(value=xacro_robot_files[0])
        robot_combo = ttk.Combobox(control_frame, textvariable=self.robot_var, values=xacro_robot_files)
        robot_combo.grid(row=0, column=3, padx=5)

        # Botões de controle
        self.start_btn = ttk.Button(control_frame, text="Iniciar Treinamento", command=self.start_training)
        self.start_btn.grid(row=0, column=4, padx=5)

        self.pause_btn = ttk.Button(control_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=5, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=6, padx=5)

        self.save_btn = ttk.Button(control_frame, text="Salvar Snapshot", command=self.save_snapshot, state=tk.DISABLED)  # TODO: Revisar
        self.save_btn.grid(row=0, column=7, padx=5)

        self.visualize_btn = ttk.Button(control_frame, text="Ativar tempo real", command=self.toggle_visualization, state=tk.DISABLED)  # TODO: Revisar
        self.visualize_btn.grid(row=0, column=8, padx=5)

        # Gráficos:
        graph_frame = ttk.LabelFrame(main_frame, text="Desempenho em Tempo Real", padding="10")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.fig, self.axs = plt.subplots(3, figsize=(10, 6))
        self.axs[0].set_title("Recompensa por Episódio")
        self.axs[1].set_title("Duração do Episódio (s)")
        self.axs[2].set_title("Distância Percorrida (m)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # Inicializar gráficos vazios
        self._initialize_plots()
        self.canvas.draw()

        # Logs:
        log_frame = ttk.LabelFrame(main_frame, text="Log de Treinamento", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10, rowspan=2)
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def start_training(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.visualize_btn.config(state=tk.NORMAL)

        self.current_env = self.env_var.get()
        self.current_robot = self.robot_var.get()
        self.current_algorithm = self.algorithm_var.get()

        # Limpar dados anteriores
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": []}
        self._initialize_plots()

        # Iniciar treinamento em processo separado
        pause_val = multiprocessing.Value("b", 0)
        exit_val = multiprocessing.Value("b", 0)
        realtime_val = multiprocessing.Value("b", 0)

        self.pause_values.append(pause_val)
        self.exit_values.append(exit_val)
        self.enable_real_time_values.append(realtime_val)

        p = multiprocessing.Process(target=self._training_process, args=(self.current_env, self.current_robot, self.current_algorithm, self.training_data_queue, pause_val, exit_val, realtime_val))
        p.start()
        self.processes.append(p)
        self._update_log_display(f"Iniciando treinamento: {self.current_algorithm} + {self.current_robot} + {self.current_env}")

    def _initialize_plots(self):
        """Inicializa os gráficos com títulos e configurações"""
        titles = ["Recompensa por Episódio", "Duração do Episódio (s)", "Distância Percorrida (m)"]
        ylabels = ["Recompensa", "Tempo (s)", "Distância (m)"]
        colors = ["blue", "orange", "green"]

        for i, (title, ylabel, color) in enumerate(zip(titles, ylabels, colors)):
            self.axs[i].clear()
            self.axs[i].set_title(title)
            self.axs[i].set_xlabel("Episódio")
            self.axs[i].set_ylabel(ylabel)
            self.axs[i].grid(True, alpha=0.3)

            # Plotar dados vazios inicialmente
            self.axs[i].plot([], [], label=ylabel, color=color, marker="o", linestyle="-", markersize=3)
            self.axs[i].legend()

    def _training_process(self, env_name, robot_name, algorithm, data_queue, pause_val, exit_val, realtime_val):
        """Processo de treinamento que envia dados para a GUI"""
        try:
            from simulation import Simulation
            from robot import Robot
            from environment import Environment
            from agent import Agent

            # Configurar ambiente
            robot = Robot(robot_name)
            env_obj = Environment(env_name)

            sim = Simulation(robot=robot, environment=env_obj, pause_value=pause_val, exit_value=exit_val, enable_real_time_value=realtime_val, enable_gui=False)

            # Configurar agente com callback para dados
            agent = Agent(env=sim, algorithm=algorithm)

            # Treinar
            agent.train(total_timesteps=100000)

        except Exception as e:
            data_queue.put({"type": "error", "message": f"Erro no treinamento: {e}"})

    def start_training(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.visualize_btn.config(state=tk.NORMAL)

        self.current_env = self.env_var.get()
        self.current_robot = self.robot_var.get()
        self.current_algorithm = self.algorithm_var.get()

        # Limpar dados anteriores
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": []}
        self._initialize_plots()
        self.canvas.draw()

        # Iniciar treinamento em processo separado
        pause_val = multiprocessing.Value("b", 0)
        exit_val = multiprocessing.Value("b", 0)
        realtime_val = multiprocessing.Value("b", 0)

        self.pause_values.append(pause_val)
        self.exit_values.append(exit_val)
        self.enable_real_time_values.append(realtime_val)

        p = multiprocessing.Process(
            target=train_process.process_runner, args=(self.current_env, self.current_robot, self.current_algorithm, self.ipc_queue, self.training_data_queue, pause_val, exit_val, realtime_val)
        )
        p.start()
        self.processes.append(p)

        self._update_log_display(f"Iniciando treinamento: {self.current_algorithm} + {self.current_robot} + {self.current_env}")
        self.logger.info(f"Processo de treinamento iniciado: {self.current_env} + {self.current_robot} + {self.current_algorithm}")

    def update_plots(self):
        """Atualiza os gráficos com novos dados da fila"""
        try:
            updated = False

            # Processar todos os dados disponíveis na fila
            while not self.training_data_queue.empty():
                try:
                    data = self.training_data_queue.get_nowait()

                    if data.get("type") == "episode_data":
                        # Adicionar dados do episódio
                        episode_num = data["episode"]
                        self.episode_data["episodes"].append(episode_num)
                        self.episode_data["rewards"].append(data["reward"])
                        self.episode_data["times"].append(data["time"])
                        self.episode_data["distances"].append(data["distance"])
                        updated = True

                    elif data.get("type") == "log":
                        # Atualizar logs
                        self._update_log_display(data["message"])

                except:
                    break  # Fila vazia

            if updated and self.episode_data["episodes"]:
                self._refresh_plots()

        except Exception as e:
            self.logger.error(f"Erro ao atualizar gráficos: {e}")

        # Agendar próxima atualização
        self.root.after(2000, self.update_plots)

    def _refresh_plots(self):
        """Atualiza os gráficos com os dados atuais"""
        if not self.episode_data["episodes"]:
            return

        titles = ["Recompensa por Episódio", "Duração do Episódio (s)", "Distância Percorrida (m)"]
        ylabels = ["Recompensa", "Tempo (s)", "Distância (m)"]
        colors = ["blue", "orange", "green"]
        data_keys = ["rewards", "times", "distances"]

        for i, (title, ylabel, color, data_key) in enumerate(zip(titles, ylabels, colors, data_keys)):
            self.axs[i].clear()
            self.axs[i].plot(self.episode_data["episodes"], self.episode_data[data_key], label=ylabel, color=color, marker="o", linestyle="-", markersize=3)
            self.axs[i].set_title(title)
            self.axs[i].set_xlabel("Episódio")
            self.axs[i].set_ylabel(ylabel)
            self.axs[i].legend()
            self.axs[i].grid(True, alpha=0.3)

        self.canvas.draw()

    def _update_log_display(self, message):
        """Atualiza a exibição de logs"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def pause_training(self):
        if not self.pause_values:
            self.logger.warning("pause_training: Nenhum processo de treinamento ativo.")
            return

        if self.pause_values[-1].value:
            self.logger.info("Retomando treinamento.")
            self.pause_values[-1].value = 0
            self.pause_btn.config(text="Pausar")

        else:
            self.logger.info("Pausando treinamento.")
            self.pause_values[-1].value = 1
            self.pause_btn.config(text="Retomar")

    def stop_training(self):
        if self.exit_values:
            self.exit_values[-1].value = 1

        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.visualize_btn.config(state=tk.DISABLED)

    def toggle_visualization(self):
        if not self.enable_real_time_values:
            self.logger.warning("toggle_visualization: Nenhum processo de treinamento ativo.")
            return

        if self.enable_real_time_values[-1].value:
            self.logger.info("Desativando visualização em tempo real.")
            self.enable_real_time_values[-1].value = 0
            self.visualize_btn.config(text="Ativar tempo real")

        else:
            self.logger.info("Ativando visualização em tempo real.")
            self.enable_real_time_values[-1].value = 1
            self.visualize_btn.config(text="Desativar tempo real")

    def save_snapshot(self):
        """Salva o modelo treinado e executa avaliação para gerar métricas de complexidade."""
        try:
            models_dir = "logs/data/models"
            os.makedirs(models_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = os.path.join(models_dir, f"model_{self.current_env}_{self.current_robot}_{timestamp}.zip")

            # TODO: Implementar lógica de salvamento do modelo
            messagebox.showinfo(
                "Salvar Snapshot",
                f"Funcionalidade de salvamento será implementada\n"
                f"Modelo: {self.current_algorithm}\n"
                f"Robô: {self.current_robot}\n"
                f"Ambiente: {self.current_env}\n"
                f"Arquivo: {model_filename}",
            )

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar snapshot: {e}")

    def ipc_runner(self):
        """Thread para monitorar a fila IPC e atualizar logs"""
        while True:
            msg = self.ipc_queue.get()

            if msg is None:
                break

            elif msg == "done":
                self.logger.info("Processo de treinamento finalizado.")
                self.start_btn.config(state=tk.NORMAL)
                self.pause_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.NORMAL)
                self.visualize_btn.config(state=tk.DISABLED)

            else:
                self.logger.info(f"Mensagem IPC: {msg}")

    def update_logs(self):
        """Atualiza a caixa de log com o arquivo de log principal"""
        self.root.after(10000, self.update_logs)

    def on_closing(self):
        self.logger.info("Gui fechada pelo usuário.")

        self.ipc_queue.put(None)  # Sinaliza para a thread IPC terminar

        for v in self.exit_values:
            v.value = 1  # Sinaliza para os processos terminarem

        self.logger.info("Aguardando thread IPC terminar...")
        self.ipc_thread.join()

        self.logger.info("Aguardando processos de treinamento terminarem...")

        for p in self.processes:
            if p.is_alive():
                p.join(timeout=20.0)
                if p.is_alive():
                    p.terminate()

        self.logger.info("Todos os processos finalizados. Fechando GUI.")
        self.root.quit()  # Terminates the mainloop

    def start(self):
        self.root.after(1000, self.update_plots)
        self.root.after(1000, self.update_logs)
        self.ipc_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
