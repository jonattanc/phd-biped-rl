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
import queue


class TrainingGUI:
    def __init__(self, device="cpu"):
        self.root = tk.Tk()
        self.root.title("Cruzada Generalization - Training Dashboard")
        self.root.geometry("1200x800")

        self.device = device

        self.processes = []
        self.pause_values = []
        self.exit_values = []
        self.enable_real_time_values = []
        self.gui_log_queue = queue.Queue()
        self.ipc_queue = multiprocessing.Queue()
        self.ipc_thread = threading.Thread(target=self.ipc_runner, daemon=True)
        self.plot_data_lock = threading.Lock()
        self.gui_closed = False
        self.new_plot_data = False

        # Dados de treinamento:
        self.current_env = ""
        self.current_robot = ""
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": [],
            "imu_x": [], "imu_y": [], "imu_z": [],
            "roll": [], "pitch": [], "yaw": []}
        self.fig, self.axs = plt.subplots(3, figsize=(10, 8))
        self.canvas = None
        self.episode_logger = None
        self.hyperparams = {}
        self.logger = utils.get_logger()

        self.total_steps = 0
        self.steps_per_second = 0
        self.last_step_time = time.time()

        self.plot_titles = ["Recompensa por Episódio", "Duração do Episódio", "Distância Percorrida",
            "Posição IMU (X, Y, Z)", "Orientação (Roll, Pitch, Yaw)"]
        self.plot_ylabels = ["Recompensa", "Tempo (s)", "Distância (m)",
            "Posição (m)", "Ângulo (rad)"]
        self.plot_colors = ["blue", "orange", "green", "red", "purple", "brown"]
        self.plot_data_keys = ["rewards", "times", "distances", "imu_xyz", "rpy"]

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
        algorithms = ["TD3", "PPO"]
        self.algorithm_var = tk.StringVar(value=algorithms[0])
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var, values=algorithms)
        algorithm_combo.grid(row=0, column=1, padx=5)

        # Seleção de ambiente
        xacro_env_files = [file.replace(".xacro", "") for file in os.listdir(utils.ENVIRONMENT_PATH) if file.endswith(".xacro")]

        if "PR" in xacro_env_files:
            xacro_env_files.remove("PR")
            xacro_env_files.insert(0, "PR")

        if len(xacro_env_files) == 0:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ENVIRONMENT_PATH}.")
            self.root.destroy()
            return

        ttk.Label(control_frame, text="Ambiente:").grid(row=0, column=2, sticky=tk.W)
        self.env_var = tk.StringVar(value=xacro_env_files[0])
        env_combo = ttk.Combobox(control_frame, textvariable=self.env_var, values=xacro_env_files)
        env_combo.grid(row=0, column=3, padx=5)

        # Seleção de robô
        xacro_robot_files = [file.replace(".xacro", "") for file in os.listdir(utils.ROBOTS_PATH) if file.endswith(".xacro")]

        if len(xacro_robot_files) == 0:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ROBOTS_PATH}.")
            self.root.destroy()
            return

        ttk.Label(control_frame, text="Robô:").grid(row=0, column=4, sticky=tk.W)
        self.robot_var = tk.StringVar(value=xacro_robot_files[0])
        robot_combo = ttk.Combobox(control_frame, textvariable=self.robot_var, values=xacro_robot_files)
        robot_combo.grid(row=0, column=5, padx=5)

        # Botões de controle
        self.start_btn = ttk.Button(control_frame, text="Iniciar Treinamento", command=self.start_training)
        self.start_btn.grid(row=0, column=6, padx=5)

        self.pause_btn = ttk.Button(control_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=7, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=8, padx=5)

        self.save_btn = ttk.Button(control_frame, text="Salvar Snapshot", command=self.save_snapshot, state=tk.DISABLED)  # TODO: Revisar
        self.save_btn.grid(row=0, column=9, padx=5)

        self.visualize_btn = ttk.Button(control_frame, text="Ativar visualização", command=self.toggle_visualization, state=tk.DISABLED)
        self.visualize_btn.grid(row=0, column=10, padx=5)

        # Gráficos:
        graph_frame = ttk.LabelFrame(main_frame, text="Desempenho em Tempo Real", padding="10")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.fig, self.axs = plt.subplots(5, 1, figsize=(10, 10), constrained_layout=True, sharex=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_plots()
        self.canvas.draw_idle()

        stats_frame = ttk.Frame(control_frame)
        stats_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)
        self.steps_label = ttk.Label(stats_frame, text="Total Steps: 0 | Steps/s: 0.0")
        self.steps_label.pack(side=tk.LEFT)

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

    def _initialize_plots(self):
        """Inicializa os gráficos com títulos e configurações"""
        for i, (title, ylabel, color) in enumerate(zip(self.plot_titles, self.plot_ylabels, self.plot_colors)):
            self.axs[i].clear()
            if i == 3:  # Gráfico de posição IMU (X, Y, Z)
                self.axs[i].plot([], [], label="X", color="red", linestyle="-", markersize=3)
                self.axs[i].plot([], [], label="Y", color="green", linestyle="-", markersize=3)
                self.axs[i].plot([], [], label="Z", color="blue", linestyle="-", markersize=3)
            elif i == 4:  # Gráfico de orientação (Roll, Pitch, Yaw)
                self.axs[i].plot([], [], label="Roll", color="red", linestyle="-", markersize=3)
                self.axs[i].plot([], [], label="Pitch", color="green", linestyle="-", markersize=3)
                self.axs[i].plot([], [], label="Yaw", color="blue", linestyle="-", markersize=3)
            else:  # Gráficos normais
                self.axs[i].plot([], [], label=ylabel, color=color, linestyle="-", markersize=3)
            
            self.axs[i].set_title(title)
            self.axs[i].set_ylabel(ylabel)
            self.axs[i].grid(True, alpha=0.3)
            self.axs[i].legend()

        self.axs[-1].set_xlabel("Episódio")

    def _refresh_plots(self):
        """Atualiza os gráficos com os dados atuais"""
        try:
            if not self.episode_data["episodes"] or not self.new_plot_data:
                self.root.after(500, self._refresh_plots)
                return

            self.new_plot_data = False

            with self.plot_data_lock:
                for i, (title, ylabel, color, data_key) in enumerate(zip(self.plot_titles, self.plot_ylabels, self.plot_colors, self.plot_data_keys)):
                    self.axs[i].clear()
                    if i == 3:  # Gráfico de posição IMU
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data["imu_x"], label="X", color="red", linestyle="-", markersize=3)
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data["imu_y"], label="Y", color="green", linestyle="-", markersize=3)
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data["imu_z"], label="Z", color="blue", linestyle="-", markersize=3)
                    elif i == 4:  # Gráfico de orientação
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data["roll"], label="Roll", color="red", linestyle="-", markersize=3)
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data["pitch"], label="Pitch", color="green", linestyle="-", markersize=3)
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data["yaw"], label="Yaw", color="blue", linestyle="-", markersize=3)
                    else:  # Gráficos normais
                        self.axs[i].plot(self.episode_data["episodes"], self.episode_data[data_key], label=ylabel, color=color, linestyle="-", markersize=3)
                    
                    self.axs[i].set_title(title)
                    self.axs[i].set_ylabel(ylabel)
                    self.axs[i].legend()
                    self.axs[i].grid(True, alpha=0.3)

                self.axs[-1].set_xlabel("Episódio")

            self.canvas.draw()

        except Exception as e:
            self.logger.exception(f"Plot error")

        self.root.after(500, self._refresh_plots)

    def _update_step_counter(self):
        """Atualiza o contador de steps a cada segundo"""
        current_time = time.time()
        time_diff = current_time - self.last_step_time
        
        if time_diff >= 1.0:  # Atualizar a cada segundo
            if time_diff > 0:
                self.steps_per_second = self.total_steps / time_diff
            self.steps_label.config(text=f"Total Steps: {self.total_steps} | Steps/s: {self.steps_per_second:.1f}")
            self.total_steps = 0
            self.last_step_time = current_time
        
        self.root.after(100, self._update_step_counter)
    
    def _update_log_display(self):
        """Atualiza a exibição de logs"""
        if self.gui_log_queue.empty():
            self.root.after(500, self._update_log_display)
            return

        self.log_text.config(state=tk.NORMAL)

        try:
            for _ in range(500):
                message = self.gui_log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")

        except queue.Empty:
            pass

        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.after(500, self._update_log_display)


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
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": [],
            "imu_x": [], "imu_y": [], "imu_z": [],
            "roll": [], "pitch": [], "yaw": []}
        self.total_steps = 0
        self.steps_per_second = 0
        self.last_step_time = time.time()
        self._initialize_plots()
        self.canvas.draw()

        # Iniciar treinamento em processo separado
        pause_val = multiprocessing.Value("b", 0)
        exit_val = multiprocessing.Value("b", 0)

        if len(self.enable_real_time_values) > 0:
            realtime_val = multiprocessing.Value("b", self.enable_real_time_values[-1].value)

        else:
            realtime_val = multiprocessing.Value("b", 0)

        self.pause_values.append(pause_val)
        self.exit_values.append(exit_val)
        self.enable_real_time_values.append(realtime_val)

        p = multiprocessing.Process(
            target=train_process.process_runner, args=(self.current_env, self.current_robot, self.current_algorithm, self.ipc_queue, pause_val, exit_val, realtime_val, self.device)
        )
        p.start()
        self.processes.append(p)

        self.logger.info(f"Processo de treinamento iniciado: {self.current_env} + {self.current_robot} + {self.current_algorithm}")

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
            self.visualize_btn.config(text="Ativar visualização")

        else:
            self.logger.info("Ativando visualização em tempo real.")
            self.enable_real_time_values[-1].value = 1
            self.visualize_btn.config(text="Desativar visualização")

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
        try:
            while True:
                msg = self.ipc_queue.get()

                if msg is None:
                    self.logger.info("ipc_runner finalizando")
                    break

                if isinstance(msg, str):
                    self.gui_log_queue.put(msg)
                    continue

                data_type = msg.get("type")

                if data_type == "episode_data":
                    with self.plot_data_lock:
                        episode_num = msg["episode"]
                        self.episode_data["episodes"].append(episode_num)
                        self.episode_data["rewards"].append(msg["reward"])
                        self.episode_data["times"].append(msg["time"])
                        self.episode_data["distances"].append(msg["distance"])
                        self.episode_data["imu_x"].append(msg.get("imu_x", 0))
                        self.episode_data["imu_y"].append(msg.get("imu_y", 0))
                        self.episode_data["imu_z"].append(msg.get("imu_z", 0))
                        self.episode_data["roll"].append(msg.get("roll", 0))
                        self.episode_data["pitch"].append(msg.get("pitch", 0))
                        self.episode_data["yaw"].append(msg.get("yaw", 0))

                    self.new_plot_data = True

                elif data_type == "step_count":
                    # Atualizar contador de steps
                    self.total_steps += msg.get("steps", 0)
                
                elif data_type == "done":
                    self.logger.info("Processo de treinamento finalizado.")
                    self.start_btn.config(state=tk.NORMAL)
                    self.pause_btn.config(state=tk.DISABLED)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.save_btn.config(state=tk.NORMAL)
                    self.visualize_btn.config(state=tk.DISABLED)

                else:
                    self.logger.error(f"Tipo de dados desconhecido: {data_type}")

        except Exception as e:
            self.logger.exception("Erro em ipc_runner")

            if not self.gui_closed:
                self.on_closing()

    def on_closing(self):
        self.logger.info("Gui fechando")

        self.gui_closed = True
        self.ipc_queue.put(None)  # Sinaliza para a thread IPC terminar

        for v in self.exit_values:
            v.value = 1  # Sinaliza para os processos terminarem

        self.logger.info("Aguardando thread IPC terminar...")
        self.ipc_thread.join(timeout=10.0)

        if self.ipc_thread.is_alive():
            self.logger.warning("Thread ipc_thread did not terminate in time")

        self.logger.info("Aguardando processos de treinamento terminarem...")

        for p in self.processes:
            if p.is_alive():
                p.join(timeout=20.0)
                if p.is_alive():
                    self.logger.warning(f"Forcing termination of process {p.pid}")
                    p.terminate()

        self.logger.info("Todos os processos finalizados. Fechando GUI.")
        self.root.quit()  # Terminates the mainloop

    def start(self):
        self.ipc_thread.start()
        self.root.after(500, self._update_log_display)
        self.root.after(500, self._refresh_plots)
        self.root.after(100, self._update_step_counter)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
