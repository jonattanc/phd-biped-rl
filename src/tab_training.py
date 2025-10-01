# gui/training_tab.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import time
import multiprocessing
import queue
import json
import shutil
from datetime import datetime

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_process
from utils import ensure_directory, setup_ipc_logging, ENVIRONMENT_PATH, ROBOTS_PATH


class TrainingTab:
    def __init__(self, parent, device, logger):
        self.frame = ttk.Frame(parent, padding="10")
        self.device = device
        self.logger = logger

        # Dados de treinamento
        self.current_env = ""
        self.current_robot = ""
        self.current_algorithm = ""
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": [], "imu_x": [], "imu_y": [], "imu_z": [], "roll": [], "pitch": [], "yaw": []}

        # Controle de processos
        self.processes = []
        self.pause_values = []
        self.exit_values = []
        self.enable_real_time_values = []
        self.gui_log_queue = queue.Queue()
        self.ipc_queue = multiprocessing.Queue()
        self.ipc_thread = None
        self.plot_data_lock = threading.Lock()
        self.gui_closed = False
        self.after_ids = {}
        self.new_plot_data = False

        # Componentes da UI
        self.algorithm_var = None
        self.env_var = None
        self.robot_var = None
        self.start_btn = None
        self.pause_btn = None
        self.stop_btn = None
        self.save_training_btn = None
        self.load_training_btn = None
        self.export_plots_btn = None
        self.save_btn = None
        self.real_time_var = None
        self.real_time_check = None
        self.steps_label = None
        self.log_text = None

        # Gráficos
        self.fig = None
        self.axs = None
        self.canvas = None

        # Configurações de treinamento
        self.total_steps = None
        self.steps_per_second = 0
        self.training_start_time = time.time()
        self.current_episode = 0
        self.loaded_episode_count = 0
        self.training_data_dir = "training_data"
        self.current_training_session = None
        self.is_resuming = False
        self.resumed_session_dir = None
        self.hyperparams = {}

        # Configurações de plot
        self.plot_titles = ["Recompensa por Episódio", "Duração do Episódio", "Distância Percorrida", "Posição IMU (X, Y, Z)", "Orientação (Roll, Pitch, Yaw)"]
        self.plot_ylabels = ["Recompensa", "Tempo (s)", "Distância (m)", "Posição (m)", "Ângulo (rad)"]
        self.plot_colors = ["blue", "orange", "green", "red", "purple", "brown"]
        self.plot_data_keys = ["rewards", "times", "distances", "imu_xyz", "rpy"]

        # Configurar IPC logging
        setup_ipc_logging(self.logger, self.ipc_queue)

        self.setup_ui()
        self.setup_ipc()

    def setup_ui(self):
        """Configura a interface da aba de treinamento"""
        # Frame principal
        main_frame = ttk.Frame(self.frame)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controle de Treinamento", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Linha1: Seleções Principais
        row1_frame = ttk.Frame(control_frame)
        row1_frame.grid(row=0, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=0)

        # Seleção de algoritmo
        ttk.Label(row1_frame, text="Algoritmo:").grid(row=0, column=0, sticky=tk.W, padx=5)
        algorithms = ["FastTD3", "TD3", "PPO"]
        self.algorithm_var = tk.StringVar(value=algorithms[0])
        algorithm_combo = ttk.Combobox(row1_frame, textvariable=self.algorithm_var, values=algorithms, width=10)
        algorithm_combo.grid(row=0, column=1, padx=5)

        # Seleção de ambiente
        xacro_env_files = self._get_xacro_files(ENVIRONMENT_PATH)
        if not xacro_env_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {ENVIRONMENT_PATH}.")
            return

        ttk.Label(row1_frame, text="Ambiente:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.env_var = tk.StringVar(value=xacro_env_files[0])
        env_combo = ttk.Combobox(row1_frame, textvariable=self.env_var, values=xacro_env_files, width=10)
        env_combo.grid(row=0, column=3, padx=5)

        # Seleção de robô
        xacro_robot_files = self._get_xacro_files(ROBOTS_PATH)
        if not xacro_robot_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {ROBOTS_PATH}.")
            return

        ttk.Label(row1_frame, text="Robô:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.robot_var = tk.StringVar(value=xacro_robot_files[0])
        robot_combo = ttk.Combobox(row1_frame, textvariable=self.robot_var, values=xacro_robot_files, width=12)
        robot_combo.grid(row=0, column=5, padx=5)

        # Botões de controle
        self.start_btn = ttk.Button(row1_frame, text="Iniciar Treino", command=self.start_training, width=15)
        self.start_btn.grid(row=0, column=6, padx=5)

        self.pause_btn = ttk.Button(row1_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED, width=10)
        self.pause_btn.grid(row=0, column=7, padx=5)

        self.stop_btn = ttk.Button(row1_frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED, width=10)
        self.stop_btn.grid(row=0, column=8, padx=5)

        # Linha 2: Botões secundários e checkboxes
        row2_frame = ttk.Frame(control_frame)
        row2_frame.grid(row=1, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=5)

        self.save_training_btn = ttk.Button(row2_frame, text="Salvar Treino", command=self.save_training_data, state=tk.DISABLED, width=15)
        self.save_training_btn.grid(row=0, column=0, padx=5)

        self.load_training_btn = ttk.Button(row2_frame, text="Carregar Treino", command=self.load_training_data, width=15)
        self.load_training_btn.grid(row=0, column=1, padx=5)

        self.export_plots_btn = ttk.Button(row2_frame, text="Exportar Gráficos", command=self.export_plots, state=tk.DISABLED, width=15)
        self.export_plots_btn.grid(row=0, column=2, padx=5)

        self.save_btn = ttk.Button(row2_frame, text="Salvar Snapshot", command=self.save_snapshot, state=tk.DISABLED, width=15)
        self.save_btn.grid(row=0, column=3, padx=5)

        self.real_time_var = tk.BooleanVar(value=False)
        self.real_time_check = ttk.Checkbutton(row2_frame, text="Visualizar Robô", variable=self.real_time_var, command=self.toggle_real_time, state=tk.DISABLED, width=15)
        self.real_time_check.grid(row=0, column=4, padx=5)

        self.steps_label = ttk.Label(row2_frame, text=self.build_steps_label_text(0, 0))
        self.steps_label.grid(row=0, column=6, padx=5)

        # Gráficos
        graph_frame = ttk.LabelFrame(main_frame, text="Desempenho em Tempo Real", padding="10")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.fig, self.axs = plt.subplots(5, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_plots()

        # Logs
        log_frame = ttk.LabelFrame(main_frame, text="Log de Treinamento", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Configurar o grid dentro do log_frame para que o texto expanda
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configurar grid
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=3)  # Gráficos expandem
        main_frame.rowconfigure(2, weight=1)  # Logs expandem

        control_frame.columnconfigure(0, weight=1)

    def _get_xacro_files(self, directory):
        """Obtém lista de arquivos .xacro de um diretório usando utils"""
        try:
            if not os.path.exists(directory):
                self.logger.error(f"Diretório não encontrado: {directory}")
                return []

            files = [file.replace(".xacro", "") for file in os.listdir(directory) if file.endswith(".xacro")]

            # Ordenar com PR primeiro se existir
            if "PR" in files:
                files.remove("PR")
                files.insert(0, "PR")

            return files

        except Exception as e:
            self.logger.error(f"Erro ao listar arquivos .xacro: {e}")
            return []

    def setup_ipc(self):
        """Configura IPC para comunicação entre processos"""
        self.ipc_thread = threading.Thread(target=self.ipc_runner, daemon=True)
        self.ipc_thread.start()

    def _initialize_plots(self):
        """Inicializa os gráficos com títulos e configurações"""
        try:
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
            self.canvas.draw_idle()

        except Exception as e:
            self.logger.error(f"Erro ao inicializar gráficos: {e}")

        self.root.after(500, self._refresh_plots)

    def start_training(self):
        """Inicia um novo treinamento do zero"""
        self.start_btn.config(state=tk.DISABLED, text="Iniciando...")

        if self.is_resuming:
            self.logger.info(f"Retomando treinamento - current_episode: {self.current_episode}")
            self._resume_training()
        else:
            self.current_episode = 0
            self.loaded_episode_count = 0
            self.logger.info(f"Iniciando NOVO treinamento - current_episode: {self.current_episode}")
            self._start_new_training()

    def _start_new_training(self):
        """Inicia um novo treinamento do zero"""
        try:
            # Validar seleções
            if not self.env_var.get() or not self.robot_var.get() or not self.algorithm_var.get():
                raise ValueError("Selecione ambiente, robô e algoritmo antes de iniciar o treinamento.")

            self.current_env = self.env_var.get()
            self.current_robot = self.robot_var.get()
            self.current_algorithm = self.algorithm_var.get()

            # Limpar dados anteriores
            self.episode_data = {key: [] for key in self.episode_data.keys()}
            self.total_steps = None
            self.steps_per_second = 0
            self.training_start_time = time.time()
            self._initialize_plots()

            # Configurar botões
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.real_time_check.config(state=tk.NORMAL)

            # Iniciar treinamento em processo separado
            pause_val = multiprocessing.Value("b", 0)
            exit_val = multiprocessing.Value("b", 0)
            realtime_val = multiprocessing.Value("b", self.real_time_var.get())

            self.pause_values.append(pause_val)
            self.exit_values.append(exit_val)
            self.enable_real_time_values.append(realtime_val)

            p = multiprocessing.Process(
                target=train_process.process_runner, args=(self.current_env, self.current_robot, self.current_algorithm, self.ipc_queue, pause_val, exit_val, realtime_val, self.device)
            )
            p.start()
            self.processes.append(p)

            self.logger.info(f"Processo de treinamento iniciado: {self.current_env} + {self.current_robot} + {self.current_algorithm}")

            # Criar sessão de treinamento atual
            self.current_training_session = {
                "start_time": datetime.now(),
                "environment": self.current_env,
                "robot": self.current_robot,
                "algorithm": self.current_algorithm,
                "episode_data": self.episode_data.copy(),
                "hyperparams": self.hyperparams,
            }

            # Habilitar botões
            self.save_training_btn.config(state=tk.NORMAL)
            self.export_plots_btn.config(state=tk.NORMAL)
            self.start_btn.config(text="Iniciar Treinamento")

        except Exception as e:
            self.logger.error(f"Erro ao iniciar treinamento: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar treinamento: {e}")
            self.start_btn.config(state=tk.NORMAL, text="Iniciar Treinamento")

    def _resume_training(self):
        """Retoma um treinamento existente"""
        try:
            # Encontrar o modelo para retomada
            model_path = self._find_model_for_resume()
            if not model_path:
                return

            # Configurar botões
            self.start_btn.config(state=tk.DISABLED, text="Retomando...")
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.real_time_check.config(state=tk.NORMAL)

            # Iniciar processo de retomada
            pause_val = multiprocessing.Value("b", 0)
            exit_val = multiprocessing.Value("b", 0)
            realtime_val = multiprocessing.Value("b", self.real_time_var.get())

            self.pause_values.append(pause_val)
            self.exit_values.append(exit_val)
            self.enable_real_time_values.append(realtime_val)

            self.logger.info(f"Retomando treinamento - episódio: {self.current_episode}")

            p = multiprocessing.Process(
                target=train_process.process_runner_resume,
                args=(self.current_env, self.current_robot, self.current_algorithm, self.ipc_queue, pause_val, exit_val, realtime_val, self.device, model_path, self.current_episode),
            )
            p.start()
            self.processes.append(p)

            self.logger.info(f"Processo de treinamento retomado: {self.current_env} + {self.current_robot} + {self.current_algorithm}")
            self.start_btn.config(text="Iniciar Treinamento")
            self.is_resuming = False

        except Exception as e:
            self.logger.error(f"Erro ao retomar treinamento: {e}")
            messagebox.showerror("Erro", f"Erro ao retomar treinamento: {e}")
            self.start_btn.config(state=tk.NORMAL, text="Iniciar Treinamento")

    def _find_model_for_resume(self):
        """Encontra modelo para retomada usando busca flexível"""
        if not self.resumed_session_dir:
            raise ValueError("Diretório de sessão não definido para retomada.")

        # Buscar em models/
        models_dir = os.path.join(self.resumed_session_dir, "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".zip"):
                    model_path = os.path.join(models_dir, file)
                    self.logger.info(f"Modelo encontrado para retomada: {file}")
                    return model_path

        # Buscar no diretório principal
        for file in os.listdir(self.resumed_session_dir):
            if file.endswith(".zip"):
                model_path = os.path.join(self.resumed_session_dir, file)
                self.logger.info(f"Modelo encontrado no diretório principal: {file}")
                return model_path

        # Buscar recursivamente
        for root, dirs, files in os.walk(self.resumed_session_dir):
            for file in files:
                if file.endswith(".zip"):
                    model_path = os.path.join(root, file)
                    self.logger.info(f"Modelo encontrado em subdiretório: {model_path}")
                    return model_path

        raise FileNotFoundError(f"Nenhum modelo (.zip) encontrado em {self.resumed_session_dir}")

    def pause_training(self):
        """Pausa ou retoma o treinamento"""
        if not self.pause_values:
            self.logger.warning("Nenhum processo de treinamento ativo.")
            return

        try:
            if self.pause_values[-1].value:
                self.logger.info("Retomando treinamento.")
                self.pause_values[-1].value = 0
                self.pause_btn.config(text="Pausar")
            else:
                self.logger.info("Pausando treinamento.")
                self.pause_values[-1].value = 1
                self.pause_btn.config(text="Retomar")

            # Salvar modelo durante pausa/retomada
            self._save_model_during_training()

        except Exception as e:
            self.logger.error(f"Erro ao pausar/retomar treinamento: {e}")

    def _save_model_during_training(self):
        """Salva o modelo durante o treinamento para permitir retomada"""
        if not self.current_training_session:
            return

        try:
            # Usar ensure_directory do utils
            models_dir = ensure_directory(os.path.join(self.training_data_dir, "current_session", "models"))
            model_path = os.path.join(models_dir, "model.zip")

            # Enviar comando para salvar o modelo via IPC
            self.ipc_queue.put({"type": "save_model", "model_path": model_path})
            self.logger.info(f"Modelo salvo durante pausa/retomada: {model_path}")

        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo durante treinamento: {e}")

    def stop_training(self):
        """Finaliza o treinamento"""
        if self.exit_values:
            self.exit_values[-1].value = 1

        # Atualizar estado dos botões
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.real_time_check.config(state=tk.DISABLED)
        self.save_training_btn.config(state=tk.DISABLED)
        self.export_plots_btn.config(state=tk.DISABLED)

        self.logger.info("Treinamento finalizado pelo usuário")

    def save_training_data(self):
        """Salva todos os dados do treinamento atual incluindo o modelo"""
        if not self.current_training_session:
            messagebox.showwarning("Aviso", "Nenhum treinamento em andamento para salvar.")
            return

        try:
            # Usar ensure_directory do utils
            ensure_directory(self.training_data_dir)

            # Criar pasta específica para esta sessão
            timestamp = self.current_training_session["start_time"].strftime("%Y%m%d_%H%M%S")
            session_name = f"{self.current_env}_{self.current_robot}_{self.current_algorithm}_{timestamp}"
            session_dir = ensure_directory(os.path.join(self.training_data_dir, session_name))

            # Determinar episódio atual
            current_episode = max(self.episode_data["episodes"]) if self.episode_data["episodes"] else 0
            total_episodes = len(self.episode_data["episodes"])

            # Salvar dados do treinamento
            training_data = {
                "session_info": {
                    "environment": self.current_env,
                    "robot": self.current_robot,
                    "algorithm": self.current_algorithm,
                    "start_time": self.current_training_session["start_time"].isoformat(),
                    "total_steps": self.total_steps,
                    "current_episode": current_episode,
                    "total_episodes": total_episodes,
                    "device": self.device,
                },
                "episode_data": self.episode_data,
                "hyperparams": self.hyperparams,
            }

            with open(os.path.join(session_dir, "training_data.json"), "w") as f:
                json.dump(training_data, f, indent=2)

            # Salvar modelo usando sistema de controle
            model_saved = self._save_model_with_control(session_dir, session_name)

            # Salvar logs e gráficos
            self._save_additional_data(session_dir)

            # Mensagem final
            if model_saved:
                messagebox.showinfo("Sucesso", f"Treinamento salvo com sucesso!\n" f"Diretório: {session_dir}\n" f"Pronto para retomada!")
            else:
                messagebox.showwarning("Aviso", f"Dados salvos, mas modelo pode não estar atualizado.\n" f"Tente pausar o treinamento antes de salvar.\n" f"Diretório: {session_dir}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar treinamento: {e}")
            self.logger.error(f"Erro ao salvar treinamento: {e}")

    def _save_model_with_control(self, session_dir, session_name):
        """Salva modelo usando sistema de controle de arquivos"""
        try:
            models_dir = ensure_directory(os.path.join(session_dir, "models"))
            model_path = os.path.join(models_dir, "model.zip")

            self.logger.info(f"INICIANDO SALVAMENTO DO MODELO: {model_path}")

            # Sistema de controle
            control_dir = ensure_directory("training_control")
            control_file = f"save_model_{int(time.time() * 1000)}.json"
            control_path = os.path.join(control_dir, control_file)

            control_data = {"model_path": model_path, "timestamp": time.time(), "session": session_name}

            with open(control_path, "w") as f:
                json.dump(control_data, f, indent=2)

            self.logger.info(f"Arquivo de controle criado: {control_path}")

            # Aguardar salvamento
            max_wait = 21
            check_interval = 1

            for wait_time in range(max_wait):
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    self.logger.info(f"MODELO SALVO: {model_path} ({file_size} bytes)")

                    # Limpar arquivo de controle
                    try:
                        if os.path.exists(control_path):
                            os.remove(control_path)
                            self.logger.info("Arquivo de controle removido")
                    except:
                        pass

                    return True

                time.sleep(check_interval)

            return False

        except Exception as e:
            self.logger.error(f"Erro no sistema de controle de salvamento: {e}")
            return False

    def _save_additional_data(self, session_dir):
        """Salva logs e gráficos adicionais"""
        try:
            # Salvar logs
            logs_dir = ensure_directory(os.path.join(session_dir, "logs"))
            log_files = [f for f in os.listdir("logs") if f.endswith(".txt")]
            for log_file in log_files:
                shutil.copy2(os.path.join("logs", log_file), logs_dir)

            # Salvar gráficos
            plots_dir = ensure_directory(os.path.join(session_dir, "plots"))
            self.save_plots_to_directory(plots_dir)

        except Exception as e:
            self.logger.error(f"Erro ao salvar dados adicionais: {e}")

    def load_training_data(self):
        """Carrega dados de treinamento salvos e prepara para retomada"""
        try:
            session_dir = filedialog.askdirectory(title="Selecione a pasta do treinamento", initialdir=self.training_data_dir)

            if not session_dir:
                return

            # Carregar dados do treinamento
            self.resumed_session_dir = session_dir
            training_data = self._load_training_data_file(session_dir)

            # Encontrar modelo
            model_path = self._find_model_for_resume()
            if not model_path:
                return

            # Restaurar dados do treinamento
            self._restore_training_data(training_data, session_dir)

            # Configurar para retomada
            self.is_resuming = True
            self.resumed_session_dir = session_dir

            # Atualizar interface
            self.start_btn.config(text="Retomar Treinamento", state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)

            # Atualizar gráficos
            self.new_plot_data = True
            self._refresh_plots()

            # Habilitar botões
            self.save_training_btn.config(state=tk.NORMAL)
            self.export_plots_btn.config(state=tk.NORMAL)

            messagebox.showinfo(
                "Sucesso", f"Treinamento carregado!\n" f"Modelo: {os.path.basename(model_path)}\n" f"Próximo episódio: {self.current_episode}\n" f"Clique em 'Retomar Treinamento' para continuar."
            )

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar treinamento: {e}")
            self.logger.error(f"Erro ao carregar treinamento: {e}")

    def _load_training_data_file(self, session_dir):
        """Carrega arquivo de dados de treinamento"""
        data_file = os.path.join(session_dir, "training_data.json")
        if not os.path.exists(data_file):
            raise FileNotFoundError("Arquivo de dados não encontrado.")

        with open(data_file, "r") as f:
            return json.load(f)

    def _restore_training_data(self, training_data, session_dir):
        """Restaura dados do treinamento carregado"""
        session_info = training_data["session_info"]
        self.episode_data = training_data["episode_data"]
        self.hyperparams = training_data.get("hyperparams", {})

        saved_current_episode = session_info.get("current_episode", 0)
        total_episodes = session_info.get("total_episodes", 0)

        if self.episode_data["episodes"]:
            last_episode_in_data = max(self.episode_data["episodes"])
            self.current_episode = max(saved_current_episode, last_episode_in_data) + 1
        else:
            self.current_episode = saved_current_episode + 1

        self.loaded_episode_count = total_episodes

        # Atualizar configurações
        self.current_env = session_info["environment"]
        self.current_robot = session_info["robot"]
        self.current_algorithm = session_info["algorithm"]

        self.current_training_session = {
            "start_time": datetime.fromisoformat(session_info["start_time"]),
            "environment": self.current_env,
            "robot": self.current_robot,
            "algorithm": self.current_algorithm,
            "episode_data": self.episode_data.copy(),
            "hyperparams": self.hyperparams,
        }

    def export_plots(self):
        """Exporta gráficos como imagens para uso na tese"""
        try:
            export_dir = filedialog.askdirectory(title="Selecione onde salvar os gráficos", initialdir=os.path.expanduser("~"))

            if not export_dir:
                return

            # Opções de dimensões
            dimension_window = tk.Toplevel(self.root)
            dimension_window.title("Configurações de Exportação")
            dimension_window.geometry("300x200")
            dimension_window.transient(self.root)
            dimension_window.grab_set()

            ttk.Label(dimension_window, text="Largura (polegadas):").pack(pady=5)
            width_var = tk.StringVar(value="10")
            width_entry = ttk.Entry(dimension_window, textvariable=width_var)
            width_entry.pack(pady=5)

            ttk.Label(dimension_window, text="Altura (polegadas):").pack(pady=5)
            height_var = tk.StringVar(value="8")
            height_entry = ttk.Entry(dimension_window, textvariable=height_var)
            height_entry.pack(pady=5)

            dpi_var = tk.StringVar(value="300")
            ttk.Label(dimension_window, text="DPI:").pack(pady=5)
            dpi_entry = ttk.Entry(dimension_window, textvariable=dpi_var)
            dpi_entry.pack(pady=5)

            button_frame = ttk.Frame(dimension_window)
            button_frame.pack(pady=10)

            def do_export():
                try:
                    width = float(width_var.get())
                    height = float(height_var.get())
                    dpi = int(dpi_var.get())
                    if width <= 0 or height <= 0 or dpi <= 0:
                        messagebox.showerror("Erro", "Valores devem ser maiores que zero.")
                        return

                    self.save_plots_to_directory(export_dir, width, height, dpi)
                    dimension_window.destroy()
                    messagebox.showinfo("Sucesso", f"Gráficos exportados para: {export_dir}")

                except ValueError:
                    messagebox.showerror("Erro", "Valores inválidos para dimensões ou DPI.")

            def cancel_export():
                dimension_window.destroy()

            ttk.Button(button_frame, text="Exportar", command=do_export).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancelar", command=cancel_export).pack(side=tk.LEFT, padx=5)

            width_entry.focus()
            dimension_window.bind("<Return>", lambda e: do_export())
            dimension_window.bind("<Escape>", lambda e: cancel_export())

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar gráficos: {e}")

    def save_plots_to_directory(self, directory, width=10, height=8, dpi=300):
        """Salva todos os gráficos em um diretório"""
        try:
            # Salvar gráfico combinado
            combined_fig, combined_axs = plt.subplots(5, 1, figsize=(width, height * 2.5), constrained_layout=True)
            self._plot_to_figure(combined_axs)
            combined_fig.savefig(os.path.join(directory, "all_plots.png"), dpi=dpi, bbox_inches="tight")
            plt.close(combined_fig)

            # Salvar gráficos individuais
            for i, (title, ylabel, color, data_key) in enumerate(zip(self.plot_titles, self.plot_ylabels, self.plot_colors, self.plot_data_keys)):
                fig, ax = plt.subplots(figsize=(width, height))

                if i == 3:  # Gráfico de posição IMU
                    ax.plot(self.episode_data["episodes"], self.episode_data["imu_x"], label="X", color="red")
                    ax.plot(self.episode_data["episodes"], self.episode_data["imu_y"], label="Y", color="green")
                    ax.plot(self.episode_data["episodes"], self.episode_data["imu_z"], label="Z", color="blue")
                elif i == 4:  # Gráfico de orientação
                    ax.plot(self.episode_data["episodes"], self.episode_data["roll"], label="Roll", color="red")
                    ax.plot(self.episode_data["episodes"], self.episode_data["pitch"], label="Pitch", color="green")
                    ax.plot(self.episode_data["episodes"], self.episode_data["yaw"], label="Yaw", color="blue")
                else:
                    ax.plot(self.episode_data["episodes"], self.episode_data[data_key], label=ylabel, color=color)

                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Episódio")
                ax.legend()
                ax.grid(True, alpha=0.3)

                filename = f"plot_{data_key}.png"
                fig.savefig(os.path.join(directory, filename), dpi=dpi, bbox_inches="tight")
                plt.close(fig)

        except Exception as e:
            self.logger.error(f"Erro ao salvar gráficos: {e}")

    def _plot_to_figure(self, axs):
        """Plota dados nos eixos fornecidos"""
        for i, (title, ylabel, color, data_key) in enumerate(zip(self.plot_titles, self.plot_ylabels, self.plot_colors, self.plot_data_keys)):
            axs[i].clear()
            if i == 3:  # Gráfico de posição IMU
                axs[i].plot(self.episode_data["episodes"], self.episode_data["imu_x"], label="X", color="red")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["imu_y"], label="Y", color="green")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["imu_z"], label="Z", color="blue")
            elif i == 4:  # Gráfico de orientação
                axs[i].plot(self.episode_data["episodes"], self.episode_data["roll"], label="Roll", color="red")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["pitch"], label="Pitch", color="green")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["yaw"], label="Yaw", color="blue")
            else:
                axs[i].plot(self.episode_data["episodes"], self.episode_data[data_key], label=ylabel, color=color)

            axs[i].set_title(title)
            axs[i].set_ylabel(ylabel)
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)

        axs[-1].set_xlabel("Episódio")

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

    def toggle_real_time(self):
        """Alterna o modo tempo real da simulação"""
        if not self.enable_real_time_values:
            self.logger.warning("toggle_real_time: Nenhum processo de treinamento ativo.")
            return

        new_value = self.real_time_var.get()
        self.enable_real_time_values[-1].value = new_value

        if new_value:
            self.logger.info("Modo tempo real ativado")
        else:
            self.logger.info("Modo tempo real desativado")

    def build_steps_label_text(self, total_steps, steps_per_second):
        return f"Total Steps: {total_steps} | Steps/s: {steps_per_second:.1f}"

    def _update_step_counter(self):
        """Atualiza o contador de steps periodicamente"""
        if self.gui_closed:
            return

        current_time = time.time()
        time_diff = current_time - self.training_start_time

        if self.total_steps is None:
            self.steps_per_second = 0

        else:
            self.steps_per_second = self.total_steps / time_diff

        self.steps_label.config(text=self.build_steps_label_text(self.total_steps, self.steps_per_second))

        after_id = self.root.after(500, self._update_step_counter)
        self.after_ids["_update_step_counter"] = after_id

    def _refresh_plots(self):
        """Atualiza os gráficos com os dados atuais"""
        if self.gui_closed:
            return

        try:
            if not self.episode_data["episodes"] or not self.new_plot_data:
                after_id = self.root.after(500, self._refresh_plots)
                self.after_ids["_refresh_plots"] = after_id
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

        after_id = self.root.after(500, self._refresh_plots)
        self.after_ids["_refresh_plots"] = after_id

    def _update_log_display(self):
        """Atualiza a exibição de logs"""
        if self.gui_closed:
            return

        if self.gui_log_queue.empty():
            after_id = self.root.after(500, self._update_log_display)
            self.after_ids["_update_log_display"] = after_id
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

        after_id = self.root.after(500, self._update_log_display)
        self.after_ids["_update_log_display"] = after_id

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
                        if self.total_steps is None:
                            self.total_steps = msg.get("steps", 0)

                        else:
                            self.total_steps += msg.get("steps", 0)

                    elif data_type == "save_model":
                        # Comando de salvamento - apenas registrar, não processar aqui
                        model_path = msg.get("model_path", "desconhecido")
                        self.logger.info(f"Comando de salvamento recebido na GUI: {model_path}")

                    elif data_type == "model_saved":
                        # Confirmação de que o modelo foi salvo
                        model_path = msg.get("model_path", "desconhecido")
                        self.logger.info(f"Confirmação: Modelo salvo pelo processo: {model_path}")

                    elif data_type == "done":
                        self.logger.info("Processo de treinamento finalizado.")
                        self.start_btn.config(state=tk.NORMAL)
                        self.pause_btn.config(state=tk.DISABLED)
                        self.stop_btn.config(state=tk.DISABLED)
                        self.save_btn.config(state=tk.NORMAL)
                        self.real_time_check.config(state=tk.DISABLED)
                        self.save_training_btn.config(state=tk.DISABLED)
                        self.export_plots_btn.config(state=tk.DISABLED)
                        self.is_resuming = False

                    else:
                        self.logger.error(f"Tipo de dados desconhecido: {data_type} - Conteúdo: {msg}")
                except queue.Empty:
                    # Timeout normal, continuar loop
                    continue
                except EOFError:
                    self.logger.info("IPC queue fechada (EOFError)")
                    break
                except Exception as e:
                    self.logger.error(f"Erro ao receber mensagem IPC: {e}")
                    continue
        except Exception as e:
            self.logger.exception("Erro em ipc_runner")

            if not self.gui_closed:
                self.on_closing()

    def on_closing(self):
        """Limpeza adequada ao fechar"""
        self.logger.info("Gui fechando")

        # Cancelar todas as callbacks agendadas
        for after_id in self.after_ids.values():
            try:
                self.root.after_cancel(after_id)
            except:
                pass

        self.gui_closed = True

        if hasattr(self, "ipc_queue"):
            self.ipc_queue.put(None)  # Sinaliza para a thread IPC terminar

        for v in self.exit_values:
            v.value = 1  # Sinaliza para os processos terminarem

        self.logger.info("Aguardando thread IPC terminar...")
        if hasattr(self, "ipc_thread") and self.ipc_thread and self.ipc_thread.is_alive():
            self.ipc_thread.join(timeout=5.0)

        # Terminar processos
        self.logger.info("Aguardando processos de treinamento terminarem...")
        for p in self.processes:
            if p.is_alive():
                p.join(timeout=10.0)
                if p.is_alive():
                    self.logger.warning(f"Forcing termination of process {p.pid}")
                    p.terminate()

        self.logger.info("Todos os processos finalizados. Fechando GUI.")
        self.root.quit()

    def start(self):
        """Inicializa a aba de treinamento"""
        self.logger.info("Aba de treinamento inicializada")

        # Iniciar threads de atualização
        after_id = self.root.after(500, self._update_log_display)
        self.after_ids["_update_log_display"] = after_id

        after_id = self.root.after(500, self._refresh_plots)
        self.after_ids["_refresh_plots"] = after_id

        after_id = self.root.after(500, self._update_step_counter)
        self.after_ids["_update_step_counter"] = after_id

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()
