# tab_training.py
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
import math
import pygetwindow as gw
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_process
import utils
from utils import ENVIRONMENT_PATH, ROBOTS_PATH


class TrainingTab:
    def __init__(self, parent, device, logger, reward_system, notebook):
        self.frame = ttk.Frame(parent, padding="10")
        self.device = device
        self.logger = logger
        self.reward_system = reward_system
        self.notebook = notebook

        # Dados de treinamento
        self.current_env = ""
        self.current_robot = ""
        self.current_algorithm = ""
        self.tracker_status = None
        self.episode_data = {
            "episodes": [],
            "rewards": [],
            "times": [],
            "distances": [],
            "imu_x": [],
            "imu_y": [],
            "imu_z": [],
            "roll_deg": [],
            "pitch_deg": [],
            "yaw_deg": [],
            "imu_x_vel": [],
            "imu_y_vel": [],
            "imu_z_vel": [],
            "roll_vel_deg": [],
            "pitch_vel_deg": [],
            "yaw_vel_deg": [],
        }

        self.keys_to_filter = [
            "rewards",
            "times",
            "distances",
            "imu_x",
            "imu_y",
            "imu_z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "imu_x_vel",
            "imu_y_vel",
            "imu_z_vel",
            "roll_vel_deg",
            "pitch_vel_deg",
            "yaw_vel_deg",
        ]

        for key in self.keys_to_filter:
            self.episode_data[f"filtered_{key}"] = []

        # Controle de processos
        self.processes = []
        self.pause_values = []
        self.exit_values = []
        self.enable_real_time_values = []
        self.enable_visualization_values = []
        self.camera_selection_values = []
        self.config_changed_values = []
        self.gui_log_queue = queue.Queue()
        self.ipc_queue = multiprocessing.Queue()
        self.ipc_queue_main_to_process = multiprocessing.Queue()
        self.ipc_thread = None
        self.plot_data_lock = threading.Lock()
        self.gui_closed = False
        self.after_ids = {}
        self.new_plot_data = False
        self.last_pause_value = 0

        # Configurações de treinamento
        self.total_steps = 0
        self.steps_per_second = 0
        self.training_start_time = None
        self.training_time = 0
        self.pause_time = None
        self.current_episode = 0
        self.loaded_episode_count = 0
        self.is_resuming = False
        self.resumed_session_dir = None

        # Configurações de plot
        self.plot_titles = ["Recompensa por Episódio", "Duração do Episódio", "Distância Percorrida (X)", "Posição IMU (Y, Z)", "Orientação (Roll, Pitch, Yaw)"]
        self.plot_ylabels = ["Recompensa", "Tempo (s)", "Distância (m)", "Posição (m)", "Ângulo (°)"]
        self.plot_colors = ["blue", "orange", "red", "green", "purple", "brown"]
        self.plot_data_keys = ["rewards", "times", "distances", "imu_xyz", "rpy"]
        self.nf_alpha = 0.5
        self.nf_linewidth = 0.5

        # Configurar IPC logging
        utils.add_queue_handler_to_logger(self.logger, self.ipc_queue)

        self.settings = utils.load_default_settings()
        self.setup_ui()
        self.setup_ipc()

    def setup_ui(self):
        """Configura a interface da aba de treinamento"""
        # Frame principal
        main_frame = ttk.Frame(self.frame)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controle de Treinamento", padding="1")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=1)

        # Linha1: Seleções Principais
        row1_frame = ttk.Frame(control_frame)
        row1_frame.grid(row=0, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=0)

        # Seleção de algoritmo
        ttk.Label(row1_frame, text="Algoritmo:").grid(row=0, column=0, sticky=tk.W, padx=1)
        algorithms = ["TD3", "FastTD3", "PPO"]
        self.algorithm_var = tk.StringVar(value=algorithms[0])
        algorithm_combo = ttk.Combobox(row1_frame, textvariable=self.algorithm_var, values=algorithms, width=10)
        algorithm_combo.grid(row=0, column=1, padx=1)

        # Seleção de ambiente
        xacro_env_files = self._get_xacro_files(ENVIRONMENT_PATH)
        if not xacro_env_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {ENVIRONMENT_PATH}.")
            return

        ttk.Label(row1_frame, text="Ambiente:").grid(row=0, column=2, sticky=tk.W, padx=1)
        self.env_var = tk.StringVar(value=xacro_env_files[0])
        env_combo = ttk.Combobox(row1_frame, textvariable=self.env_var, values=xacro_env_files, width=10)
        env_combo.grid(row=0, column=3, padx=5)

        # Seleção de robô
        xacro_robot_files = self._get_xacro_files(ROBOTS_PATH)
        if not xacro_robot_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {ROBOTS_PATH}.")
            return

        ttk.Label(row1_frame, text="Robô:").grid(row=0, column=4, sticky=tk.W, padx=1)
        self.robot_var = tk.StringVar(value=self.settings.get("default_robot", xacro_robot_files[-1]))
        robot_combo = ttk.Combobox(row1_frame, textvariable=self.robot_var, values=xacro_robot_files, width=12)
        robot_combo.grid(row=0, column=5, padx=5)

        # Botões de controle
        self.start_btn = ttk.Button(row1_frame, text="Iniciar Treino", command=self.start_training, width=15)
        self.start_btn.grid(row=0, column=6, padx=1)

        self.pause_btn = ttk.Button(row1_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED, width=10)
        self.pause_btn.grid(row=0, column=7, padx=1)

        self.stop_btn = ttk.Button(row1_frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED, width=10)
        self.stop_btn.grid(row=0, column=8, padx=1)

        # Linha 2: Botões secundários e checkboxes
        row2_frame = ttk.Frame(control_frame)
        row2_frame.grid(row=1, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=1)

        self.save_training_btn = ttk.Button(row2_frame, text="Salvar Treino", command=self.save_training_callback_btn, state=tk.DISABLED, width=15)
        self.save_training_btn.grid(row=0, column=0, padx=1)

        self.load_training_btn = ttk.Button(row2_frame, text="Carregar Treino", command=self.load_training_data, width=15)
        self.load_training_btn.grid(row=0, column=1, padx=1)

        self.export_plots_btn = ttk.Button(row2_frame, text="Exportar Gráficos", command=self.export_plots, width=15)
        self.export_plots_btn.grid(row=0, column=2, padx=1)

        self.enable_dpg_var = tk.BooleanVar(value=self.settings.get("enable_dynamic_policy_gradient", True))
        self.enable_dpg_check = ttk.Checkbutton(row2_frame, text="Dynamic Policy Gradient", variable=self.enable_dpg_var, width=22)
        self.enable_dpg_check.grid(row=0, column=3, padx=1)

        self.enable_visualization_var = tk.BooleanVar(value=self.settings.get("enable_visualize_robot", False))
        self.enable_visualization_check = ttk.Checkbutton(row2_frame, text="Visualizar Robô", variable=self.enable_visualization_var, command=self.toggle_visualization, width=15)
        self.enable_visualization_check.grid(row=0, column=4, padx=1)

        self.real_time_var = tk.BooleanVar(value=self.settings.get("enable_real_time", True))
        self.real_time_check = ttk.Checkbutton(row2_frame, text="Tempo Real", variable=self.real_time_var, command=self.toggle_real_time, width=15)
        self.real_time_check.grid(row=0, column=5, padx=5)

        ttk.Label(row2_frame, text="Câmera:").grid(row=0, column=6, sticky=tk.W, padx=5)

        camera_options = {
            1: "Ambiente geral",
            2: "Robô - Diagonal direita",
            3: "Robô - Diagonal esquerda",
            4: "Robô - Lateral direita",
            5: "Robô - Lateral esquerda",
            6: "Robô - Frontal",
            7: "Robô - Traseira",
        }
        self.camera_selection_int = self.settings.get("camera_index", 1)
        self.camera_selection_var = tk.StringVar(value=camera_options[self.camera_selection_int])
        self.camera_selection_combobox = ttk.Combobox(row2_frame, textvariable=self.camera_selection_var, values=list(camera_options.values()), state="readonly", width=25)
        self.camera_selection_combobox.grid(row=0, column=7, padx=5)
        self.camera_selection_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_camera_selection(event, camera_options))

        # Gráficos
        graph_frame = ttk.LabelFrame(main_frame, text="Desempenho em Tempo Real", padding="1")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=1)

        self.graph_notebook = ttk.Notebook(graph_frame)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)

        # Aba 1: Gráficos principais (Recompensa, Tempo, Distância)
        self.tab_main = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.tab_main, text="Principais")
        self.fig_main, self.axs_main = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.tab_main)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Aba 2: Posições (X, Y, Z)
        self.tab_positions = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.tab_positions, text="Posições")
        self.fig_pos, self.axs_pos = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
        self.canvas_pos = FigureCanvasTkAgg(self.fig_pos, master=self.tab_positions)
        self.canvas_pos.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Aba 3: Orientação (Roll, Pitch, Yaw)
        self.tab_orientation = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.tab_orientation, text="Orientação")
        self.fig_ori, self.axs_ori = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
        self.canvas_ori = FigureCanvasTkAgg(self.fig_ori, master=self.tab_orientation)
        self.canvas_ori.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Aba 4: Velocidade (vx, vy, vz)
        self.tab_velocity = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.tab_velocity, text="Velocidade")
        self.fig_vel, self.axs_vel = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
        self.canvas_vel = FigureCanvasTkAgg(self.fig_vel, master=self.tab_velocity)
        self.canvas_vel.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Aba 5: Velocidade Angular (wx, wy, wz)
        self.tab_ang_velocity = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.tab_ang_velocity, text="Velocidade Angular")
        self.fig_angvel, self.axs_angvel = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
        self.canvas_angvel = FigureCanvasTkAgg(self.fig_angvel, master=self.tab_ang_velocity)
        self.canvas_angvel.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_plots()

        # Logs
        log_frame = ttk.LabelFrame(main_frame, text="Log de Treinamento", padding="1")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=1)

        status_frame = ttk.Frame(log_frame)
        status_frame.grid(row=0, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=1)

        self.steps_label = ttk.Label(status_frame, text=self.build_steps_label_text(0, 0, 0))
        self.steps_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.tracker_status_label = ttk.Label(status_frame, text="Melhor recompensa: N/A | Steps sem melhoria: 0")
        self.tracker_status_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Configurar o grid dentro do log_frame para que o texto expanda
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        self.log_text = tk.Text(log_frame, height=12, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))

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
            self.logger.exception("Erro ao listar arquivos .xacro")
            return []

    def setup_ipc(self):
        """Configura IPC para comunicação entre processos"""
        self.ipc_thread = threading.Thread(target=self.ipc_runner, daemon=True)
        self.ipc_thread.start()

    def _initialize_plots(self):
        """Inicializa os gráficos com títulos e configurações"""
        try:
            # Principais: Recompensa, Tempo, Distância
            titles_main = self.plot_titles[:3]
            ylabels_main = self.plot_ylabels[:3]
            colors_main = self.plot_colors[:3]
            data_keys_main = self.plot_data_keys[:3]

            for i, (title, ylabel, color, data_key) in enumerate(zip(titles_main, ylabels_main, colors_main, data_keys_main)):
                self.axs_main[i].clear()
                self.axs_main[i].plot([], [], label=ylabel, color=color)
                self.axs_main[i].set_title(title)
                self.axs_main[i].set_ylabel(ylabel)
                self.axs_main[i].grid(True, alpha=0.3)
                self.axs_main[i].legend(loc="upper left")
                self.axs_main[i].set_xlim(1, 2)

            self.axs_main[-1].set_xlabel("Episódio")
            self.canvas_main.draw_idle()

            # Posições IMU (X, Y, Z)
            for i, (label, color, key) in enumerate(zip(["X", "Y", "Z"], ["red", "green", "blue"], ["imu_x", "imu_y", "imu_z"])):
                self.axs_pos[i].clear()
                self.axs_pos[i].plot(self.episode_data["episodes"], self.episode_data[key], color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                self.axs_pos[i].plot(self.episode_data["episodes"], self.episode_data[f"filtered_{key}"], label=label, color=color, linestyle="-")
                self.axs_pos[i].set_title(f"Posição {label}")
                self.axs_pos[i].set_ylabel(f"{label} (m)")
                self.axs_pos[i].grid(True, alpha=0.3)
                self.axs_pos[i].legend(loc="upper left")
                self.axs_pos[i].set_xlim(1, None)
            self.axs_pos[-1].set_xlabel("Episódio")
            self.canvas_pos.draw_idle()

            # Orientação (Roll, Pitch, Yaw)
            for i, (label, color, key) in enumerate(zip(["Roll", "Pitch", "Yaw"], ["red", "green", "blue"], ["roll_deg", "pitch_deg", "yaw_deg"])):
                self.axs_ori[i].clear()
                self.axs_ori[i].plot(self.episode_data["episodes"], self.episode_data[key], color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                self.axs_ori[i].plot(self.episode_data["episodes"], self.episode_data[f"filtered_{key}"], label=label, color=color, linestyle="-")
                self.axs_ori[i].set_title(label)
                self.axs_ori[i].set_ylabel(f"{label} (°)")
                self.axs_ori[i].grid(True, alpha=0.3)
                self.axs_ori[i].legend(loc="upper left")
                self.axs_ori[i].set_xlim(1, None)
            self.axs_ori[-1].set_xlabel("Episódio")
            self.canvas_ori.draw_idle()

            # Velocidade (vx, vy, vz)
            for i, (label, color, key) in enumerate(zip(["Vx", "Vy", "Vz"], ["red", "green", "blue"], ["imu_x_vel", "imu_y_vel", "imu_z_vel"])):
                self.axs_vel[i].clear()
                self.axs_vel[i].plot(self.episode_data["episodes"], self.episode_data.get(key, []), color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                self.axs_vel[i].plot(self.episode_data["episodes"], self.episode_data.get(f"filtered_{key}", []), label=label, color=color, linestyle="-")
                self.axs_vel[i].set_title(f"Velocidade {label}")
                self.axs_vel[i].set_ylabel(f"{label} (m/s)")
                self.axs_vel[i].grid(True, alpha=0.3)
                self.axs_vel[i].legend(loc="upper left")
                self.axs_vel[i].set_xlim(1, None)
            self.axs_vel[-1].set_xlabel("Episódio")
            self.canvas_vel.draw_idle()

            # Velocidade Angular (wx, wy, wz)
            for i, (label, color, key) in enumerate(zip(["Wx", "Wy", "Wz"], ["red", "green", "blue"], ["roll_vel_deg", "pitch_vel_deg", "yaw_vel_deg"])):
                self.axs_angvel[i].clear()
                self.axs_angvel[i].plot(self.episode_data["episodes"], self.episode_data.get(key, []), color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                self.axs_angvel[i].plot(self.episode_data["episodes"], self.episode_data.get(f"filtered_{key}", []), label=label, color=color, linestyle="-")
                self.axs_angvel[i].set_title(f"Velocidade Angular {label}")
                self.axs_angvel[i].set_ylabel(f"{label} (°/s)")
                self.axs_angvel[i].grid(True, alpha=0.3)
                self.axs_angvel[i].legend(loc="upper left")
                self.axs_angvel[i].set_xlim(1, None)
            self.axs_angvel[-1].set_xlabel("Episódio")
            self.canvas_angvel.draw_idle()

        except Exception as e:
            self.logger.exception("Erro ao inicializar gráficos")

        self.root.after(500, self._refresh_plots)

    def start_training(self):
        """Inicia um novo treinamento do zero"""
        self.start_btn.config(state=tk.DISABLED, text="Iniciando...")
        self.disable_other_tabs()

        if self.is_resuming:
            self.logger.info(f"Retomando treinamento - current_episode: {self.current_episode}")
            # self._resume_training()
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
            self.tracker_status = None
            self.episode_data = {key: [] for key in self.episode_data.keys()}
            self.training_time = 0
            self.total_steps = 0
            self.steps_per_second = 0
            self._initialize_plots()

            shutil.rmtree(utils.TEMP_MODEL_SAVE_PATH)
            utils.ensure_directory(utils.TEMP_MODEL_SAVE_PATH)
            self.last_autosave_folder = None

            self.logger.info("Sistema de tracking preparado para novo treinamento")

            # Iniciar treinamento em processo separado
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

            # Iniciar processo com config
            p = multiprocessing.Process(
                target=train_process.process_runner,
                args=(
                    self.current_env,
                    self.current_robot,
                    self.current_algorithm,
                    self.ipc_queue,
                    self.ipc_queue_main_to_process,
                    self.reward_system,
                    pause_val,
                    exit_val,
                    enable_visualization_val,
                    realtime_val,
                    camera_selection_val,
                    config_changed_val,
                    self.device,
                    0,
                    None,
                    self.enable_dpg_var.get(),
                ),
            )
            p.start()
            self.training_start_time = time.time()
            self.pause_time = None
            self.processes.append(p)

            self.logger.info(f"Processo de treinamento iniciado: {self.current_env} + {self.current_robot} + {self.current_algorithm}")

            # Habilitar botões
            self.save_training_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.start_btn.config(text="Iniciar Treino")
            self.pause_btn.config(text="Pausar")

        except Exception as e:
            self.logger.exception("Erro ao iniciar treinamento")
            messagebox.showerror("Erro", f"Erro ao iniciar treinamento: {e}")
            self.start_btn.config(state=tk.NORMAL, text="Iniciar Treino")
            self.enable_other_tabs()

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

                if self.pause_time is not None:
                    paused_duration = time.time() - self.pause_time
                    self.training_start_time += paused_duration
                    self.pause_time = None

            else:
                self.logger.info("Pausando treinamento.")
                self.pause_values[-1].value = 1
                self.pause_btn.config(text="Retomar")
                self.pause_time = time.time()

            self.last_pause_value = self.pause_values[-1].value
            self.config_changed_values[-1].value = 1

        except Exception as e:
            self.logger.exception("Erro ao pausar/retomar treinamento")

    def disable_other_tabs(self):
        current = self.notebook.select()

        for tab_id in self.notebook.tabs():
            if tab_id != current:
                self.notebook.tab(tab_id, state="disabled")

    def enable_other_tabs(self):
        current = self.notebook.select()

        for tab_id in self.notebook.tabs():
            if tab_id != current:
                self.notebook.tab(tab_id, state="normal")

    def stop_training(self):
        """Finaliza o treinamento"""

        if self.exit_values:
            self.exit_values[-1].value = 1
            self.config_changed_values[-1].value = 1

        # Atualizar estado dos botões
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_training_btn.config(state=tk.DISABLED)
        self.enable_other_tabs()

        self.logger.info("Treinamento finalizado pelo usuário")

    def save_training_callback_btn(self):
        """Salva todos os dados do treinamento atual incluindo o modelo"""
        if self.training_start_time is None:
            messagebox.showwarning("Aviso", "Nenhum treinamento em andamento para salvar.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{self.current_env}__{self.current_robot}__{timestamp}"
        save_session_path = utils.ensure_directory(os.path.join(utils.TRAINING_DATA_PATH, session_name))

        self.ipc_queue_main_to_process.put({"type": "save_request", "save_session_path": save_session_path})
        self.config_changed_values[-1].value = 1

    def save_training_data(self, is_autosave, save_path, tracker_status):
        try:
            total_episodes = len(self.episode_data["episodes"])

            training_data = {
                "session_info": {
                    "is_autosave": is_autosave,
                    "environment": self.current_env,
                    "robot": self.current_robot,
                    "algorithm": self.current_algorithm,
                    "training_start_time": self.training_start_time,
                    "save_time": datetime.now().isoformat(),
                    "total_steps": self.total_steps,
                    "total_episodes": total_episodes,
                    "device": self.device,
                },
                "tracker_status": tracker_status,
                "episode_data": self.episode_data,
            }

            training_data_path = os.path.join(save_path, "training_data.json")

            with open(training_data_path, "w", encoding="utf-8") as f:
                json.dump(training_data, f, indent=4, ensure_ascii=False)

            self._save_additional_data(save_path)

            if is_autosave:
                self.last_autosave_folder = save_path

            else:
                if self.last_autosave_folder is None:
                    messagebox.showwarning("Aviso", f"Treinamento salvo com sucesso, porém não há histórico de salvamento automático por melhoria de recompensa\nDiretório: {save_path}")

                else:
                    autosave_copy_path = os.path.join(save_path, "last_autosave")
                    shutil.copytree(self.last_autosave_folder, autosave_copy_path, dirs_exist_ok=True)
                    messagebox.showinfo("Sucesso", f"Treinamento salvo com sucesso!\nDiretório: {save_path}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar treinamento: {e}")
            self.logger.exception("Erro ao salvar treinamento")

    def _save_additional_data(self, save_path):
        """Salva logs e gráficos adicionais"""
        try:
            # Salvar logs
            logs_dir = utils.ensure_directory(os.path.join(save_path, "logs"))
            log_files = [f for f in os.listdir("logs") if f.endswith(".txt")]
            for log_file in log_files:
                shutil.copy2(os.path.join("logs", log_file), logs_dir)

            # Salvar gráficos
            plots_dir = utils.ensure_directory(os.path.join(save_path, "plots"))
            self.save_plots_to_directory(plots_dir)

        except Exception as e:
            self.logger.exception("Erro ao salvar dados adicionais")

    def load_training_data(self):
        """Carrega dados de treinamento salvos e prepara para retomada"""
        try:
            session_dir = filedialog.askdirectory(title="Selecione a pasta do treinamento", initialdir=utils.TRAINING_DATA_PATH)

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
            self.start_btn.config(text="Retomar Treino", state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)

            # Atualizar gráficos
            self.new_plot_data = True
            self._refresh_plots()

            # Habilitar botões
            self.save_training_btn.config(state=tk.NORMAL)

            messagebox.showinfo(
                "Sucesso", f"Treinamento carregado!\n" f"Modelo: {os.path.basename(model_path)}\n" f"Próximo episódio: {self.current_episode}\n" f"Clique em 'Retomar Treino' para continuar."
            )

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar treinamento: {e}")
            self.logger.exception("Erro ao carregar treinamento")

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

        saved_current_episode = session_info.get("current_episode", 0)
        total_episodes = session_info.get("total_episodes", 0)

        if self.episode_data["episodes"]:
            self.current_episode = max(self.episode_data["episodes"])
            self.loaded_episode_count = len(self.episode_data["episodes"])
        else:
            self.current_episode = session_info.get("current_episode", 0)
            self.loaded_episode_count = 0

        # Atualizar configurações
        self.current_env = session_info["environment"]
        self.current_robot = session_info["robot"]
        self.current_algorithm = session_info["algorithm"]

        # Tentar carregar informações do tracker se existirem
        training_info_path = os.path.join(session_dir, "training_info.json")
        if os.path.exists(training_info_path):
            try:
                with open(training_info_path, "r") as f:
                    training_info = json.load(f)

                # Restaurar dados básicos
                self.tracker.best_reward = training_info.get("best_reward", -float("inf"))
                self.tracker.best_distance = training_info.get("best_distance", 0.0)
                self.total_steps = training_info.get("total_steps", 0)

                self.logger.info(f"Informações de treino carregadas: recompensa={self.tracker.best_reward}, steps={self.total_steps}")

            except Exception as e:
                self.logger.exception("Erro ao carregar informações de treino")

    def export_plots(self):
        """Exporta gráficos como imagens para uso na tese"""
        try:
            export_dir = filedialog.askdirectory(title="Selecione onde salvar os gráficos", initialdir=os.path.expanduser("~"))

            if not export_dir:
                return

            # Opções de dimensões
            dimension_window = tk.Toplevel(self.root)
            dimension_window.title("Configurações de Exportação")
            dimension_window.geometry("300x250")
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
            self.logger.exception("Erro ao exportar gráficos")
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
                    ax.plot(self.episode_data["episodes"], self.episode_data["filtered_imu_y"], label="Y", color="green")
                    ax.plot(self.episode_data["episodes"], self.episode_data["filtered_imu_z"], label="Z", color="blue")
                elif i == 4:  # Gráfico de orientação
                    ax.plot(self.episode_data["episodes"], self.episode_data["filtered_roll_deg"], label="Roll", color="red")
                    ax.plot(self.episode_data["episodes"], self.episode_data["filtered_pitch_deg"], label="Pitch", color="green")
                    ax.plot(self.episode_data["episodes"], self.episode_data["filtered_yaw_deg"], label="Yaw", color="blue")
                else:
                    ax.plot(self.episode_data["episodes"], self.episode_data[f"filtered_{data_key}"], label=ylabel, color=color)

                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Episódio")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper left")
                ax.set_xlim(1, None)

                filename = f"plot_{data_key}.png"
                fig.savefig(os.path.join(directory, filename), dpi=dpi, bbox_inches="tight")
                plt.close(fig)

        except Exception as e:
            self.logger.exception("Erro ao salvar gráficos")

    def _plot_to_figure(self, axs):
        """Plota dados nos eixos fornecidos"""
        for i, (title, ylabel, color, data_key) in enumerate(zip(self.plot_titles, self.plot_ylabels, self.plot_colors, self.plot_data_keys)):
            axs[i].clear()

            if i == 3:  # Gráfico de posição IMU
                axs[i].plot(self.episode_data["episodes"], self.episode_data["filtered_imu_y"], label="Y", color="green")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["filtered_imu_z"], label="Z", color="blue")
            elif i == 4:  # Gráfico de orientação
                axs[i].plot(self.episode_data["episodes"], self.episode_data["filtered_roll_deg"], label="Roll", color="red")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["filtered_pitch_deg"], label="Pitch", color="green")
                axs[i].plot(self.episode_data["episodes"], self.episode_data["filtered_yaw_deg"], label="Yaw", color="blue")
            else:
                axs[i].plot(self.episode_data["episodes"], self.episode_data[f"filtered_{data_key}"], label=ylabel, color=color)

            axs[i].set_title(title)
            axs[i].set_ylabel(ylabel)
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc="upper left")
            axs[i].set_xlim(1, None)

        axs[-1].set_xlabel("Episódio")

    def toggle_visualization(self):
        """Alterna entre visualizar ou não o robô durante o treinamento"""
        new_value = self.enable_visualization_var.get()

        if new_value:
            self.logger.info("Visualização do robô ativada")
        else:
            self.logger.info("Visualização do robô desativada")

        if self.enable_visualization_values:
            self.enable_visualization_values[-1].value = new_value
            self.config_changed_values[-1].value = 1

        else:
            self.logger.info("toggle_visualization: Nenhum processo de treinamento ativo.")

    def toggle_real_time(self):
        """Alterna o modo tempo real da simulação"""
        new_value = self.real_time_var.get()

        if new_value:
            self.logger.info("Modo tempo real ativado")
        else:
            self.logger.info("Modo tempo real desativado")

        if self.enable_real_time_values:
            self.enable_real_time_values[-1].value = new_value
            self.config_changed_values[-1].value = 1

        else:
            self.logger.info("toggle_real_time: Nenhum processo de treinamento ativo.")

    def update_camera_selection(self, event, camera_options):
        """Atualiza o valor da câmera selecionada no multiprocessing com base no nome exibido"""
        selected_name = self.camera_selection_combobox.get()
        self.logger.info(f"Câmera selecionada: {selected_name}")
        selected_value = next((key for key, value in camera_options.items() if value == selected_name), None)

        self.camera_selection_var.set(selected_name)
        self.camera_selection_int = selected_value

        if self.camera_selection_values:
            self.camera_selection_values[-1].value = selected_value
            self.config_changed_values[-1].value = 1

        else:
            self.logger.info("update_camera_selection: Nenhum processo de treinamento ativo.")

    def build_steps_label_text(self, training_time, total_steps, steps_per_second):
        time_struct = time.gmtime(training_time)
        formatted_time = time.strftime("%H:%M:%S", time_struct)

        if len(self.episode_data["filtered_distances"]) > 0:
            mean_distance = self.episode_data["filtered_distances"][-1]

        else:
            mean_distance = 0.0

        return f"Training time: {formatted_time} | Total Steps: {total_steps:,} | Steps/s: {steps_per_second:.1f} | Distância filtrada: {mean_distance:.2f}m"

    def _update_gui_info(self):
        """Atualiza o contador de steps e outras informações periodicamente"""
        if self.gui_closed:
            return

        self._update_tracker_status()

        if self.pause_time is None and self.training_start_time is not None and self.total_steps is not None and self.exit_values[-1].value == 0:
            current_time = time.time()
            self.training_time = current_time - self.training_start_time

            if self.total_steps is None or self.training_time <= 0:
                self.steps_per_second = 0

            else:
                self.steps_per_second = self.total_steps / self.training_time

            self.steps_label.config(text=self.build_steps_label_text(self.training_time, self.total_steps, self.steps_per_second))

        after_id = self.root.after(500, self._update_gui_info)
        self.after_ids["_update_gui_info"] = after_id

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
                current_tab = self.graph_notebook.index(self.graph_notebook.select())
                if current_tab == 0:
                    titles_main = self.plot_titles[:3]
                    ylabels_main = self.plot_ylabels[:3]
                    colors_main = self.plot_colors[:3]
                    data_keys_main = self.plot_data_keys[:3]
                    for i, (title, ylabel, color, key) in enumerate(zip(titles_main, ylabels_main, colors_main, data_keys_main)):
                        self.axs_main[i].clear()
                        self.axs_main[i].plot(self.episode_data["episodes"], self.episode_data[key], color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                        self.axs_main[i].plot(self.episode_data["episodes"], self.episode_data[f"filtered_{key}"], label=ylabel, color=color, linestyle="-")
                        self.axs_main[i].set_ylabel(ylabel)
                        self.axs_main[i].grid(True, alpha=0.3)
                        self.axs_main[i].legend(loc="upper left")
                        self.axs_main[i].set_xlim(1, None)
                        if i == 2:
                            self.axs_main[i].set_ylim(-1.5, 10)
                    self.axs_main[-1].set_xlabel("Episódio")
                    self.canvas_main.draw()
                elif current_tab == 1:
                    for i, (label, color, key) in enumerate(zip(["X", "Y", "Z"], ["red", "green", "blue"], ["imu_x", "imu_y", "imu_z"])):
                        self.axs_pos[i].clear()
                        self.axs_pos[i].plot(self.episode_data["episodes"], self.episode_data[key], color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                        self.axs_pos[i].plot(self.episode_data["episodes"], self.episode_data[f"filtered_{key}"], label=label, color=color, linestyle="-")
                        self.axs_pos[i].set_ylabel(f"{label} (m)")
                        self.axs_pos[i].grid(True, alpha=0.3)
                        self.axs_pos[i].legend(loc="upper left")
                        self.axs_pos[i].set_xlim(1, None)
                    self.axs_pos[-1].set_xlabel("Episódio")
                    self.canvas_pos.draw()
                elif current_tab == 2:
                    for i, (label, color, key) in enumerate(zip(["Roll", "Pitch", "Yaw"], ["red", "green", "blue"], ["roll_deg", "pitch_deg", "yaw_deg"])):
                        self.axs_ori[i].clear()
                        self.axs_ori[i].plot(self.episode_data["episodes"], self.episode_data[key], color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                        self.axs_ori[i].plot(self.episode_data["episodes"], self.episode_data[f"filtered_{key}"], label=label, color=color, linestyle="-")
                        self.axs_ori[i].set_ylabel(f"{label} (°)")
                        self.axs_ori[i].grid(True, alpha=0.3)
                        self.axs_ori[i].legend(loc="upper left")
                        self.axs_ori[i].set_xlim(1, None)
                    self.axs_ori[-1].set_xlabel("Episódio")
                    self.canvas_ori.draw()
                elif current_tab == 3:
                    for i, (label, color, key) in enumerate(zip(["Vx", "Vy", "Vz"], ["red", "green", "blue"], ["imu_x_vel", "imu_y_vel", "imu_z_vel"])):
                        self.axs_vel[i].clear()
                        self.axs_vel[i].plot(self.episode_data["episodes"], self.episode_data.get(key, []), color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                        self.axs_vel[i].plot(self.episode_data["episodes"], self.episode_data.get(f"filtered_{key}", []), label=label, color=color, linestyle="-")
                        self.axs_vel[i].set_ylabel(f"{label} (m/s)")
                        self.axs_vel[i].grid(True, alpha=0.3)
                        self.axs_vel[i].legend(loc="upper left")
                        self.axs_vel[i].set_xlim(1, None)
                    self.axs_vel[-1].set_xlabel("Episódio")
                    self.canvas_vel.draw()
                elif current_tab == 4:
                    for i, (label, color, key) in enumerate(zip(["Wx", "Wy", "Wz"], ["red", "green", "blue"], ["roll_vel_deg", "pitch_vel_deg", "yaw_vel_deg"])):
                        self.axs_angvel[i].clear()
                        self.axs_angvel[i].plot(self.episode_data["episodes"], self.episode_data.get(key, []), color=color, linestyle="-", alpha=self.nf_alpha, linewidth=self.nf_linewidth)
                        self.axs_angvel[i].plot(self.episode_data["episodes"], self.episode_data.get(f"filtered_{key}", []), label=label, color=color, linestyle="-")
                        self.axs_angvel[i].set_ylabel(f"{label} (°/s)")
                        self.axs_angvel[i].grid(True, alpha=0.3)
                        self.axs_angvel[i].legend(loc="upper left")
                        self.axs_angvel[i].set_xlim(1, None)
                    self.axs_angvel[-1].set_xlabel("Episódio")
                    self.canvas_angvel.draw()

        except Exception as e:
            self.logger.exception("Plot error")

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

    def _trigger_auto_pause(self):
        """Ativa pausa automática por plateau"""
        try:
            # Pausar o treinamento
            if self.pause_values and not self.pause_values[-1].value:
                self.pause_values[-1].value = 1
                self.config_changed_values[-1].value = 1
                self.last_pause_value = self.pause_values[-1].value
                self.pause_btn.config(text="Retomar")

                # Mostrar mensagem informativa
                self.root.after(100, lambda: self._show_auto_pause_message())

        except Exception as e:
            self.logger.exception("Erro ao ativar pausa automática")

    def _show_auto_pause_message(self, tracker_status):
        """Mostra mensagem de pausa automática"""
        messagebox.showinfo(
            "Treinamento Pausado Automaticamente",
            f"O treinamento foi pausado automaticamente após {tracker_status["patience_steps"]:,} steps sem melhoria significativa.\n\n"
            f"• Melhor recompensa: {tracker_status["best_reward"]:.2f}\n"
            f"• Melhor distância: {tracker_status["best_distance"]:.2f}m\n"
            f"• Steps totais: {self.total_steps:,}\n"
            f"• Steps sem melhoria: {tracker_status["steps_since_improvement"]:,}\n\n"
            "Clique em 'Retomar' para continuar o treinamento ou 'Salvar Treino' para finalizar.",
        )

    def _update_tracker_status(self):
        """Atualiza o label de status do tracker"""
        try:
            if self.tracker_status is None:
                return

            status = self.tracker_status

            # Criar texto de status com emojis para melhor visualização
            status_parts = []
            status_parts.append(f"Melhor: {status['best_reward']:.2f}")
            status_parts.append(f"Distância: {status['best_distance']:.2f}m")
            status_parts.append(f"Sem melhoria: {status['steps_since_improvement']:,}")

            auto_save_count = status.get("auto_save_count", 0)

            if auto_save_count > 0:
                status_parts.append(f"Salvamentos: {auto_save_count}")

            status_text = " | ".join(status_parts)
            self.tracker_status_label.config(text=status_text)

            # Mudar cor do texto se estiver perto de pausar
            if status["steps_since_improvement"] > status["patience_steps"] * 0.8:
                self.tracker_status_label.config(foreground="orange")
            elif status["steps_since_improvement"] > status["patience_steps"] * 0.9:
                self.tracker_status_label.config(foreground="red")
            else:
                self.tracker_status_label.config(foreground="black")

        except Exception as e:
            self.logger.exception("Erro ao atualizar status do tracker")
            self.tracker_status_label.config(text="Status: Erro no tracker")

    def _handle_episode_data(self, episode_data):
        try:
            # Atualizar DPG com fases da marcha
            if hasattr(self.reward_system, "dpg_manager") and self.reward_system.dpg_manager:
                self.reward_system.dpg_manager.update_phase_progression(episode_data)
            elif hasattr(self.reward_system, "gait_phase_dpg") and self.reward_system.gait_phase_dpg:
                self.reward_system.gait_phase_dpg.update_phase(episode_data)

        except Exception as e:
            self.logger.exception("Erro ao processar dados do episódio para tracker")

    def update_filtered_data(self):
        window_size = 20

        for key in self.keys_to_filter:
            filtered_key = "filtered_" + key
            data = self.episode_data[key]

            if len(data) < 5:
                self.episode_data[filtered_key].append(np.mean(data))

            else:
                self.episode_data[filtered_key].append(np.mean(data[-window_size:]))

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

                    data_type = msg.pop("type")

                    if data_type == "episode_data":
                        self.total_steps = msg.pop("total_steps")

                        with self.plot_data_lock:
                            for key, value in msg.items():
                                if key not in self.episode_data:
                                    self.episode_data[key] = []

                                self.episode_data[key].append(value)

                            self.episode_data["roll_deg"].append(math.degrees(msg.get("roll", 0)))
                            self.episode_data["pitch_deg"].append(math.degrees(msg.get("pitch", 0)))
                            self.episode_data["yaw_deg"].append(math.degrees(msg.get("yaw", 0)))
                            self.episode_data["roll_vel_deg"].append(math.degrees(msg.get("roll_vel", 0)))
                            self.episode_data["pitch_vel_deg"].append(math.degrees(msg.get("pitch_vel", 0)))
                            self.episode_data["yaw_vel_deg"].append(math.degrees(msg.get("yaw_vel", 0)))
                            self.update_filtered_data()

                        self.new_plot_data = True
                        self._handle_episode_data(msg)

                    elif data_type == "tracker_status":
                        self.tracker_status = msg.get("tracker_status")

                    elif data_type == "training_progress":
                        # Atualizar contador de steps
                        self.total_steps = msg.get("steps_completed", 0)
                        self.logger.debug(f"Progresso: {self.total_steps} steps")

                    elif data_type == "pybullet_window_ready":
                        self.logger.info("Janela do PyBullet pronta para visualização.")
                        self.focus_pybullet_window()

                    elif data_type == "agent_model_saved":
                        # O modelo do agente foi salvo
                        self.save_training_data(msg["autosave"], msg["save_path"], msg["tracker_status"])  # Salvar dados da gui na mesma pasta do modelo do agente
                        self.pause_values[-1].value = self.last_pause_value  # Restaurar pause, pois o treinamento é pausado ao salvar o modelo do agente
                        self.config_changed_values[-1].value = 1  # Ativa verificação de pause value pelo processo de treinamento

                    elif data_type == "autopause_request":
                        self.pause_training(force_pause=True)
                        self._show_auto_pause_message(msg["tracker_status"])

                    elif data_type == "done":
                        self.logger.info("Processo de treinamento finalizado.")
                        self.start_btn.config(state=tk.NORMAL)
                        self.pause_btn.config(state=tk.DISABLED)
                        self.stop_btn.config(state=tk.DISABLED)
                        self.save_training_btn.config(state=tk.DISABLED)
                        self.enable_other_tabs()
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
                    self.logger.exception("Erro ao receber mensagem IPC")
                    continue
        except Exception as e:
            self.logger.exception("Erro em ipc_runner")

            if not self.gui_closed:
                self.on_closing()

    def focus_pybullet_window(self):
        """Traz a janela do PyBullet para o foco"""
        try:
            windows = gw.getWindowsWithTitle("Bullet Physics")

            if windows:
                pybullet_window = windows[0]
                pybullet_window.activate()
                self.logger.info("Janela do PyBullet focada")

            else:
                self.logger.warning("Janela do PyBullet não encontrada para foco")

        except Exception as e:
            self.logger.exception("Erro ao focar a janela do PyBullet")

    def on_closing(self):
        """Limpeza adequada ao fechar"""
        self.logger.info("Gui fechando")

        # Marcar como fechado ANTES de cancelar os callbacks
        self.gui_closed = True

        # Cancelar todas as callbacks agendadas
        for after_id in self.after_ids.values():
            try:
                self.root.after_cancel(after_id)
            except Exception as e:
                pass

        # Limpar o dicionário de after_ids
        self.after_ids.clear()

        if hasattr(self, "ipc_queue"):
            self.ipc_queue.put(None)  # Sinaliza para a thread IPC terminar

        for v in self.exit_values:
            v.value = 1  # Sinaliza para os processos terminarem

        for v in self.config_changed_values:
            v.value = 1  # Necessário para processos verificarem o exit

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
        self.logger.info("Programa finalizado com sucesso.")  # Log adicional
        self.root.quit()

    def start(self):
        """Inicializa a aba de treinamento"""
        self.logger.info("Aba de treinamento inicializada")

        # Iniciar threads de atualização
        after_id = self.root.after(500, self._update_log_display)
        self.after_ids["_update_log_display"] = after_id

        after_id = self.root.after(500, self._refresh_plots)
        self.after_ids["_refresh_plots"] = after_id

        after_id = self.root.after(500, self._update_gui_info)
        self.after_ids["_update_gui_info"] = after_id

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()
