# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import time
import utils
import train_process
import multiprocessing
import queue
import json
import shutil
from datetime import datetime


class TrainingGUI:
    def __init__(self, device="cpu"):
        self.root = tk.Tk()
        self.root.title("Cruzada Generalization - Training Dashboard")
        self.root.geometry("1400x1000") #tamanho da gui

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
        self.current_episode = 0
        self.loaded_episode_count = 0

        self.training_data_dir = "training_data"
        self.current_training_session = None
        self.is_resuming = False
        self.resumed_session_dir = None

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
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Linha1: Seleções Principais
        row1_frame = ttk.LabelFrame(control_frame)
        row1_frame.grid(row=0, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=0)

        # Seleção de algoritmo
        ttk.Label(row1_frame, text="Algoritmo:").grid(row=0, column=0, sticky=tk.W, padx=5)
        algorithms = ["FastTD3", "TD3", "PPO"]
        self.algorithm_var = tk.StringVar(value=algorithms[0])
        algorithm_combo = ttk.Combobox(row1_frame, textvariable=self.algorithm_var, values=algorithms)
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

        ttk.Label(row1_frame, text="Ambiente:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.env_var = tk.StringVar(value=xacro_env_files[0])
        env_combo = ttk.Combobox(row1_frame, textvariable=self.env_var, values=xacro_env_files)
        env_combo.grid(row=0, column=3, padx=5)

        # Seleção de robô
        xacro_robot_files = [file.replace(".xacro", "") for file in os.listdir(utils.ROBOTS_PATH) if file.endswith(".xacro")]

        if len(xacro_robot_files) == 0:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ROBOTS_PATH}.")
            self.root.destroy()
            return

        ttk.Label(row1_frame, text="Robô:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.robot_var = tk.StringVar(value=xacro_robot_files[0])
        robot_combo = ttk.Combobox(row1_frame, textvariable=self.robot_var, values=xacro_robot_files, width=12)
        robot_combo.grid(row=0, column=5, padx=5)

        # Botões de controle
        self.start_btn = ttk.Button(row1_frame, text="Iniciar Treinamento", command=self.start_training)
        self.start_btn.grid(row=0, column=6, padx=5)

        self.pause_btn = ttk.Button(row1_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=7, padx=5)

        self.stop_btn = ttk.Button(row1_frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=8, padx=5)

        # Linha 2: Botões secundários e checkboxes
        row2_frame = ttk.Frame(control_frame)
        row2_frame.grid(row=1, column=0, columnspan=12, sticky=(tk.W, tk.E), pady=5)
        
        self.save_training_btn = ttk.Button(row2_frame, text="Salvar Treinamento", command=self.save_training_data, state=tk.DISABLED)
        self.save_training_btn.grid(row=0, column=0, padx=5)
        
        self.load_training_btn = ttk.Button(row2_frame, text="Carregar Treinamento", command=self.load_training_data)
        self.load_training_btn.grid(row=0, column=1, padx=5)
        
        self.export_plots_btn = ttk.Button(row2_frame, text="Exportar Gráficos", command=self.export_plots, state=tk.DISABLED)
        self.export_plots_btn.grid(row=0, column=2, padx=5)

        self.save_btn = ttk.Button(row2_frame, text="Salvar Snapshot", command=self.save_snapshot, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=3, padx=5)
        
        self.real_time_var = tk.BooleanVar(value=False)
        self.real_time_check = ttk.Checkbutton(
            row2_frame, 
            text="Ativar Tempo Real", 
            variable=self.real_time_var,
            command=self.toggle_real_time,
            state=tk.DISABLED
        )
        self.real_time_check.grid(row=0, column=4, padx=5)

        self.steps_label = ttk.Label(row2_frame, text="Total Steps: 0 | Steps/s: 0.0")
        self.steps_label.grid(row=0, column=6, padx=5)

        # Gráficos:
        graph_frame = ttk.LabelFrame(main_frame, text="Desempenho em Tempo Real", padding="10")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.fig, self.axs = plt.subplots(5, 1, figsize=(10, 10), constrained_layout=True, sharex=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_plots()
        self.canvas.draw_idle()

        # Logs:
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
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=3) # Gráficos expandem
        main_frame.rowconfigure(2, weight=1) # Logs expandem

        control_frame.columnconfigure(0, weight=1)
        

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
        except Exception as e:
            self.logger.exception(f"Plot error")

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
        self.start_btn.config(state=tk.DISABLED, text="Iniciando...")
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.real_time_check.config(state=tk.NORMAL)

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

        # Usar valores dos checkboxes
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
            'start_time': datetime.now(),
            'environment': self.current_env,
            'robot': self.current_robot,
            'algorithm': self.current_algorithm,
            'episode_data': self.episode_data.copy(),
            'hyperparams': self.hyperparams
        }
        
        # Habilitar botão de salvamento
        self.save_training_btn.config(state=tk.NORMAL)
        self.export_plots_btn.config(state=tk.NORMAL)
        self.start_btn.config(text="Iniciar Treinamento", state=tk.DISABLED)


    def _resume_training(self):
        """Retoma um treinamento existente"""
        self.start_btn.config(state=tk.DISABLED, text="Retomando...")
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.real_time_check.config(state=tk.NORMAL)

        # Encontrar o modelo para retomada
        model_path = None
        models_dir = os.path.join(self.resumed_session_dir, 'models')
        
        # Busca flexível por modelos
        if os.path.exists(models_dir):
            zip_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            if zip_files:
                model_path = os.path.join(models_dir, zip_files[0])
                self.logger.info(f"Modelo encontrado para retomada: {zip_files[0]}")
        
        # Se não encontrou nos modelos, procurar no diretório principal
        if not model_path:
            zip_files = [f for f in os.listdir(self.resumed_session_dir) if f.endswith('.zip')]
            if zip_files:
                model_path = os.path.join(self.resumed_session_dir, zip_files[0])
                self.logger.info(f"Modelo encontrado no diretório principal: {zip_files[0]}")
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Erro", 
                f"Modelo não encontrado para retomada.\n"
                f"Diretório: {self.resumed_session_dir}\n"
                f"Certifique-se de que o treinamento foi salvo com sucesso.")
            self.start_btn.config(state=tk.NORMAL, text="Iniciar Treinamento")
            self.is_resuming = False
            return

        # Iniciar processo de retomada
        pause_val = multiprocessing.Value("b", 0)
        exit_val = multiprocessing.Value("b", 0)
        realtime_val = multiprocessing.Value("b", self.real_time_var.get())

        self.pause_values.append(pause_val)
        self.exit_values.append(exit_val)
        self.enable_real_time_values.append(realtime_val)

        self.logger.info(f"DEBUG _resume_training - current_episode: {self.current_episode}")
        self.logger.info(f"DEBUG _resume_training - Passando initial_episode: {self.current_episode} para o processo")        
        
        p = multiprocessing.Process(
            target=train_process.process_runner_resume,
            args=(self.current_env, self.current_robot, self.current_algorithm, self.ipc_queue, pause_val, exit_val, realtime_val, self.device, model_path, self.current_episode)
        )
        p.start()
        self.processes.append(p)

        self.logger.info(f"Processo de treinamento retomado: {self.current_env} + {self.current_robot} + {self.current_algorithm}")
        self.logger.info(f"Episódio inicial para retomada: {self.current_episode}")
        self.start_btn.config(text="Iniciar Treinamento", state=tk.DISABLED)
        self.is_resuming = False
        
    
    def pause_training(self):
        if not self.pause_values:
            self.logger.warning("pause_training: Nenhum processo de treinamento ativo.")
            return

        if self.pause_values[-1].value:
            self.logger.info("Retomando treinamento.")
            self.pause_values[-1].value = 0
            self.pause_btn.config(text="Pausar")
            
            # Salvar modelo ao retomar da pausa
            self._save_model_during_training()

        else:
            self.logger.info("Pausando treinamento.")
            self.pause_values[-1].value = 1
            self.pause_btn.config(text="Retomar")
            
            # Salvar modelo ao pausar
            self._save_model_during_training()
            

    def _save_model_during_training(self):
        """Salva o modelo durante o treinamento para permitir retomada"""
        if hasattr(self, 'current_training_session') and self.current_training_session:
            try:
                # Criar diretório de modelos se não existir
                models_dir = os.path.join(self.training_data_dir, "current_session", "models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Salvar modelo atual
                model_path = os.path.join(models_dir, "model.zip")
                
                # Enviar comando para salvar o modelo via IPC
                self.ipc_queue.put({"type": "save_model", "model_path": model_path})
                
                self.logger.info(f"Modelo salvo durante pausa/retomada: {model_path}")
            except Exception as e:
                self.logger.error(f"Erro ao salvar modelo durante treinamento: {e}")


    def stop_training(self):
        if self.exit_values:
            self.exit_values[-1].value = 1

        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.real_time_check.config(state=tk.DISABLED)
        self.save_training_btn.config(state=tk.DISABLED)
        self.export_plots_btn.config(state=tk.DISABLED)

            
    def save_training_data(self):
        """Salva todos os dados do treinamento atual incluindo o modelo"""
        if not self.current_training_session:
            messagebox.showwarning("Aviso", "Nenhum treinamento em andamento para salvar.")
            return
        
        try:
            # Criar diretório principal se não existir
            os.makedirs(self.training_data_dir, exist_ok=True)
            
            # Criar pasta específica para esta sessão
            timestamp = self.current_training_session['start_time'].strftime("%Y%m%d_%H%M%S")
            session_name = f"{self.current_env}_{self.current_robot}_{self.current_algorithm}_{timestamp}"
            session_dir = os.path.join(self.training_data_dir, session_name)
            os.makedirs(session_dir, exist_ok=True)
            
            if self.episode_data["episodes"]:
                current_episode_to_save = max(self.episode_data["episodes"])
                total_episodes = len(self.episode_data["episodes"])
            else:
                current_episode_to_save = 0
                total_episodes = 0
        
            # Salvar dados do treinamento
            training_data = {
                'session_info': {
                    'environment': self.current_env,
                    'robot': self.current_robot,
                    'algorithm': self.current_algorithm,
                    'start_time': self.current_training_session['start_time'].isoformat(),
                    'total_steps': self.total_steps,
                    'current_episode': current_episode_to_save,
                    'total_episodes': total_episodes,
                    'device': self.device
                },
                'episode_data': self.episode_data,
                'hyperparams': self.hyperparams
            }
            
            with open(os.path.join(session_dir, 'training_data.json'), 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # SALVAR MODELO - Usando sistema de arquivo de controle
            models_dir = os.path.join(session_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, 'model.zip')
            
            self.logger.info(f"INICIANDO SALVAMENTO DO MODELO: {model_path}")
            
            # Estratégia: Criar arquivo de controle
            control_dir = "training_control"
            os.makedirs(control_dir, exist_ok=True)
            
            control_file = f"save_model_{int(time.time() * 1000)}.json"
            control_path = os.path.join(control_dir, control_file)
            
            control_data = {
                "model_path": model_path,
                "timestamp": time.time(),
                "session": session_name
            }
            
            # Criar arquivo de controle
            with open(control_path, 'w') as f:
                json.dump(control_data, f, indent=2)
            
            self.logger.info(f"Arquivo de controle criado: {control_path}")
            
            # Aguardar salvamento
            max_wait = 21  # segundos
            check_interval = 1  # segundo
            model_saved = False
            
            self.logger.info(f"Aguardando salvamento (máximo {max_wait}s)...")
            
            for wait_time in range(max_wait):
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    self.logger.info(f"MODELO SALVO: {model_path} ({file_size} bytes)")
                    model_saved = True
                    break
                
                self.logger.info(f"Aguardando... {wait_time + 1}/{max_wait}s")
                time.sleep(check_interval)
            
            try:
                if os.path.exists(control_path):
                    os.remove(control_path)
                    self.logger.info("Arquivo de controle removido")
            except:
                pass
            
            # Salvar logs
            logs_dir = os.path.join(session_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Copiar arquivos de log relevantes
            log_files = [f for f in os.listdir('logs') if f.endswith('.txt')]
            for log_file in log_files:
                shutil.copy2(os.path.join('logs', log_file), logs_dir)
            
            # Salvar gráficos
            plots_dir = os.path.join(session_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            self.save_plots_to_directory(plots_dir)
            
            # Mensagem final
            if model_saved:
                messagebox.showinfo("Sucesso", 
                    f"Treinamento salvo com sucesso!\n"
                    f"Diretório: {session_dir}\n"
                    f"Modelo: {os.path.basename(model_path)}\n"
                    f"Tamanho: {os.path.getsize(model_path)} bytes\n"
                    f"Pronto para retomada!")
            else:
                # Verificar se o processo está ativo
                process_alive = False
                if self.processes:
                    process_alive = any(p.is_alive() for p in self.processes)
                
                if process_alive:
                    messagebox.showwarning("Aviso", 
                        f"Dados salvos, mas modelo não foi gerado.\n"
                        f"O processo está ativo mas não respondeu.\n"
                        f"Tente pausar o treinamento antes de salvar.\n"
                        f"Diretório: {session_dir}")
                else:
                    messagebox.showerror("Erro", 
                        f"Processo de treinamento não está ativo!\n"
                        f"O treinamento pode ter terminado ou travado.\n"
                        f"Diretório: {session_dir}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar treinamento: {e}")

    
    def load_training_data(self):
        """Carrega dados de treinamento salvos e prepara para retomada"""
        try:
            session_dir = filedialog.askdirectory(
                title="Selecione a pasta do treinamento",
                initialdir=self.training_data_dir
            )

            if not session_dir:
                return

            # Carregar dados do treinamento
            data_file = os.path.join(session_dir, 'training_data.json')
            if not os.path.exists(data_file):
                messagebox.showerror("Erro", "Arquivo de dados não encontrado.")
                return

            with open(data_file, 'r') as f:
                training_data = json.load(f)

            # BUSCA FLEXÍVEL POR MODELOS
            models_dir = os.path.join(session_dir, 'models')
            model_path = None

            # Primeiro: procurar por model.zip no diretório models
            if os.path.exists(models_dir):
                potential_path = os.path.join(models_dir, 'model.zip')
                if os.path.exists(potential_path):
                    model_path = potential_path
                    self.logger.info(f"Modelo encontrado: {model_path}")

            # Segundo: procurar por qualquer arquivo .zip no diretório models
            if not model_path and os.path.exists(models_dir):
                zip_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
                if zip_files:
                    model_path = os.path.join(models_dir, zip_files[0])
                    self.logger.info(f"Modelo alternativo encontrado: {zip_files[0]}")

            # Terceiro: procurar no diretório raiz da sessão
            if not model_path:
                zip_files = [f for f in os.listdir(session_dir) if f.endswith('.zip')]
                if zip_files:
                    model_path = os.path.join(session_dir, zip_files[0])
                    self.logger.info(f"Modelo encontrado no diretório principal: {zip_files[0]}")

            # Quarto: procurar em subdiretórios
            if not model_path:
                for root, dirs, files in os.walk(session_dir):
                    for file in files:
                        if file.endswith('.zip'):
                            model_path = os.path.join(root, file)
                            self.logger.info(f"Modelo encontrado em subdiretório: {model_path}")
                            break
                    if model_path:
                        break
                    
            if not model_path:
                messagebox.showerror("Erro", 
                    f"Nenhum modelo (.zip) encontrado para retomada.\n"
                    f"Diretório: {session_dir}\n"
                    f"Certifique-se de que:\n"
                    f"1. O treinamento foi salvo com sucesso\n"
                    f"2. O processo de treinamento estava ativo ao salvar\n"
                    f"3. O arquivo do modelo existe no diretório 'models/'")
                return

            # Restaurar dados do treinamento
            session_info = training_data['session_info']
            self.episode_data = training_data['episode_data']
            self.hyperparams = training_data.get('hyperparams', {})

            saved_current_episode = session_info.get('current_episode', 0)
            total_episodes = session_info.get('total_episodes', 0)
            if self.episode_data["episodes"]:
                last_episode_in_data = max(self.episode_data["episodes"])
                self.current_episode = max(saved_current_episode, last_episode_in_data) + 1
            else:
                self.current_episode = saved_current_episode + 1

            self.loaded_episode_count = total_episodes

            # Atualizar interface
            self.current_env = session_info['environment']
            self.current_robot = session_info['robot'] 
            self.current_algorithm = session_info['algorithm']

            # Configurar para retomada
            self.is_resuming = True
            self.resumed_session_dir = session_dir
            self.current_training_session = {
                'start_time': datetime.fromisoformat(session_info['start_time']),
                'environment': self.current_env,
                'robot': self.current_robot,
                'algorithm': self.current_algorithm,
                'episode_data': self.episode_data.copy(),
                'hyperparams': self.hyperparams
            }

            # Atualizar botão para "Retomar Treinamento"
            self.start_btn.config(text="Retomar Treinamento", state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)

            # Atualizar gráficos
            self.new_plot_data = True
            self._refresh_plots()

            # Habilitar botões
            self.save_training_btn.config(state=tk.NORMAL)
            self.export_plots_btn.config(state=tk.NORMAL)

            messagebox.showinfo("Sucesso", 
                f"Treinamento carregado!\n"
                f"Modelo: {os.path.basename(model_path)}\n"
                f"Último episódio: {saved_current_episode}\n"
                f"Próximo episódio: {self.current_episode}\n"
                f"Clique em 'Retomar Treinamento' para continuar.")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar treinamento: {e}")

    
    def export_plots(self):
        """Exporta gráficos como imagens para uso na tese"""
        try:
            export_dir = filedialog.askdirectory(
                title="Selecione onde salvar os gráficos",
                initialdir=os.path.expanduser("~")
            )
            
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
            dimension_window.bind('<Return>', lambda e: do_export())
            dimension_window.bind('<Escape>', lambda e: cancel_export())
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar gráficos: {e}")


    def save_plots_to_directory(self, directory, width=10, height=8, dpi=300):
        """Salva todos os gráficos em um diretório"""
        try:
            # Salvar gráfico combinado
            combined_fig, combined_axs = plt.subplots(5, 1, figsize=(width, height*2.5), constrained_layout=True)
            self._plot_to_figure(combined_axs)
            combined_fig.savefig(os.path.join(directory, 'all_plots.png'), dpi=dpi, bbox_inches='tight')
            plt.close(combined_fig)
            
            # Salvar gráficos individuais
            for i, (title, ylabel, color, data_key) in enumerate(zip(
                self.plot_titles, self.plot_ylabels, self.plot_colors, self.plot_data_keys
            )):
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
                fig.savefig(os.path.join(directory, filename), dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar gráficos: {e}")


    def _plot_to_figure(self, axs):
        """Plota dados nos eixos fornecidos"""
        for i, (title, ylabel, color, data_key) in enumerate(zip(
            self.plot_titles, self.plot_ylabels, self.plot_colors, self.plot_data_keys
        )):
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
