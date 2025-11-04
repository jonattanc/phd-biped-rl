# common_tab.py
import utils
import tkinter as tk
from tkinter import ttk, messagebox
import os
import json


class GUITab:
    def __init__(self, gui, device, logger, reward_system, notebook):
        self.gui = gui
        self.device = device
        self.logger = logger
        self.reward_system = reward_system
        self.notebook = notebook

        self.processes = []
        self.pause_values = []
        self.exit_values = []
        self.enable_real_time_values = []
        self.enable_visualization_values = []
        self.camera_selection_values = []
        self.config_changed_values = []

    def create_environment_selector(self, frame, column):
        xacro_env_files = self._get_xacro_files(utils.ENVIRONMENT_PATH)

        if not xacro_env_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ENVIRONMENT_PATH}.")
            return

        ttk.Label(frame, text="Ambiente:").grid(row=0, column=column, sticky=tk.W, padx=1)
        self.env_var = tk.StringVar(value=xacro_env_files[0])
        self.env_combo = ttk.Combobox(frame, textvariable=self.env_var, values=xacro_env_files, width=10)
        self.env_combo.grid(row=0, column=column + 1, padx=5)

    def create_robot_selector(self, frame, column, enabled=True):
        xacro_robot_files = self._get_xacro_files(utils.ROBOTS_PATH)

        if not xacro_robot_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ROBOTS_PATH}.")
            return

        ttk.Label(frame, text="Robô:").grid(row=0, column=column, sticky=tk.W, padx=1)
        self.robot_var = tk.StringVar(value=self.gui.settings.get("default_robot", xacro_robot_files[-1]))
        self.robot_combo = ttk.Combobox(frame, textvariable=self.robot_var, values=xacro_robot_files, width=12)
        self.robot_combo.grid(row=0, column=column + 1, padx=5)

        if not enabled:
            self.robot_combo.config(state=tk.DISABLED)

    def create_seed_selector(self, frame, column):
        self.seed_var = tk.IntVar(value=42)
        self.seed_input = ttk.Spinbox(frame, from_=0, to=100000, textvariable=self.seed_var, width=8)
        ttk.Label(frame, text="Seed:").grid(row=0, column=column, sticky=tk.W, padx=1)
        self.seed_input.grid(row=0, column=column + 1, padx=5)

    def create_dpg_selector(self, frame, column):
        self.enable_dpg_var = tk.BooleanVar(value=self.gui.settings.get("enable_dynamic_policy_gradient", True))
        self.enable_dpg_check = ttk.Checkbutton(frame, text="Dynamic Policy Gradient", variable=self.enable_dpg_var, width=22)
        self.enable_dpg_check.grid(row=0, column=column, padx=1)

    def create_enable_visualization_selector(self, frame, column):
        self.enable_visualization_var = tk.BooleanVar(value=self.gui.settings.get("enable_visualize_robot", False))
        self.enable_visualization_check = ttk.Checkbutton(frame, text="Visualizar Robô", variable=self.enable_visualization_var, command=self.toggle_visualization, width=15)
        self.enable_visualization_check.grid(row=0, column=column, padx=1)

    def create_real_time_selector(self, frame, column):
        self.real_time_var = tk.BooleanVar(value=self.gui.settings.get("enable_real_time", True))
        self.real_time_check = ttk.Checkbutton(frame, text="Tempo Real", variable=self.real_time_var, command=self.toggle_real_time, width=15)
        self.real_time_check.grid(row=0, column=column, padx=5)

    def create_camera_selector(self, frame, column):
        ttk.Label(frame, text="Câmera:").grid(row=0, column=column, sticky=tk.W, padx=5)

        camera_options = {
            1: "Ambiente geral",
            2: "Robô - Diagonal direita",
            3: "Robô - Diagonal esquerda",
            4: "Robô - Lateral direita",
            5: "Robô - Lateral esquerda",
            6: "Robô - Frontal",
            7: "Robô - Traseira",
        }
        self.camera_selection_int = self.gui.settings.get("camera_index", 1)
        self.camera_selection_var = tk.StringVar(value=camera_options[self.camera_selection_int])
        self.camera_selection_combobox = ttk.Combobox(frame, textvariable=self.camera_selection_var, values=list(camera_options.values()), state="readonly", width=25)
        self.camera_selection_combobox.grid(row=0, column=column + 1, padx=5)
        self.camera_selection_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_camera_selection(event, camera_options))

    def create_pause_btn(self, frame, column):
        self.pause_btn = ttk.Button(frame, text="Pausar", command=self.pause_training, state=tk.DISABLED, width=10)
        self.pause_btn.grid(row=0, column=column, padx=1)

    def create_stop_btn(self, frame, column):
        self.stop_btn = ttk.Button(frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED, width=10)
        self.stop_btn.grid(row=0, column=column, padx=1)

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

    def _load_training_data_file(self, session_dir):
        """Carrega arquivo de dados de treinamento"""
        data_file = os.path.join(session_dir, "training_data.json")
        if not os.path.exists(data_file):
            raise FileNotFoundError("Arquivo de dados não encontrado.")

        with open(data_file, "r") as f:
            return json.load(f)

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
        if self.exit_values:
            self.exit_values[-1].value = 1
            self.config_changed_values[-1].value = 1

        self.unlock_gui()

        self.logger.info("Simulação finalizada pelo usuário")

    def get_environment_settings(self, env):
        environment_settings = {"default": {"lateral_friction": 2.0, "spinning_friction": 1.0, "rolling_friction": 0.001, "restitution": 0.0}}

        if self.env_var.get() == "PBA":
            environment_settings["middle_link"] = {"lateral_friction": 1.0, "spinning_friction": 0.5, "rolling_friction": 0.001, "restitution": 0.0}

        self.logger.info(f"environment_settings: {environment_settings}")
        return environment_settings

    def set_gui_state(self, state):
        if state == tk.NORMAL:
            opposite_state = tk.DISABLED

        else:
            opposite_state = tk.NORMAL

        if hasattr(self, "start_btn"):
            self.start_btn.config(state=opposite_state)

        if hasattr(self, "eval_start_btn"):
            self.eval_start_btn.config(state=opposite_state)

        if hasattr(self, "save_training_btn"):
            self.save_training_btn.config(state=state)

        if hasattr(self, "load_training_btn"):
            self.load_training_btn.config(state=opposite_state)

        if hasattr(self, "enable_dpg_check"):
            self.enable_dpg_check.config(state=opposite_state)

        if hasattr(self, "pause_btn"):
            self.pause_btn.config(state=state)
            self.pause_btn.config(text="Pausar")

        if hasattr(self, "stop_btn"):
            self.stop_btn.config(state=state)

        if hasattr(self, "algorithm_combo"):
            self.algorithm_combo.config(state=opposite_state)

        if hasattr(self, "env_combo"):
            self.env_combo.config(state=opposite_state)

        if hasattr(self, "robot_combo"):
            self.robot_combo.config(state=opposite_state)

        if hasattr(self, "seed_input"):
            self.seed_input.config(state=opposite_state)

        if hasattr(self, "load_model_btn"):
            self.load_model_btn.config(state=opposite_state)

        if hasattr(self, "episodes_spinbox"):
            self.episodes_spinbox.config(state=opposite_state)

    def lock_gui(self):
        self.disable_other_tabs()
        self.set_gui_state(tk.NORMAL)

    def unlock_gui(self):
        self.enable_other_tabs()
        self.set_gui_state(tk.DISABLED)
