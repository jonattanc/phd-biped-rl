# common_tab.py
import utils
import tkinter as tk
from tkinter import ttk, messagebox
import os


class GUITab:
    def __init__(self, gui, logger):
        self.gui = gui
        self.logger = logger

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
        env_combo = ttk.Combobox(frame, textvariable=self.env_var, values=xacro_env_files, width=10)
        env_combo.grid(row=0, column=column + 1, padx=5)

    def create_robot_selector(self, frame, column):
        xacro_robot_files = self._get_xacro_files(utils.ROBOTS_PATH)

        if not xacro_robot_files:
            messagebox.showerror("Erro", f"Nenhum arquivo .xacro encontrado em {utils.ROBOTS_PATH}.")
            return

        ttk.Label(frame, text="Robô:").grid(row=0, column=column, sticky=tk.W, padx=1)
        self.robot_var = tk.StringVar(value=self.gui.settings.get("default_robot", xacro_robot_files[-1]))
        robot_combo = ttk.Combobox(frame, textvariable=self.robot_var, values=xacro_robot_files, width=12)
        robot_combo.grid(row=0, column=column + 1, padx=5)

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
