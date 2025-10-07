# tab_reward.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime


class RewardTab:
    def __init__(self, parent, device, logger, reward_system):
        self.frame = ttk.Frame(parent, padding="10")
        self.device = device
        self.logger = logger
        self.reward_system = reward_system

        # Configurações
        self.config_dir = "reward_configs"
        self.active_config_path = os.path.join(self.config_dir, "active.json")
        self.setup_directories()

        self.setup_ui()

    def setup_directories(self):
        """Cria diretórios de configuração"""
        self.logger.info(" RewardTab.setup_directories called")

        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "experiments"), exist_ok=True)

        # Criar config padrão se não existir
        default_config = os.path.join(self.config_dir, "default.json")
        if not os.path.exists(default_config):
            self.create_default_config()

    def create_default_config(self):
        """Cria configuração padrão"""
        self.logger.info(" RewardTab.create_default_config called")

        default_config = {
            "metadata": {"name": "default", "version": "1.0", "description": "Configuração balanceada para locomoção básica", "created": datetime.now().isoformat(), "author": "system"},
            "global_settings": {"fall_threshold": 0.5, "success_distance": 10.0, "platform_width": 1.0, "safe_zone": 0.2, "warning_zone": 0.4},
            "components": {
                "progress": {"weight": 15.0, "enabled": True},
                "distance_bonus": {"weight": 2.0, "enabled": True},
                "stability_roll": {"weight": -0.1, "enabled": True},
                "stability_pitch": {"weight": -0.4, "enabled": True},
                "yaw_penalty": {"weight": -2.0, "enabled": True},
                "fall_penalty": {"weight": -100.0, "enabled": True},
                "success_bonus": {"weight": 200.0, "enabled": True},
                "effort_penalty": {"weight": -0.001, "enabled": True},
                "jerk_penalty": {"weight": -0.05, "enabled": True},
                "center_bonus": {"weight": 5.0, "enabled": True},
                "warning_penalty": {"weight": -3.0, "enabled": True},
            },
        }

        with open(os.path.join(self.config_dir, "default.json"), "w") as f:
            json.dump(default_config, f, indent=2)

        # Tornar default a configuração ativa
        self.activate_configuration("default.json")

    def setup_ui(self):
        """Configura interface simplificada"""
        main_frame = ttk.Frame(self.frame)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controles principais
        control_frame = ttk.LabelFrame(main_frame, text="Gerenciamento de Configurações", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # Seletor de configuração
        ttk.Label(control_frame, text="Configuração Ativa:").grid(row=0, column=0, sticky=tk.W, pady=5)

        self.config_var = tk.StringVar()
        self.config_combo = ttk.Combobox(control_frame, textvariable=self.config_var, width=40, state="readonly")
        self.config_combo.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.config_combo.bind("<<ComboboxSelected>>", self.on_config_selected)

        ttk.Button(control_frame, text="Ativar", command=self.activate_selected_config).grid(row=0, column=2, padx=5)

        # Botões de ação
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Salvar Nova", command=self.create_new_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Salvar Como...", command=self.save_config_as).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Carregar", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Configuração Padrão", command=self.reset_to_default).pack(side=tk.LEFT, padx=5)

        # EDITOR COMPLETO COM TODAS AS CATEGORIAS
        editor_frame = ttk.LabelFrame(main_frame, text="Editor de Componentes de Recompensa", padding="10")
        editor_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.quick_editors = {}

        # CRÍTICOS (SOBREVIVÊNCIA)
        ttk.Label(editor_frame, text="Críticos (Sobrevivência)", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=5, sticky=tk.W, pady=(10, 5), padx=5)

        critical_components = [
            ("progress", "Progresso", 0, 250),
            ("distance_bonus", "Bônus de Distância", 0, 250),
            ("stability_roll", "Estabilidade Lateral", -50, 0),
            ("stability_pitch", "Estabilidade Frontal", -50, 0),
            ("yaw_penalty", "Rotação Horizontal", -50, 0),
            ("fall_penalty", "Penalidade por queda", -500, 0),
            ("success_bonus", "Bônus de Sucesso", 0, 500),
        ]

        row_idx = 1
        for comp_id, label, min_val, max_val in critical_components:
            self._create_component_row(editor_frame, comp_id, label, min_val, max_val, row_idx)
            row_idx += 1

        # EFICIÊNCIA
        ttk.Label(editor_frame, text="Eficiência", font=("Arial", 10, "bold")).grid(row=row_idx, column=0, columnspan=5, sticky=tk.W, pady=(15, 5), padx=5)
        row_idx += 1

        efficiency_components = [("effort_penalty", "Penalidade Esforço", -5, 0), ("jerk_penalty", "Penalidade Agito", -5, 0)]

        for comp_id, label, min_val, max_val in efficiency_components:
            self._create_component_row(editor_frame, comp_id, label, min_val, max_val, row_idx)
            row_idx += 1

        # NAVEGAÇÃO
        ttk.Label(editor_frame, text="Navegação", font=("Arial", 10, "bold")).grid(row=row_idx, column=0, columnspan=5, sticky=tk.W, pady=(15, 5), padx=5)
        row_idx += 1

        navigation_components = [("center_bonus", "Bônus Centro", 0, 200), ("warning_penalty", "Penalidade por Zona", -100, 0)]

        for comp_id, label, min_val, max_val in navigation_components:
            self._create_component_row(editor_frame, comp_id, label, min_val, max_val, row_idx)
            row_idx += 1

        # AVANÇADOS
        ttk.Label(editor_frame, text="Avançados", font=("Arial", 10, "bold")).grid(row=row_idx, column=0, columnspan=5, sticky=tk.W, pady=(15, 5), padx=5)
        row_idx += 1

        advanced_components = [("gait_regularity", "Regularidade da Marcha", 0, 50), ("symmetry_bonus", "Bônus de Simetria", 0, 50), ("clearance_bonus", "Bônus Elevação Pé", 0, 50)]

        for comp_id, label, min_val, max_val in advanced_components:
            self._create_component_row(editor_frame, comp_id, label, min_val, max_val, row_idx, default_enabled=False)
            row_idx += 1

        # Status
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)

        self.status_label = ttk.Label(status_frame, text="Configuração ativa: default")
        self.status_label.pack()

        # Configurar grid
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        editor_frame.columnconfigure(1, weight=1)

        # Carregar estado inicial
        self.refresh_config_list()
        self.load_active_config()

    def _create_component_row(self, parent, comp_id, label, min_val, max_val, row, default_enabled=True):
        """Cria uma linha de controle para um componente"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

        var = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, length=200)
        scale.grid(row=row, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))

        value_label = ttk.Label(parent, text="0.0", width=8)
        value_label.grid(row=row, column=2, padx=5, pady=2)

        value_entry = ttk.Entry(parent, width=8)
        value_entry.grid(row=row, column=3, padx=5, pady=2)
        value_entry.insert(0, "0.0")

        enabled_var = tk.BooleanVar(value=default_enabled)
        check_btn = ttk.Checkbutton(parent, variable=enabled_var)
        check_btn.grid(row=row, column=4, padx=5, pady=2)

        # Bind events
        var.trace("w", lambda *args, cid=comp_id, lbl=value_label, v=var, ent=value_entry: self.on_scale_change(cid, lbl, v, ent))
        value_entry.bind("<Return>", lambda e, cid=comp_id, v=var, ent=value_entry: self.on_entry_change(cid, v, ent))
        enabled_var.trace("w", lambda *args, cid=comp_id, ev=enabled_var: self.on_enabled_change(cid, ev))

        self.quick_editors[comp_id] = {"var": var, "scale": scale, "label": value_label, "entry": value_entry, "enabled_var": enabled_var, "check_btn": check_btn}

    def refresh_config_list(self):
        """Atualiza lista de configurações disponíveis"""
        self.logger.info(" RewardTab.refresh_config_list called")

        configs = []

        # Configurações de treino
        training_dir = os.path.join(self.config_dir, "training")
        for config_file in os.listdir(training_dir):
            if config_file.endswith(".json"):
                configs.append(f"training/{config_file[:-5]}")

        # Configurações experimentais
        experiments_dir = os.path.join(self.config_dir, "experiments")
        for config_file in os.listdir(experiments_dir):
            if config_file.endswith(".json"):
                configs.append(f"experiments/{config_file[:-5]}")

        # Configurações na raiz
        for config_file in os.listdir(self.config_dir):
            if config_file.endswith(".json") and config_file != "active.json":
                configs.append(config_file[:-5])

        self.config_combo["values"] = configs

    def load_active_config(self):
        """Carrega configuração ativa"""
        self.logger.info(" RewardTab.load_active_config called")

        try:
            if os.path.exists(self.active_config_path):
                with open(self.active_config_path, "r") as f:
                    active_info = json.load(f)

                config_path = active_info.get("active_config", "default.json")

                if config_path.endswith(".json"):
                    config_name = config_path[:-5]
                else:
                    config_name = config_path

                full_path = os.path.join(self.config_dir, config_path)

                if os.path.exists(full_path):
                    # Carregar no sistema de recompensas
                    self.reward_system.load_configuration_file(full_path)

                    # Atualizar UI
                    self.config_var.set(config_name)
                    self.status_label.config(text=f"Configuração ativa: {config_name}")

                    # Atualizar sliders
                    self.update_quick_editors()
                    return True

            # Fallback para default
            self.activate_configuration("default")
            return True

        except Exception as e:
            self.logger.exception("Erro ao carregar configuração ativa")
            return False

    def update_quick_editors(self):
        """Atualiza todos os controles com valores atuais"""
        self.logger.info(" RewardTab.update_quick_editors called")

        config = self.reward_system.get_configuration()

        for comp_id, editor in self.quick_editors.items():
            if comp_id in config:
                weight = config[comp_id]["weight"]
                enabled = config[comp_id]["enabled"]

                # Atualizar variáveis (isso dispara os callbacks)
                editor["var"].set(weight)
                editor["enabled_var"].set(enabled)

                # Atualizar widgets diretamente para evitar loops
                editor["label"].config(text=f"{weight:.3f}")
                editor["entry"].delete(0, tk.END)
                editor["entry"].insert(0, f"{weight:.3f}")

    def on_config_selected(self, event):
        """Quando uma configuração é selecionada no combobox"""
        self.logger.info(" RewardTab.on_config_selected called")

        config_name = self.config_var.get()
        self.load_config_by_name(config_name)

    def load_config_by_name(self, config_name):
        """Carrega configuração pelo nome"""
        self.logger.info(" RewardTab.load_config_by_name called")

        try:
            if "/" in config_name:
                category, name = config_name.split("/")
                config_path = os.path.join(self.config_dir, category, f"{name}.json")
            else:
                config_path = os.path.join(self.config_dir, f"{config_name}.json")

            if os.path.exists(config_path):
                self.reward_system.load_configuration_file(config_path)
                self.update_quick_editors()
                self.status_label.config(text=f"Configuração carregada: {config_name}")
            else:
                messagebox.showerror("Erro", f"Configuração não encontrada: {config_path}")

        except Exception as e:
            self.logger.exception("Erro ao carregar configuração selecionada")
            messagebox.showerror("Erro", f"Falha ao carregar configuração: {e}")

    def activate_selected_config(self):
        """Ativa a configuração selecionada"""
        self.logger.info(" RewardTab.activate_selected_config called")

        config_name = self.config_var.get()
        if config_name:
            success = self.activate_configuration(config_name)
            if success:
                messagebox.showinfo("Sucesso", f"Configuração '{config_name}' ativada!")

    def activate_configuration(self, config_name):
        """Ativa uma configuração específica"""
        self.logger.info(" RewardTab.activate_configuration called")

        try:
            # Remover .json se já estiver presente
            if config_name.endswith(".json"):
                config_name = config_name[:-5]

            # Determinar caminho completo
            if "/" in config_name:
                category, name = config_name.split("/")
                config_path = f"{category}/{name}.json"
            else:
                config_path = f"{config_name}.json"

            full_path = os.path.join(self.config_dir, config_path)

            if os.path.exists(full_path):
                # Criar arquivo de configuração ativa
                active_info = {"active_config": config_path, "activated_at": datetime.now().isoformat(), "name": config_name}

                with open(self.active_config_path, "w") as f:
                    json.dump(active_info, f, indent=2)

                # Carregar no sistema
                self.reward_system.load_configuration_file(full_path)
                self.config_var.set(config_name)
                self.status_label.config(text=f"Configuração ativa: {config_name}")
                self.update_quick_editors()

                self.logger.info(f"Configuração ativada: {config_name}")
                return True
            else:
                messagebox.showerror("Erro", f"Arquivo de configuração não encontrado: {full_path}")
                return False

        except Exception as e:
            self.logger.exception("Erro ao ativar configuração")
            messagebox.showerror("Erro", f"Falha ao ativar configuração: {e}")
            return False

    def create_new_config(self):
        """Cria nova configuração baseada na atual"""
        self.logger.info(" RewardTab.create_new_config called")

        name = tk.simpledialog.askstring("Nova Configuração", "Nome da configuração:")
        if name:
            description = tk.simpledialog.askstring("Descrição", "Descrição da configuração:")

            config_data = {
                "metadata": {"name": name, "version": "1.0", "description": description or "", "created": datetime.now().isoformat(), "based_on": self.config_var.get() or "default"},
                "global_settings": {
                    "fall_threshold": self.reward_system.fall_threshold,
                    "success_distance": self.reward_system.success_distance,
                    "platform_width": self.reward_system.platform_width,
                    "safe_zone": self.reward_system.safe_zone,
                    "warning_zone": self.reward_system.warning_zone,
                },
                "components": self.reward_system.get_configuration(),
            }

            # Salvar em training por padrão
            config_path = os.path.join(self.config_dir, "training", f"{name}.json")

            try:
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)

                self.refresh_config_list()
                self.config_var.set(f"training/{name}")
                messagebox.showinfo("Sucesso", f"Configuração '{name}' criada!")

            except Exception as e:
                self.logger.exception("Erro ao criar nova configuração")
                messagebox.showerror("Erro", f"Falha ao criar configuração: {e}")

    def save_config_as(self):
        """Salva configuração atual com novo nome"""
        self.logger.info(" RewardTab.save_config_as called")

        name = tk.simpledialog.askstring("Salvar Como", "Nome da configuração:")
        if name:
            self.create_new_config_with_name(name)

    def create_new_config_with_name(self, name):
        """Cria configuração com nome específico"""
        self.logger.info(" RewardTab.create_new_config_with_name called")

        config_data = {
            "metadata": {
                "name": name,
                "version": "1.0",
                "description": f"Configuração salva em {datetime.now().isoformat()}",
                "created": datetime.now().isoformat(),
                "based_on": self.config_var.get() or "default",
            },
            "global_settings": {
                "fall_threshold": self.reward_system.fall_threshold,
                "success_distance": self.reward_system.success_distance,
                "platform_width": self.reward_system.platform_width,
                "safe_zone": self.reward_system.safe_zone,
                "warning_zone": self.reward_system.warning_zone,
            },
            "components": self.reward_system.get_configuration(),
        }

        config_path = os.path.join(self.config_dir, "training", f"{name}.json")

        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            self.refresh_config_list()
            messagebox.showinfo("Sucesso", f"Configuração salva como '{name}'")

        except Exception as e:
            self.logger.exception("Erro ao salvar configuração")
            messagebox.showerror("Erro", f"Falha ao salvar configuração: {e}")

    def load_config(self):
        """Carrega configuração de arquivo"""
        self.logger.info(" RewardTab.load_config called")

        filepath = filedialog.askopenfilename(title="Carregar Configuração", initialdir=self.config_dir, filetypes=[("JSON files", "*.json"), ("All files", "*.*")])

        if filepath:
            try:
                # Copiar para diretório de configurações
                filename = os.path.basename(filepath)
                dest_path = os.path.join(self.config_dir, "training", filename)

                # Evitar sobrescrever
                if os.path.exists(dest_path):
                    overwrite = messagebox.askyesno("Confirmar", f"Configuração '{filename}' já existe. Sobrescrever?")
                    if not overwrite:
                        return

                import shutil

                shutil.copy2(filepath, dest_path)

                self.refresh_config_list()
                config_name = f"training/{filename[:-5]}"
                self.config_var.set(config_name)
                self.load_config_by_name(config_name)

                messagebox.showinfo("Sucesso", f"Configuração '{filename}' carregada!")

            except Exception as e:
                self.logger.exception("Erro ao carregar configuração")
                messagebox.showerror("Erro", f"Falha ao carregar configuração: {e}")

    def on_scale_change(self, component_id, label_widget, var, entry_widget):
        """Callback quando slider é movido"""
        value = var.get()
        label_widget.config(text=f"{value:.3f}")
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, f"{value:.3f}")

        # Atualizar no sistema imediatamente
        self.reward_system.update_component(component_id, weight=value)

    def on_entry_change(self, component_id, var, entry_widget):
        """Callback quando valor é digitado no entry"""
        try:
            value = float(entry_widget.get())
            var.set(value)
            # on_scale_change será chamado automaticamente pelo trace
        except ValueError:
            # Se valor inválido, restaurar o anterior
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, f"{var.get():.3f}")

    def on_enabled_change(self, component_id, enabled_var):
        """Callback quando checkbox é alterado"""
        enabled = enabled_var.get()
        self.reward_system.update_component(component_id, enabled=enabled)

    def reset_to_default(self):
        """Restaura configuração padrão"""
        self.logger.info(" RewardTab.reset_to_default called")

        # Passar apenas o nome sem .json
        self.activate_configuration("default")
        messagebox.showinfo("Sucesso", "Configuração restaurada para padrão")

    def start(self):
        """Inicia a aba"""
        self.logger.info("Aba de recompensas simplificada inicializada")
        self.load_active_config()

    def on_closing(self):
        """Limpeza ao fechar"""
        pass
