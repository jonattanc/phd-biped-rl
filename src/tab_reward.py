# tab_reward.py
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import utils


class RewardTab:
    def __init__(self, parent, device, logger, reward_system):
        self.frame = ttk.Frame(parent, padding="10")
        self.device = device
        self.logger = logger
        self.reward_system = reward_system

        # Configurações
        self.config_dir = utils.REWARD_CONFIGS_PATH
        self.settings = utils.load_default_settings()
        self.setup_ui()

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
        self.config_var.set("default")
        self.config_combo = ttk.Combobox(control_frame, textvariable=self.config_var, width=40, state="readonly")
        self.config_combo.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.config_combo.bind("<<ComboboxSelected>>", self.on_config_selected)

        ttk.Button(control_frame, text="Criar Nova Configuração", command=self.create_new_config).grid(row=0, column=2, padx=5)

        # EDITOR COMPLETO COM TODAS AS CATEGORIAS
        editor_frame = ttk.LabelFrame(main_frame, text="Editor de Componentes de Recompensa", padding="10")
        editor_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.quick_editors = {}
        components = self.reward_system.get_configuration_as_dict()
        row_idx = 1

        for key, value in components.items():
            self._create_component_row(editor_frame, key, value, row_idx)
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
        editor_frame.columnconfigure(4, weight=1)

        # Carregar estado inicial
        self.refresh_config_list()
        self.update_quick_editors()

        self.config_var.set(self.settings.get("reward_config_file", "default"))
        self.on_config_selected()

    def _create_component_row(self, parent, component_name, component_content, row, default_enabled=True):
        """Cria uma linha de controle para um componente"""
        ttk.Label(parent, text=component_name).grid(row=row, column=3, sticky=tk.W, padx=5, pady=2)

        var = tk.DoubleVar()
        scale = ttk.Scale(parent, from_=component_content["min_value"], to=component_content["max_value"], variable=var, orient=tk.HORIZONTAL, length=200)
        scale.grid(row=row, column=4, padx=5, pady=2, sticky=(tk.W, tk.E))

        value_label = ttk.Label(parent, text=component_content["weight"], width=8)
        value_label.grid(row=row, column=1, padx=5, pady=2)

        value_entry = ttk.Entry(parent, width=8)
        value_entry.grid(row=row, column=2, padx=5, pady=2)
        value_entry.insert(0, component_content["weight"])

        enabled_var = tk.BooleanVar(value=default_enabled)
        check_btn = ttk.Checkbutton(parent, variable=enabled_var)
        check_btn.grid(row=row, column=0, padx=5, pady=2)

        # Bind events
        var.trace("w", lambda *args, cid=component_name, lbl=value_label, v=var, ent=value_entry: self.on_scale_change(cid, lbl, v, ent))
        value_entry.bind("<Return>", lambda e, cid=component_name, v=var, ent=value_entry: self.on_entry_change(cid, v, ent))
        enabled_var.trace("w", lambda *args, cid=component_name, ev=enabled_var: self.on_enabled_change(cid, ev))

        self.quick_editors[component_name] = {"var": var, "scale": scale, "label": value_label, "entry": value_entry, "enabled_var": enabled_var, "check_btn": check_btn}

    def refresh_config_list(self):
        """Atualiza lista de configurações disponíveis"""
        self.logger.info(" RewardTab.refresh_config_list called")

        config_files = []

        for root, _, files in os.walk(self.config_dir):
            for file in files:
                full_path = os.path.join(root, file)
                self.logger.info(f"Found config file: {full_path}")

                rel_path = os.path.relpath(full_path, self.config_dir)
                self.logger.info(f"Relative path: {rel_path}")

                config_files.append(rel_path.replace("\\", "/").replace(".json", ""))

        self.config_combo["values"] = config_files

    def update_quick_editors(self):
        """Atualiza todos os controles com valores atuais"""
        self.logger.info(" RewardTab.update_quick_editors called")

        config = self.reward_system.get_configuration_as_dict()

        for comp_id, editor in self.quick_editors.items():
            weight = config[comp_id]["weight"]
            enabled = config[comp_id]["enabled"]

            # Atualizar variáveis (isso dispara os callbacks)
            editor["var"].set(weight)
            editor["enabled_var"].set(enabled)

            # Atualizar widgets diretamente para evitar loops
            editor["label"].config(text=f"{weight}")
            editor["entry"].delete(0, tk.END)
            editor["entry"].insert(0, f"{weight}")

    def on_config_selected(self, event=None):
        """Quando uma configuração é selecionada no combobox"""
        self.logger.info(" RewardTab.on_config_selected called")

        try:
            config_name = self.config_var.get()
            self.reward_system.load_configuration_file(config_name)
            self.update_quick_editors()
            self.status_label.config(text=f"Configuração carregada: {config_name}")

        except Exception as e:
            self.logger.exception("Erro ao carregar configuração selecionada")
            messagebox.showerror("Erro", f"Falha ao carregar configuração: {e}")

    def create_new_config(self):
        """Cria nova configuração baseada na atual"""
        self.logger.info(" RewardTab.create_new_config called")

        try:
            name = tk.simpledialog.askstring("Nova Configuração", "Nome da configuração:")

            if not name:
                return

            config_data = {
                "components": self.reward_system.get_configuration_as_dict(),
            }

            config_path = os.path.join(self.config_dir, f"{name}.json")

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            self.refresh_config_list()
            self.config_var.set(name)

        except Exception as e:
            self.logger.exception("Erro ao criar nova configuração")
            messagebox.showerror("Erro", f"Falha ao criar configuração: {e}")

    def on_scale_change(self, component_id, label_widget, var, entry_widget):
        """Callback quando slider é movido"""
        try:
            value = var.get()
            label_widget.config(text=f"{value}")
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, f"{value}")

            # Atualizar no sistema imediatamente
            self.save_config_change(component_id, weight=value)

        except Exception as e:
            self.logger.exception(f"Erro ao atualizar valor do componente {component_id}: {e}")
            messagebox.showerror("Erro", f"Falha ao atualizar valor: {e}")

    def on_entry_change(self, component_id, var, entry_widget):
        """Callback quando valor é digitado no entry"""
        try:
            try:
                value = float(entry_widget.get())
                var.set(value)
                # on_scale_change será chamado automaticamente pelo trace
            except ValueError:
                # Se valor inválido, restaurar o anterior
                self.logger.warning(f"Valor inválido digitado para {component_id}: {entry_widget.get()}")
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, f"{var.get()}")

        except Exception as e:
            self.logger.exception(f"Erro ao processar entrada do componente {component_id}: {e}")
            messagebox.showerror("Erro", f"Falha ao processar entrada: {e}")

    def on_enabled_change(self, component_id, enabled_var):
        """Callback quando checkbox é alterado"""
        try:
            enabled = enabled_var.get()
            self.save_config_change(component_id, enabled=enabled)

        except Exception as e:
            self.logger.exception(f"Erro ao atualizar estado do componente {component_id}: {e}")
            messagebox.showerror("Erro", f"Falha ao atualizar estado: {e}")

    def save_config_change(self, component_id, weight=None, enabled=None):
        self.reward_system.update_component(component_id, weight=weight, enabled=enabled)

        config_data = {
            "components": self.reward_system.get_configuration_as_dict(),
        }

        config_name = self.config_var.get()
        config_path = os.path.join(self.config_dir, f"{config_name}.json")

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def start(self):
        """Inicia a aba"""
        self.logger.info("Aba de recompensas simplificada inicializada")

    def on_closing(self):
        """Limpeza ao fechar"""
        pass
