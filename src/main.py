# main.py
import torch
import os
import shutil
import tkinter as tk
from tkinter import ttk
import utils
import environment
import multiprocessing

# Importar as tabs
from tab_training import TrainingTab
from tab_evaluation import EvaluationTab
from tab_comparison import ComparisonTab
from tab_reward import RewardTab
from reward_system import RewardSystem


class TrainingGUI:
    def __init__(self, device="cpu"):
        self.root = tk.Tk()
        self.root.title("Generalização Cruzada")
        self.root.geometry("1400x1000")

        self.device = device
        self.logger = utils.get_logger()
        self.reward_system = RewardSystem(self.logger)

        self.settings = utils.load_default_settings()
        self.setup_ui()

        # Configurar o handler para fechar a janela
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Notebook com abas
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Criar abas
        self.reward_tab = RewardTab(notebook, self.device, self.logger, self.reward_system)
        self.training_tab = TrainingTab(self, notebook, self.device, self.logger, self.reward_system, notebook)
        self.evaluation_tab = EvaluationTab(self, notebook, self.device, self.logger)
        self.comparison_tab = ComparisonTab(notebook, self.device, self.logger)

        # Adicionar abas ao notebook
        notebook.add(self.training_tab.frame, text="Treinamento")
        notebook.add(self.evaluation_tab.frame, text="Avaliação Individual")
        notebook.add(self.comparison_tab.frame, text="Avaliação Cruzada")
        notebook.add(self.reward_tab.frame, text="Configuração de Recompensas")

    def on_closing(self):
        """Fecha todas as abas adequadamente"""
        self.logger.info("Fechando aplicação...")

        # Chamar cleanup nas abas que possuem o método
        if hasattr(self, "training_tab") and hasattr(self.training_tab, "on_closing"):
            self.training_tab.on_closing()

        if hasattr(self, "evaluation_tab") and hasattr(self.evaluation_tab, "cleanup"):
            self.evaluation_tab.cleanup()
        elif hasattr(self, "evaluation_tab") and hasattr(self.evaluation_tab, "on_closing"):
            self.evaluation_tab.on_closing()

        if hasattr(self, "comparison_tab") and hasattr(self.comparison_tab, "cleanup"):
            self.comparison_tab.cleanup()
        elif hasattr(self, "comparison_tab") and hasattr(self.comparison_tab, "on_closing"):
            self.comparison_tab.on_closing()

        self.root.destroy()

    def start(self):
        # Iniciar componentes das abas
        self.reward_tab.start()
        self.training_tab.start()
        self.evaluation_tab.start()
        self.comparison_tab.start()

        self.root.mainloop()


def setup_folders():
    """Configura as pastas necessárias para a aplicação"""
    logs = []

    for folder in [utils.TMP_PATH, utils.LOGS_PATH]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except PermissionError:
                logs.append(f"Não foi possível deletar {folder} (arquivo em uso). Continuando...")

        os.makedirs(folder, exist_ok=True)

    utils.ensure_directory(utils.TRAINING_DATA_PATH)
    utils.ensure_directory(utils.TEMP_MODEL_SAVE_PATH)

    return logs


def check_gpu():
    """Verifica disponibilidade da GPU e retorna dispositivo"""
    logger = utils.get_logger()

    logger.info("Verificando GPU")
    logger.info(f"CUDA version: {torch.version.cuda}")
    is_gpu_available = torch.cuda.is_available()
    logger.info(f"GPU available: {is_gpu_available}")

    if is_gpu_available:
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"

    return device


if __name__ == "__main__":
    # Configurar ambiente
    folder_logs = setup_folders()
    logger = utils.get_logger()
    environment.create_ramp_stl("ramp_up.stl", ascending=True)
    environment.create_ramp_stl("ramp_down.stl", ascending=False)

    if folder_logs:
        logger.info("\n".join(folder_logs))

    logger.info(f"Executando em {utils.PROJECT_ROOT}")

    multiprocessing.set_start_method("spawn")

    # Verificar GPU
    device = check_gpu()

    # Iniciar aplicação
    logger.info("Iniciando GUI...")
    try:
        app = TrainingGUI(device)
        app.start()
        logger.info("Programa finalizado com sucesso.")
    except Exception as e:
        logger.exception("Erro ao executar aplicação")
        raise
