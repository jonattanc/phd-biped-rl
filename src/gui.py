# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import logging
import os
import csv
import time
import pybullet as p
from simulation import Simulation
from robot import Robot
from environment import Environment
from agent import Agent

class TrainingGUI:
    def __init__(self, root, agent=None):
        self.root = root
        self.agent = agent
        self.root.title("Cruzada Generalization - Training Dashboard")
        self.root.geometry("1200x800")

        # Dados de treinamento
        self.training_queue = queue.Queue()
        self.running = False
        self.current_env = ""
        self.current_robot = ""
        self.episode_data = {"episodes": [], "rewards": [], "times": [], "distances": []}
        self.fig, self.axs = plt.subplots(3, figsize=(10, 8))
        self.canvas = None
        self.episode_logger = None
        self.hyperparams = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Interface de treinamento inicializada.")
        self.setup_ui()

    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controle de Treinamento", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(control_frame, text="Ambiente:").grid(row=0, column=0, sticky=tk.W)
        self.env_var = tk.StringVar(value="PR")
        env_combo = ttk.Combobox(control_frame, textvariable=self.env_var, values=["PR", "P<μ", "RamA", "RamD", "PG", "PRB"])
        env_combo.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Robô:").grid(row=0, column=2, sticky=tk.W)
        self.robot_var = tk.StringVar(value="robot_stage1")
        robot_combo = ttk.Combobox(control_frame, textvariable=self.robot_var, values=["robot_stage1"])
        robot_combo.grid(row=0, column=3, padx=5)

        self.start_btn = ttk.Button(control_frame, text="Iniciar Treinamento", command=self.start_training)
        self.start_btn.grid(row=0, column=4, padx=5)

        self.pause_btn = ttk.Button(control_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=5, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Finalizar", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=6, padx=5)

        self.save_btn = ttk.Button(control_frame, text="Salvar Snapshot", command=self.save_snapshot, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=7, padx=5)

        self.visualize_btn = ttk.Button(control_frame, text="Visualizar Simulação", command=self.toggle_visualization)
        self.visualize_btn.grid(row=0, column=8, padx=5)

        # Gráficos
        graph_frame = ttk.LabelFrame(main_frame, text="Desempenho em Tempo Real", padding="10")
        graph_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.fig, self.axs = plt.subplots(3, figsize=(10, 6))
        self.axs[0].set_title("Recompensa por Episódio")
        self.axs[1].set_title("Duração do Episódio (s)")
        self.axs[2].set_title("Distância Percorrida (m)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # Logs
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
        if self.running:
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

        self.current_env = self.env_var.get()
        self.current_robot = self.robot_var.get()

        # Iniciar treinamento em thread separada
        self.thread = threading.Thread(target=self.run_training_loop, daemon=True)
        self.thread.start()

    def run_training_loop(self):
        
        """Loop principal de treinamento em thread"""
        try:
            self.logger.info(f"Iniciando treinamento PPO para {self.current_env} com agente {self.current_robot}")

            # Treinar o agente por 100k passos
            self.agent.train(total_timesteps=100_000)

            self.logger.info("Treinamento concluído!")

        except Exception as e:
            self.logger.error(f"Erro inesperado no treinamento: {e}", exc_info=True)
        finally:
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.NORMAL)  # Habilita salvar o modelo treinado

    def update_plots(self):
        """Atualiza os gráficos com novos dados da fila"""
        while not self.training_queue.empty():
            data = self.training_queue.get()
            if data.get("type") == "log":
                # Atualiza a caixa de texto de log
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, data["message"] + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
            else:
                # É um dado de desempenho (recompensa, tempo, distância)
                self.episode_data["episodes"].append(data["episode"])
                self.episode_data["rewards"].append(data["reward"])
                self.episode_data["times"].append(data["time"])
                self.episode_data["distances"].append(data["distance"])

                # Atualizar gráficos
                self.axs[0].clear()
                self.axs[0].plot(self.episode_data["episodes"], self.episode_data["rewards"], label="Recompensa", marker='o', linestyle='-')
                self.axs[0].legend()
                self.axs[0].set_title("Recompensa por Episódio")
                self.axs[0].grid(True)

                self.axs[1].clear()
                self.axs[1].plot(self.episode_data["episodes"], self.episode_data["times"], label="Duração (s)", color='orange', marker='s', linestyle='-')
                self.axs[1].legend()
                self.axs[1].set_title("Duração do Episódio (s)")
                self.axs[1].grid(True)

                self.axs[2].clear()
                self.axs[2].plot(self.episode_data["episodes"], self.episode_data["distances"], label="Distância (m)", color='green', marker='^', linestyle='-')
                self.axs[2].legend()
                self.axs[2].set_title("Distância Percorrida (m)")
                self.axs[2].grid(True)

                self.canvas.draw()

        # Atualizar periodicamente
        self.root.after(1000, self.update_plots)

    def pause_training(self):
        self.running = False
        self.pause_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)

    def stop_training(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Treinamento Finalizado", "Treinamento encerrado.")

    def save_snapshot(self):
        """Salva o modelo treinado e executa avaliação para gerar métricas de complexidade."""
        if self.agent.model is None:
            messagebox.showwarning("Aviso", "Nenhum modelo treinado para salvar ou avaliar.")
            return

        # --- Passo 1: Salvar o modelo PPO ---
        # Alterado: Salvar em logs/data/models em vez de perguntar ao usuário
        models_dir = "logs/data/models"
        os.makedirs(models_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(models_dir, f"model_{self.current_env}_{self.current_robot}_{timestamp}.zip")

        self.agent.model.save(model_filename)
        self.logger.info(f"Modelo salvo em: {model_filename}")

        # --- Passo 2: Criar ambiente de avaliação ---
        try:
            from gym_env import ExoskeletonPRst1
            eval_env = ExoskeletonPRst1(enable_gui=False, seed=42)
        except ImportError:
            messagebox.showerror("Erro", "Ambiente Gym 'ExoskeletonPRst1' não encontrado. Verifique se gym_env.py existe.")
            return
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao criar ambiente de avaliação: {str(e)}")
            return

        # --- Passo 3: Executar Avaliação usando a função do evaluate_model ---
        try:
            self.logger.info("Iniciando avaliação do agente treinado (20 episódios, deterministic=True)...")

            # Usar a função de avaliação do evaluate_model
            metrics = self.evaluate_and_save(model_filename, self.current_env, self.current_robot)

            if metrics:
                self.logger.info(f"Avaliação concluída. "
                                 f"Sucesso: {metrics['success_rate']*100:.1f}% | "
                                 f"Tm: {metrics['avg_time']:.2f}s ± {metrics['std_time']:.2f}s")

                # --- Passo 4: Compilar resultados e gerar relatório ---
                from metrics_saver import compile_results, generate_report

                # Compilar todos os resultados
                compiled_df = compile_results()

                if compiled_df is not None:
                    # Gerar relatório completo
                    generate_report()

                    messagebox.showinfo("Avaliação Concluída",
                                        f"Modelo salvo em: {model_filename}\n\n"
                                        f"Taxa de Sucesso: {metrics['success_rate']*100:.1f}%\n"
                                        f"Tempo Médio: {metrics['avg_time']:.2f}s ± {metrics['std_time']:.2f}s\n\n"
                                        f"Relatório completo gerado em logs/data/")
                else:
                    messagebox.showinfo("Avaliação Concluída",
                                        f"Modelo salvo em: {model_filename}\n\n"
                                        f"Taxa de Sucesso: {metrics['success_rate']*100:.1f}%\n"
                                        f"Tempo Médio: {metrics['avg_time']:.2f}s ± {metrics['std_time']:.2f}s")

            else:
                messagebox.showerror("Erro", "Falha na avaliação do modelo.")

        except Exception as e:
            self.logger.error(f"Erro durante a avaliação: {e}", exc_info=True)
            messagebox.showerror("Erro na Avaliação", f"Não foi possível avaliar o agente.\nErro: {str(e)}")
        finally:
            try:
                eval_env.close()  # Fecha a conexão do ambiente de avaliação
            except:
                pass

    def evaluate_and_save(self, model_path, circuit_name="PR", avatar_name="robot_stage1", 
                         role="AE", num_episodes=20, seed=42):
        """Avalia um modelo e salva as métricas - integração com evaluate_model.py"""

        self.logger.info(f"Avaliando {avatar_name} no circuito {circuit_name}...")

        try:
            from gym_env import ExoskeletonPRst1
            env = ExoskeletonPRst1(enable_gui=False, seed=seed)

            # Carregar o agente com o modelo salvo
            agent = Agent(model_path=model_path)
            metrics = agent.evaluate(env, num_episodes=num_episodes)

            hyperparams = {
                "algorithm": "PPO",
                "learning_rate": 3e-4,
                "total_timesteps": 100000  # Ajustado para o treinamento da GUI
            }

            # Salvar métricas usando metrics_saver
            from metrics_saver import save_complexity_metrics
            save_complexity_metrics(
                metrics=metrics,
                circuit_name=circuit_name,
                avatar_name=avatar_name,
                role=role,
                seed=seed,
                hyperparams=hyperparams
            )

            self.logger.info(f"Sucesso: {metrics['success_rate']*100:.1f}% | Tempo: {metrics['avg_time']:.2f}s")
            return metrics

        except Exception as e:
            self.logger.error(f"Erro na avaliação: {e}")
            return None
        finally:
            try:
                env.close()
            except:
                pass

    def toggle_visualization(self):
        """Abre uma nova janela com simulação em tempo real (GUI Bullet)"""
        # Implementação futura: pode ser feita com outro processo
        # Por enquanto, apenas alerta
        messagebox.showinfo("Visualização", "Funcionalidade de visualização em tempo real ainda não implementada.\n\nUse `enable_gui=True` na simulação direta.")

    def update_logs(self):
        """Atualiza a caixa de log com as últimas linhas do arquivo de log principal"""
        log_file = os.path.join("logs", "training_log.txt") 
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:  # <-- IGNORA CARACTERES INVÁLIDOS
                    lines = f.readlines()
                    # Mostrar as últimas 50 linhas (ou menos, se o arquivo for pequeno)
                    last_lines = lines[-50:] if len(lines) > 50 else lines
                    log_content = ''.join(last_lines)
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.delete(1.0, tk.END)
                    self.log_text.insert(tk.END, log_content)
                    self.log_text.see(tk.END)  # Rolagem automática para o final
                    self.log_text.config(state=tk.DISABLED)
            except Exception as e:
                self.logger.error(f"Erro ao ler o arquivo de log: {e}")
        # Atualizar periodicamente
        self.root.after(2000, self.update_logs)
    
    def start(self):
        self.root.after(500, self.update_plots)
        self.root.after(500, self.update_logs)       # Começa a atualizar logs
        self.root.mainloop()