# tab_comparison.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import queue
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_ipc_logging


class ComparisonTab:
    def __init__(self, parent, device, logger):
        self.frame = ttk.Frame(parent)
        self.device = device
        self.logger = logger
        
        # IPC Queue para comunicação
        self.ipc_queue = queue.Queue()
        
        # Dados de comparação
        self.comparison_results = []
        self.models_to_compare = []
        
        # Componentes da UI
        self.models_listbox = None
        self.comparison_text = None
        self.fig_comparison = None
        self.axs_comparison = None
        self.canvas_comparison = None

        # Configurar IPC logging
        setup_ipc_logging(self.logger, self.ipc_queue)

        # Controle de callbacks
        self.after_ids = []
        self.gui_active = True
        
        self.setup_ui()

    def setup_ui(self):
        """Configura a interface da aba de comparação"""
        # Frame principal
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controles de comparação
        control_frame = ttk.LabelFrame(main_frame, text="Comparação de Modelos", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Frame para lista de modelos
        models_frame = ttk.Frame(control_frame)
        models_frame.pack(fill=tk.X, pady=5)

        ttk.Label(models_frame, text="Modelos para Comparar:").pack(side=tk.TOP, anchor=tk.W, pady=(0, 5))

        # Listbox para modelos selecionados com frame e scrollbar
        listbox_frame = ttk.Frame(models_frame)
        listbox_frame.pack(fill=tk.X, expand=True)

        self.models_listbox = tk.Listbox(listbox_frame, height=6, selectmode=tk.EXTENDED)
        listbox_scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.models_listbox.yview)
        self.models_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        self.models_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Botões para gerenciar lista
        buttons_frame = ttk.Frame(models_frame)
        buttons_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Button(buttons_frame, text="Adicionar Modelo", 
                  command=self.add_comparison_model, width=15).pack(pady=2, fill=tk.X)
        ttk.Button(buttons_frame, text="Remover Selecionado", 
                  command=self.remove_comparison_model, width=15).pack(pady=2, fill=tk.X)
        ttk.Button(buttons_frame, text="Limpar Lista", 
                  command=self.clear_comparison_models, width=15).pack(pady=2, fill=tk.X)

        # Configurações de avaliação
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=10)

        ttk.Label(settings_frame, text="Ambiente:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.comparison_env_var = tk.StringVar(value="PR")
        env_combo = ttk.Combobox(settings_frame, textvariable=self.comparison_env_var, 
                                values=["PR", "Pmu", "RamA", "RamD", "PG", "PRB"], width=10)
        env_combo.grid(row=0, column=1, padx=5)

        ttk.Label(settings_frame, text="Robô:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.comparison_robot_var = tk.StringVar(value="robot_stage1")
        robot_combo = ttk.Combobox(settings_frame, textvariable=self.comparison_robot_var, 
                                  values=["robot_stage1", "robot_stage2", "robot_stage3"], width=12)
        robot_combo.grid(row=0, column=3, padx=5)

        ttk.Label(settings_frame, text="Episódios por modelo:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.comparison_episodes_var = tk.StringVar(value="10")
        ttk.Entry(settings_frame, textvariable=self.comparison_episodes_var, width=8).grid(row=0, column=5, padx=5)

        # Botão de execução
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.run_comparison_btn = ttk.Button(button_frame, text="Executar Comparação", 
                                           command=self.run_comparison, width=20)
        self.run_comparison_btn.pack(pady=5)

        # Frame para resultados e gráficos
        results_frame = ttk.LabelFrame(main_frame, text="Resultados da Comparação", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Notebook para organizar resultados textuais e gráficos
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)

        # Aba de resultados textuais
        text_frame = ttk.Frame(results_notebook)
        results_notebook.add(text_frame, text="Tabela Comparativa")

        # Texto com resultados
        self.comparison_text = tk.Text(text_frame, height=15, state=tk.DISABLED, wrap=tk.NONE)
        text_scrollbar_y = ttk.Scrollbar(text_frame, orient="vertical", command=self.comparison_text.yview)
        text_scrollbar_x = ttk.Scrollbar(text_frame, orient="horizontal", command=self.comparison_text.xview)
        self.comparison_text.configure(yscrollcommand=text_scrollbar_y.set, xscrollcommand=text_scrollbar_x.set)
        
        self.comparison_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        text_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Aba de gráficos
        graph_frame = ttk.Frame(results_notebook)
        results_notebook.add(graph_frame, text="Gráficos Comparativos")

        # Gráficos de comparação
        self.fig_comparison, self.axs_comparison = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_comparison = FigureCanvasTkAgg(self.fig_comparison, master=graph_frame)
        self.canvas_comparison.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._initialize_comparison_plots()

        # Botões de exportação
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, pady=5)

        ttk.Button(export_frame, text="Exportar Resultados", 
                  command=self.export_comparison_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Exportar Gráficos", 
                  command=self.export_comparison_plots).pack(side=tk.LEFT, padx=5)

    def _initialize_comparison_plots(self):
        """Inicializa os gráficos de comparação"""
        try:
            # Gráfico de comparação de taxa de sucesso
            self.axs_comparison[0, 0].set_title("Taxa de Sucesso")
            self.axs_comparison[0, 0].set_ylabel("Taxa de Sucesso (%)")
            self.axs_comparison[0, 0].grid(True, alpha=0.3)
            
            # Gráfico de comparação de tempo médio
            self.axs_comparison[0, 1].set_title("Tempo Médio")
            self.axs_comparison[0, 1].set_ylabel("Tempo (s)")
            self.axs_comparison[0, 1].grid(True, alpha=0.3)
            
            # Gráfico de comparação de desvio padrão
            self.axs_comparison[1, 0].set_title("Desvio Padrão")
            self.axs_comparison[1, 0].set_ylabel("Desvio Padrão (s)")
            self.axs_comparison[1, 0].grid(True, alpha=0.3)
            
            # Gráfico radar de performance
            self.axs_comparison[1, 1].set_title("Performance Relativa")
            self.axs_comparison[1, 1].grid(True, alpha=0.3)
            
            self.canvas_comparison.draw_idle()
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar gráficos de comparação: {e}")

    def add_comparison_model(self):
        """Adiciona modelo à lista de comparação"""
        filename = filedialog.askopenfilename(
            title="Selecionar Modelo para Comparação",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")],
            initialdir="training_data"
        )
        if filename:
            model_name = os.path.basename(filename)
            display_name = model_name.replace('.zip', '').replace('model_', '')
            self.models_listbox.insert(tk.END, f"{display_name}||{filename}")
            self.logger.info(f"Modelo adicionado para comparação: {display_name}")

    def remove_comparison_model(self):
        """Remove modelo selecionado da lista de comparação"""
        selection = self.models_listbox.curselection()
        if selection:
            for index in reversed(selection):
                item = self.models_listbox.get(index)
                model_name = item.split("||")[0]
                self.models_listbox.delete(index)
                self.logger.info(f"Modelo removido da comparação: {model_name}")

    def clear_comparison_models(self):
        """Limpa toda a lista de modelos para comparação"""
        if self.models_listbox.size() > 0:
            self.models_listbox.delete(0, tk.END)
            self.logger.info("Lista de modelos para comparação limpa")

    def run_comparison(self):
        """Executa comparação entre modelos selecionados"""
        if self.models_listbox.size() < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 modelos para comparação.")
            return

        try:
            episodes = int(self.comparison_episodes_var.get())
            if episodes <= 0:
                raise ValueError("Número de episódios deve ser positivo")
        except ValueError:
            messagebox.showerror("Erro", "Número de episódios deve ser um número inteiro positivo.")
            return

        # Coletar modelos selecionados
        models = []
        for i in range(self.models_listbox.size()):
            item = self.models_listbox.get(i)
            name, path = item.split("||")
            models.append({"name": name, "path": path})

        # Desabilitar botão durante comparação
        self.run_comparison_btn.config(state=tk.DISABLED, text="Comparando...")

        # Executar comparação em thread separada
        comparison_thread = threading.Thread(target=self._run_comparison, args=(models, episodes), daemon=True)
        comparison_thread.start()

    def _run_comparison(self, models, episodes):
        """Executa a comparação entre modelos em thread separada"""
        try:
            from evaluate_model import evaluate_and_save
            
            comparison_results = []
            environment = self.comparison_env_var.get()
            robot = self.comparison_robot_var.get()
            
            total_models = len(models)
            
            for i, model in enumerate(models):
                self.logger.info(f"Avaliando modelo {i+1}/{total_models}: {model['name']}")
                
                # Atualizar progresso na interface
                self._update_progress(i, total_models, model['name'])
                
                metrics = evaluate_and_save(
                    model_path=model['path'],
                    circuit_name=environment,
                    avatar_name=robot,
                    num_episodes=episodes,
                    seed=42
                )
                
                if metrics:
                    comparison_results.append({
                        'name': model['name'],
                        'metrics': metrics,
                        'path': model['path']
                    })
                else:
                    self.logger.warning(f"Falha na avaliação do modelo: {model['name']}")
            
            # Atualizar interface com resultados
            self.root.after(0, lambda: self._display_comparison_results(comparison_results))
            
        except Exception as e:
            self.logger.error(f"Erro na comparação: {e}")
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na comparação: {e}"))
        finally:
            self.root.after(0, lambda: self.run_comparison_btn.config(state=tk.NORMAL, text="Executar Comparação"))

    def _update_progress(self, current, total, model_name):
        """Atualiza o progresso na interface"""
        progress_text = f"Avaliando {current+1}/{total}: {model_name}"
        self.root.after(0, lambda: self.run_comparison_btn.config(text=progress_text))

    def _display_comparison_results(self, results):
        """Exibe os resultados da comparação"""
        try:
            self.comparison_results = results
            
            self.comparison_text.config(state=tk.NORMAL)
            self.comparison_text.delete(1.0, tk.END)
            
            comparison_text = "=== COMPARAÇÃO DE MODELOS ===\n\n"
            comparison_text += f"Ambiente: {self.comparison_env_var.get()} | "
            comparison_text += f"Robô: {self.comparison_robot_var.get()} | "
            comparison_text += f"Episódios por modelo: {self.comparison_episodes_var.get()}\n"
            comparison_text += f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
            
            # Criar tabela comparativa
            headers = ["Modelo", "Taxa Sucesso", "Tempo Médio", "Desvio Padrão", "Melhor Tempo", "Pior Tempo"]
            comparison_text += f"{headers[0]:<30} {headers[1]:<15} {headers[2]:<12} {headers[3]:<15} {headers[4]:<12} {headers[5]:<12}\n"
            comparison_text += "-" * 110 + "\n"
            
            for result in results:
                metrics = result['metrics']
                name = result['name']
                success_rate = metrics.get('success_rate', 0) * 100
                avg_time = metrics.get('avg_time', 0)
                std_time = metrics.get('std_time', 0)
                times = metrics.get('total_times', [0])
                best_time = min(times) if times else 0
                worst_time = max(times) if times else 0
                
                comparison_text += f"{name:<30} {success_rate:<15.1f}% {avg_time:<12.2f}s {std_time:<15.2f}s {best_time:<12.2f}s {worst_time:<12.2f}s\n"
            
            # Análise comparativa
            if results:
                comparison_text += "\n" + "="*50 + "\n"
                comparison_text += "ANÁLISE COMPARATIVA:\n"
                comparison_text += "="*50 + "\n"
                
                # Melhor por categoria
                best_success = max(results, key=lambda x: x['metrics'].get('success_rate', 0))
                best_time = min(results, key=lambda x: x['metrics'].get('avg_time', float('inf')))
                most_consistent = min(results, key=lambda x: x['metrics'].get('std_time', float('inf')))
                
                comparison_text += f"• Melhor taxa de sucesso: {best_success['name']} ({best_success['metrics']['success_rate']*100:.1f}%)\n"
                comparison_text += f"• Melhor tempo médio: {best_time['name']} ({best_time['metrics']['avg_time']:.2f}s)\n"
                comparison_text += f"• Mais consistente: {most_consistent['name']} (σ={most_consistent['metrics']['std_time']:.2f}s)\n\n"
                
                # Ranking geral
                comparison_text += "RANKING GERAL (tempo médio):\n"
                ranked_results = sorted(results, key=lambda x: x['metrics'].get('avg_time', float('inf')))
                for i, result in enumerate(ranked_results, 1):
                    comparison_text += f"{i}º - {result['name']}: {result['metrics']['avg_time']:.2f}s "
                    comparison_text += f"({result['metrics']['success_rate']*100:.1f}% sucesso)\n"
            
            self.comparison_text.insert(1.0, comparison_text)
            self.comparison_text.config(state=tk.DISABLED)
            
            # Atualizar gráficos
            self._update_comparison_plots(results)
            
            messagebox.showinfo("Sucesso", f"Comparação concluída! {len(results)} modelos avaliados.")
            
        except Exception as e:
            self.logger.error(f"Erro ao exibir comparação: {e}")
            messagebox.showerror("Erro", f"Erro ao exibir resultados: {e}")

    def _update_comparison_plots(self, results):
        """Atualiza os gráficos de comparação com os resultados"""
        try:
            if not results:
                return
                
            # Limpar gráficos
            for ax in self.axs_comparison.flat:
                ax.clear()
            
            model_names = [result['name'] for result in results]
            success_rates = [result['metrics'].get('success_rate', 0) * 100 for result in results]
            avg_times = [result['metrics'].get('avg_time', 0) for result in results]
            std_times = [result['metrics'].get('std_time', 0) for result in results]
            
            # Gráfico de barras - Taxa de sucesso
            bars1 = self.axs_comparison[0, 0].bar(model_names, success_rates, color='lightgreen', alpha=0.7)
            self.axs_comparison[0, 0].set_title("Taxa de Sucesso por Modelo")
            self.axs_comparison[0, 0].set_ylabel("Taxa de Sucesso (%)")
            self.axs_comparison[0, 0].tick_params(axis='x', rotation=45)
            self.axs_comparison[0, 0].grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars1, success_rates):
                height = bar.get_height()
                self.axs_comparison[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                                              f'{value:.1f}%', ha='center', va='bottom')
            
            # Gráfico de barras - Tempo médio
            bars2 = self.axs_comparison[0, 1].bar(model_names, avg_times, color='lightblue', alpha=0.7)
            self.axs_comparison[0, 1].set_title("Tempo Médio por Modelo")
            self.axs_comparison[0, 1].set_ylabel("Tempo (s)")
            self.axs_comparison[0, 1].tick_params(axis='x', rotation=45)
            self.axs_comparison[0, 1].grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars2, avg_times):
                height = bar.get_height()
                self.axs_comparison[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                              f'{value:.2f}s', ha='center', va='bottom')
            
            # Gráfico de barras - Desvio padrão
            bars3 = self.axs_comparison[1, 0].bar(model_names, std_times, color='lightcoral', alpha=0.7)
            self.axs_comparison[1, 0].set_title("Desvio Padrão por Modelo")
            self.axs_comparison[1, 0].set_ylabel("Desvio Padrão (s)")
            self.axs_comparison[1, 0].tick_params(axis='x', rotation=45)
            self.axs_comparison[1, 0].grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars3, std_times):
                height = bar.get_height()
                self.axs_comparison[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                              f'{value:.2f}s', ha='center', va='bottom')
            
            # Gráfico de radar (simplificado) - Performance relativa
            metrics_to_plot = ['success_rate', 'avg_time', 'consistency']
            metrics_values = []
            
            for result in results:
                metrics = result['metrics']
                # Normalizar valores para radar chart
                success_norm = metrics.get('success_rate', 0)
                time_norm = 1 - (metrics.get('avg_time', 0) / max(avg_times)) if max(avg_times) > 0 else 0
                consistency_norm = 1 - (metrics.get('std_time', 0) / max(std_times)) if max(std_times) > 0 else 0
                
                metrics_values.append([success_norm, time_norm, consistency_norm])
            
            # Plot simples de performance
            performance_scores = [(sr * 0.4 + tt * 0.4 + cs * 0.2) * 100 for sr, tt, cs in metrics_values]
            
            bars4 = self.axs_comparison[1, 1].bar(model_names, performance_scores, color='gold', alpha=0.7)
            self.axs_comparison[1, 1].set_title("Score de Performance")
            self.axs_comparison[1, 1].set_ylabel("Score (%)")
            self.axs_comparison[1, 1].tick_params(axis='x', rotation=45)
            self.axs_comparison[1, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, performance_scores):
                height = bar.get_height()
                self.axs_comparison[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                              f'{value:.1f}%', ha='center', va='bottom')
            
            # Ajustar layout
            self.fig_comparison.tight_layout()
            self.canvas_comparison.draw_idle()
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar gráficos de comparação: {e}")

    def export_comparison_results(self):
        """Exporta os resultados da comparação para arquivo JSON"""
        if not self.comparison_results:
            messagebox.showwarning("Aviso", "Nenhum resultado de comparação para exportar.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Salvar Resultados da Comparação",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if filename:
                export_data = {
                    'metadata': {
                        'environment': self.comparison_env_var.get(),
                        'robot': self.comparison_robot_var.get(),
                        'episodes_per_model': int(self.comparison_episodes_var.get()),
                        'timestamp': datetime.now().isoformat(),
                        'total_models': len(self.comparison_results)
                    },
                    'results': self.comparison_results
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Sucesso", f"Resultados exportados para:\n{filename}")
                self.logger.info(f"Resultados de comparação exportados: {filename}")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar resultados: {e}")
            self.logger.error(f"Erro ao exportar resultados: {e}")

    def export_comparison_plots(self):
        """Exporta os gráficos de comparação como imagens"""
        if not self.comparison_results:
            messagebox.showwarning("Aviso", "Nenhum gráfico para exportar.")
            return
        
        try:
            directory = filedialog.askdirectory(
                title="Selecione onde salvar os gráficos"
            )
            
            if directory:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Salvar gráfico combinado
                fig_combined, axs_combined = plt.subplots(2, 2, figsize=(12, 10))
                self._plot_to_export_figure(axs_combined, self.comparison_results)
                fig_combined.suptitle(f"Comparação de Modelos - {self.comparison_env_var.get()}", fontsize=16)
                fig_combined.savefig(os.path.join(directory, f'comparison_combined_{timestamp}.png'), 
                                   dpi=300, bbox_inches='tight')
                plt.close(fig_combined)
                
                # Salvar gráficos individuais
                plots_dir = os.path.join(directory, f'comparison_plots_{timestamp}')
                os.makedirs(plots_dir, exist_ok=True)
                
                plot_types = ['success_rate', 'avg_time', 'std_dev', 'performance']
                for plot_type in plot_types:
                    fig_single = plt.figure(figsize=(10, 6))
                    self._create_single_plot(fig_single, plot_type, self.comparison_results)
                    fig_single.savefig(os.path.join(plots_dir, f'{plot_type}_{timestamp}.png'), 
                                     dpi=300, bbox_inches='tight')
                    plt.close(fig_single)
                
                messagebox.showinfo("Sucesso", f"Gráficos exportados para:\n{directory}")
                self.logger.info(f"Gráficos de comparação exportados: {directory}")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar gráficos: {e}")
            self.logger.error(f"Erro ao exportar gráficos: {e}")

    def _plot_to_export_figure(self, axs, results):
        """Plota dados nos eixos fornecidos para exportação"""
        model_names = [result['name'] for result in results]
        success_rates = [result['metrics'].get('success_rate', 0) * 100 for result in results]
        avg_times = [result['metrics'].get('avg_time', 0) for result in results]
        std_times = [result['metrics'].get('std_time', 0) for result in results]
        
        # Replicar a lógica de plotagem do _update_comparison_plots
        bars1 = axs[0, 0].bar(model_names, success_rates, color='lightgreen', alpha=0.7)
        axs[0, 0].set_title("Taxa de Sucesso por Modelo")
        axs[0, 0].set_ylabel("Taxa de Sucesso (%)")
        axs[0, 0].tick_params(axis='x', rotation=45)
        axs[0, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, success_rates):
            height = bar.get_height()
            axs[0, 0].text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}%', 
                          ha='center', va='bottom', fontsize=8)
        

    def _create_single_plot(self, fig, plot_type, results):
        """Cria um gráfico individual para exportação"""
        # Implementação similar aos gráficos individuais
        pass

    def start(self):
        """Inicializa a aba de comparação"""
        self.logger.info("Aba de comparação inicializada")

    @property
    def root(self):
        """Retorna a root window do tkinter"""
        return self.frame.winfo_toplevel()