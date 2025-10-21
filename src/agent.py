# agent.py
import time
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise


class FastTD3(TD3):
    """
    Implementação do Fast TD3 com atualizações mais frequentes
    """

    def __init__(self, *args, **kwargs):
        action_dim = kwargs.pop("action_dim", 1)

        # Configurações otimizadas para Fast TD3
        kwargs.update(
            {
                "learning_rate": 3e-4,
                "buffer_size": 200000,
                "learning_starts": 5000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": (1, "step"),
                "gradient_steps": 1,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "action_noise": NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)),
            }
        )
        super().__init__(*args, **kwargs)


class TrainingCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.custom_logger = logger

    def _on_step(self) -> bool:
        try:
            # Verificar se há informações de ambiente
            infos = self.locals.get("infos", [])

            # Se houver infos e alguma indicar saída, parar
            if infos and any(isinstance(info, dict) and info.get("exit", False) for info in infos):
                return False

        except Exception as e:
            # Em caso de erro, continuar o treinamento
            self.custom_logger.exception("Erro no callback de treinamento")

        return True


class Agent:
    def __init__(self, logger, env=None, model_path=None, algorithm="PPO", device="cpu", initial_episode=0):
        self.logger = logger
        self.model = None
        self.algorithm = algorithm
        self.env = env
        self.action_dim = 0
        self.initial_episode = initial_episode

        if model_path is None:
            # Criar ambiente vetorizado
            dummy_env = DummyVecEnv([lambda: env])
            self.env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True)
            self.action_dim = env.action_dim
            self.model = self._create_model(algorithm, device)

        else:
            self._load_model(model_path)
            self.set_env(env)

    def _create_model(self, algorithm, device="cpu"):
        # Criar modelo baseado no algoritmo selecionado
        if algorithm.upper() == "PPO":
            return PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=1e-4,
                n_steps=4096,
                batch_size=128,
                n_epochs=20,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
                vf_coef=0.7,
                max_grad_norm=0.8,
                tensorboard_log="./logs/",
                device="cpu",
            )
        elif algorithm.upper() == "TD3":
            # policy_kwargs = dict(net_arch=[256, 256, 256])
            return TD3(
                "MlpPolicy",
                self.env,
                # policy_kwargs=policy_kwargs,
                # verbose=1,
                learning_rate=1.0e-4,
                buffer_size=int(1e6),
                learning_starts=1e4,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_delay=3,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                tensorboard_log="./logs/",
                device=device,
            )
        elif algorithm.upper() == "FASTTD3":
            return FastTD3(
                "MlpPolicy",
                self.env,
                verbose=1,
                action_dim=self.action_dim,
                tensorboard_log="./logs/",
                device=device,
            )
        else:
            raise ValueError(f"Algoritmo {algorithm} não suportado. Use 'PPO', 'TD3' ou 'FastTD3'")

    def save_model(self, model_path):
        """Salva o modelo treinado"""
        if self.model is not None:
            self.model.save(model_path)
            self.logger.info(f"Modelo salvo em: {model_path}")
        else:
            raise ValueError("Nenhum modelo para salvar")

    def _load_model(self, model_path):
        # Carrega modelo treinado detectando automaticamente o tipo
        try:
            # Tentar carregar como PPO primeiro
            self.model = PPO.load(model_path)
            self.algorithm = "PPO"
            if hasattr(self.model, "action_space") and self.model.action_space is not None:
                self.action_dim = self.model.action_space.shape[0]
            self.logger.info(f"Modelo PPO carregado: {model_path}")
        except Exception as e:
            try:
                self.model = TD3.load(model_path)
                self.algorithm = "TD3"
                if hasattr(self.model, "action_space") and self.model.action_space is not None:
                    self.action_dim = self.model.action_space.shape[0]
                self.logger.info(f"Modelo TD3 carregado: {model_path}")
            except Exception as e:
                # Tentar carregar como FastTD3
                self.model = FastTD3.load(model_path)
                self.algorithm = "FastTD3"
                if hasattr(self.model, "action_space") and self.model.action_space is not None:
                    self.action_dim = self.model.action_space.shape[0]
                self.logger.info(f"Modelo FastTD3 carregado: {model_path}")

    def set_env(self, env):
        """Configura o ambiente para um modelo carregado"""
        self.logger.info("=== CONFIGURANDO AMBIENTE NO AGENT ===")

        if self.model is None:
            self.logger.error("Modelo não disponível para configurar ambiente")
            return

        if env is None:
            self.logger.error("Ambiente não disponível")
            return

        try:
            # Criar ambiente vetorizado
            self.logger.info("Criando ambiente vetorizado...")
            vec_env = DummyVecEnv([lambda: env])

            # Configurar o ambiente no modelo
            self.logger.info("Configurando ambiente no modelo...")
            self.model.set_env(vec_env)
            self.env = vec_env

            # Tentar obter action_dim de várias formas
            if hasattr(env, "action_dim"):
                self.action_dim = env.action_dim
                self.logger.info(f"Action dim do env: {self.action_dim}")
            elif hasattr(env, "action_space"):
                self.action_dim = env.action_space.shape[0]
                self.logger.info(f"Action dim do action_space: {self.action_dim}")
            elif self.model.action_space is not None:
                self.action_dim = self.model.action_space.shape[0]
                self.logger.info(f"Action dim do model: {self.action_dim}")
            else:
                self.logger.warning("Não foi possível determinar action_dim")

            self.logger.info("Ambiente configurado com sucesso no agente")

        except Exception as e:
            self.logger.exception("Erro ao configurar ambiente")
            raise

    def set_agent(self, agent):
        """Configura o agente no ambiente"""
        self.agent = agent
        self.logger.info(f"Agente {agent.algorithm} configurado na simulação")

    def evaluate(self, env, num_episodes=20, deterministic=True):
        """
        Avalia o agente treinado em um ambiente.
        """
        self.logger.info("=== INICIANDO AVALIAÇÃO ===")

        if self.model is None:
            self.logger.error("Nenhum modelo treinado carregado para avaliação.")
            return None

        # Configurar ambiente se necessário - MAS NÃO substituir a referência existente
        try:
            if self.model.get_env() is None:
                self.logger.info("Configurando ambiente no modelo...")
                vec_env = DummyVecEnv([lambda: env])
                self.model.set_env(vec_env)
                self.logger.info("Ambiente configurado")
        except Exception as e:
            self.logger.exception("Erro ao configurar ambiente")
            return None

        # Configurar o agente no ambiente
        env.agent = self
        self.logger.info("Agente configurado no ambiente")

        total_times = []
        success_count = 0
        total_rewards = []

        self.logger.info(f"Executando {num_episodes} episódios de avaliação...")

        for episode in range(num_episodes):
            try:
                self.logger.info(f"--- Episódio {episode + 1}/{num_episodes} ---")

                # Verificar se o modelo ainda está disponível
                if self.model is None:
                    self.logger.error("Modelo se tornou None durante a avaliação")
                    break

                # Reset do ambiente
                obs, _ = env.reset()
                done = False
                steps = 0
                episode_reward = 0
                episode_success = False
                episode_start_time = time.time()

                while not done and steps < 1000:  # Limite de steps por segurança
                    # Verificar novamente se o modelo está disponível
                    if self.model is None:
                        self.logger.error("Modelo se tornou None durante o episódio")
                        break

                    # Obter ação do modelo
                    action, _ = self.model.predict(obs, deterministic=deterministic)

                    # Executar ação
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                    episode_reward += reward

                    # Verificar sucesso
                    if info.get("success", False):
                        episode_success = True
                        self.logger.info(f"Episódio {episode + 1} SUCESSO no step {steps}")
                        break

                    # Condições de término
                    if terminated or truncated:
                        if info.get("termination") == "success":
                            episode_success = True
                            self.logger.info(f"Episódio {episode + 1} SUCESSO por término")
                        else:
                            self.logger.info(f"Episódio {episode + 1} FALHA: {info.get('termination', 'unknown')}")
                        break

                # Calcular tempo do episódio
                episode_time = time.time() - episode_start_time
                total_times.append(episode_time)
                total_rewards.append(episode_reward)

                if episode_success:
                    success_count += 1

                self.logger.info(f"Episódio {episode + 1} finalizado: {episode_time:.2f}s, {steps} steps, sucesso: {episode_success}")

            except Exception as e:
                self.logger.exception(f"Erro no episódio {episode + 1}: {e}")

                # Continuar para o próximo episódio
                total_times.append(0.0)
                total_rewards.append(0.0)
                continue

        # Calcular métricas finais
        if not total_times:
            self.logger.error("NENHUM episódio foi concluído com sucesso")
            return None

        avg_time = np.mean(total_times)
        std_time = np.std(total_times) if len(total_times) > 1 else 0
        success_rate = success_count / num_episodes

        metrics = {
            "avg_time": avg_time,
            "std_time": std_time,
            "success_rate": success_rate,
            "success_count": success_count,
            "total_times": total_times,
            "total_rewards": total_rewards,
            "num_episodes": num_episodes,
        }

        self.logger.info(f"=== AVALIAÇÃO FINALIZADA ===")
        self.logger.info(f"Sucessos: {success_count}/{num_episodes} ({success_rate*100:.1f}%)")
        self.logger.info(f"Tempo médio: {avg_time:.2f}s")
        self.logger.info(f"Desvio padrão: {std_time:.2f}s")

        return metrics


class DynamicPolicyGradientCallback(BaseCallback):
    """Callback para Dynamic Policy Gradient com pesos adaptativos"""

    def __init__(self, logger, num_components=4, verbose=0):
        super().__init__(verbose)
        self.custom_logger = logger
        self.num_components = num_components
        self.reward_weights = np.ones(num_components)  # Pesos iniciais iguais
        self.reward_history = []  # Histórico de recompensas por componente
        self.episode_component_rewards = []  # Recompensas do episódio atual
        self.update_frequency = 10  # Atualizar pesos a cada 10 episódios
        self.episode_count = 0

    def _on_step(self) -> bool:
        try:
            # Coletar informações do ambiente para análise de componentes
            infos = self.locals.get("infos", [])
            if infos and isinstance(infos[0], dict):
                # Extrair componentes de recompensa se disponíveis
                if "reward_components" in infos[0]:
                    component_rewards = infos[0]["reward_components"]
                    self.episode_component_rewards.append(component_rewards)

            # Verificar se há sinal de saída
            if infos and any(isinstance(info, dict) and info.get("exit", False) for info in infos):
                return False

        except Exception as e:
            self.custom_logger.warning(f"Erro no DPG callback: {e}")

        return True

    def _on_rollout_end(self) -> None:
        """Chamado no final de cada rollout - atualiza pesos dinâmicos"""
        if not self.episode_component_rewards:
            return

        # Calcular médias do episódio para cada componente
        episode_avg = np.mean(self.episode_component_rewards, axis=0)
        self.reward_history.append(episode_avg)

        # Manter apenas histórico recente
        if len(self.reward_history) > 50:
            self.reward_history.pop(0)

        self.episode_count += 1
        self.episode_component_rewards = []  # Reset para próximo episódio

        # Atualizar pesos dinamicamente a cada N episódios
        if self.episode_count % self.update_frequency == 0 and len(self.reward_history) >= 10:
            self._update_dynamic_weights()

    def _update_dynamic_weights(self):
        """Atualiza pesos baseado na variância e média das recompensas"""
        rewards_array = np.array(self.reward_history)

        # Calcular estatísticas
        means = np.mean(rewards_array, axis=0)
        variances = np.var(rewards_array, axis=0)

        # Fórmula do paper: priorizar componentes com alta média + alta variância
        # Componentes instáveis (alta variância) precisam de mais atenção
        stability_scores = means + np.exp(np.clip(variances, 0, 2))  # Clip para evitar overflow

        # Normalizar e aplicar suavização
        new_weights = stability_scores / np.sum(stability_scores)

        # Suavizar transição (evitar mudanças bruscas)
        self.reward_weights = 0.7 * self.reward_weights + 0.3 * new_weights

        # Garantir pesos mínimos (nenhum componente é completamente ignorado)
        self.reward_weights = np.maximum(self.reward_weights, 0.1)

        # Renormalizar
        self.reward_weights = self.reward_weights / np.sum(self.reward_weights)

        self.custom_logger.info(f"DPG - Pesos atualizados: {self.reward_weights}")
        self.custom_logger.info(f"DPG - Médias: {means}, Variâncias: {variances}")

    def get_current_weights(self):
        """Retorna os pesos atuais para uso externo"""
        return self.reward_weights.copy()


class EnhancedAgent(Agent):
    """Agente com Dynamic Policy Gradient"""

    def __init__(self, logger, env=None, model_path=None, algorithm="PPO", device="cpu", initial_episode=0):
        super().__init__(logger, env, model_path, algorithm, device, initial_episode)
        self.dpg_callback = None
        self.reward_components_enabled = False

    def enable_dpg(self, num_components=4):
        """Ativa o Dynamic Policy Gradient"""
        self.reward_components_enabled = True
        self.dpg_callback = DynamicPolicyGradientCallback(self.logger, num_components)
        self.logger.info(f"DPG ativado com {num_components} componentes")

    def learn(self, total_timesteps, callback=None, reset_num_timesteps=True, progress_bar=False):
        """Aprendizado com suporte a DPG"""

        # Combinar callbacks se DPG estiver ativo
        if self.reward_components_enabled and self.dpg_callback:
            callbacks = [self.dpg_callback]
            if callback:
                if isinstance(callback, list):
                    callbacks.extend(callback)
                else:
                    callbacks.append(callback)
        else:
            callbacks = callback

        # Chamar aprendizado original com callbacks combinados
        return self.model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar)

    def get_dpg_weights(self):
        """Retorna os pesos atuais do DPG"""
        if self.dpg_callback:
            return self.dpg_callback.get_current_weights()
        return None
