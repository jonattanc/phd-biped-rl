# agent.py
import random
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
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
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            # Verificar se há informações de ambiente
            infos = self.locals.get("infos", [])

            # Se houver infos e alguma indicar saída, parar
            if infos and any(isinstance(info, dict) and info.get("exit", False) for info in infos):
                return False

        except Exception as e:
            # Em caso de erro, continuar o treinamento
            pass

        return True


class Agent:
    def __init__(self, logger, env=None, model_path=None, algorithm="PPO", device="cpu", initial_episode=0):
        self.logger = logger
        self.model = None
        self.algorithm = algorithm
        self.env = env
        self.action_dim = 0
        self.initial_episode = initial_episode

        if env is not None:
            # Criar ambiente vetorizado
            self.env = DummyVecEnv([lambda: env])
            self.action_dim = env.action_dim
            self.model = self._create_model(algorithm, device)

        elif model_path is not None:
            self._load_model(model_path)

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
            return TD3(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=2e-4,
                buffer_size=int(1e6),
                learning_starts=1e4,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_delay=2,
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
        except:
            try:
                self.model = TD3.load(model_path)
                self.algorithm = "TD3"
                if hasattr(self.model, "action_space") and self.model.action_space is not None:
                    self.action_dim = self.model.action_space.shape[0]
                self.logger.info(f"Modelo TD3 carregado: {model_path}")
            except:
                # Tentar carregar como FastTD3
                self.model = FastTD3.load(model_path)
                self.algorithm = "FastTD3"
                if hasattr(self.model, "action_space") and self.model.action_space is not None:
                    self.action_dim = self.model.action_space.shape[0]
                self.logger.info(f"Modelo FastTD3 carregado: {model_path}")

    def set_env(self, env):
        """Configura o ambiente para um modelo carregado"""
        if self.model is not None and env is not None:
            # Criar ambiente vetorizado
            vec_env = DummyVecEnv([lambda: env])

            # Configurar o ambiente no modelo
            self.model.set_env(vec_env)
            self.env = vec_env
            self.action_dim = env.action_dim
            self.logger.info(f"Ambiente configurado para modelo {self.algorithm}")
        else:
            self.logger.warning("Não foi possível configurar ambiente: modelo ou ambiente não disponível")

    def train(self, total_timesteps=100_000):
        """Treina o agente."""
        self.logger.info(f"Executando agent.train por {total_timesteps} timesteps")

        if self.model is not None:
            # Verificar se o ambiente está configurado
            if self.model.get_env() is None:
                self.logger.error("Ambiente não configurado para o modelo!")
                raise ValueError("O ambiente deve ser configurado antes do treinamento. Chame set_env() primeiro.")

            callback = TrainingCallback()
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
        else:
            raise ValueError("Modelo não foi inicializado.")

    def get_action(self, obs=None):
        """Obtém uma ação do modelo ou, se não houver modelo, retorna uma ação aleatória. Não é usada automaticamente por stable_baselines3"""
        if self.model is None or obs is None:
            # Fallback para ação aleatória
            return [random.uniform(-10, 10) for _ in range(self.action_dim)]

        action, _ = self.model.predict(obs, deterministic=False)
        return action.flatten()  # Garante que é um array 1D

    def evaluate(self, env, num_episodes=20, deterministic=True):
        """
        Avalia o agente treinado em um ambiente.
        Retorna métricas completas incluindo contagem de sucessos.
        """
        self.logger.info("Executando agent.evaluate")
        self.logger.info(f"Modo determinístico: {deterministic}")

        if self.model is None:
            raise ValueError("Nenhum modelo treinado carregado para avaliação.")

        total_times = []
        success_count = 0
        total_rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            episode_reward = 0
            episode_success = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, _, info = env.step(action)
                steps += 1
                episode_reward += reward

                # Verificar sucesso
                if info.get("success", False) or info.get("termination") == "success":
                    episode_success = True
                    success_count += 1
                    break

            episode_time = steps * (1 / 240.0)
            total_times.append(episode_time)
            total_rewards.append(episode_reward)

        # Calcular métricas
        avg_time = np.mean(total_times) if total_times else 0
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

        return metrics
