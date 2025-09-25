# agent.py
import random
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    def __init__(self, data_callback=None, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.data_callback = data_callback
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            episode_info = self.model.ep_info_buffer[0]
            if "r" in episode_info and "l" in episode_info:
                episode_reward = episode_info["r"]
                episode_length = episode_info["l"]

                # Chamar callback quando o episódio terminar
                if self.data_callback and episode_reward is not None:
                    self.data_callback.on_episode_end(
                        {"reward": episode_reward, "time": episode_length * (1 / 240.0), "distance": episode_info.get("distance", 0), "success": episode_info.get("success", False)}
                    )
                    self.episode_count += 1

                    # Limpar buffer após processamento
                    self.model.ep_info_buffer = []

        infos = self.locals.get("infos")

        if infos and any(info.get("exit", False) for info in infos):
            return False  # returning False stops training

        return True


class Agent:
    def __init__(self, logger, env=None, model_path=None, algorithm="PPO", data_callback=None):
        self.logger = logger
        self.revolute_indices = []
        self.len_revolute_indices = 0
        self.model = None
        self.algorithm = algorithm
        self.env = env
        self.action_dim = 0
        self.data_callback = data_callback

        if env is not None:
            # Criar ambiente vetorizado
            self.env = DummyVecEnv([lambda: env])
            self.action_dim = env.action_space.shape[0]
            self._create_model(algorithm)

        elif model_path is not None:
            self.model = PPO.load(model_path)

    def _create_model(self, algorithm):
        # Criar modelo baseado no algoritmo selecionado
        if algorithm.upper() == "PPO":
            self.model = PPO(
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
            )
        elif algorithm.upper() == "TD3":
            self.model = TD3(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=10000,
                batch_size=128,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "episode"),
                gradient_steps=-1,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                tensorboard_log="./logs/",
            )
        else:
            raise ValueError(f"Algoritmo {algorithm} não suportado. Use 'PPO' ou 'TD3'")

    def _load_model(self, model_path):
        # Carrega modelo treinado detectando automaticamente o tipo
        try:
            # Tentar carregar como PPO primeiro
            self.model = PPO.load(model_path)
            self.algorithm = "PPO"
            if hasattr(self.model, "action_space") and self.model.action_space is not None:
                self.action_dim = self.model.action_space.shape[0]
            print(f"Modelo PPO carregado: {model_path}")
        except:
            try:
                self.model = TD3.load(model_path)
                self.algorithm = "TD3"
                if hasattr(self.model, "action_space") and self.model.action_space is not None:
                    self.action_dim = self.model.action_space.shape[0]
                print(f"Modelo TD3 carregado: {model_path}")
            except Exception as e:
                raise ValueError(f"Erro ao carregar modelo {model_path}: {e}")

    def set_revolute_indices(self, revolute_indices):
        self.revolute_indices = revolute_indices
        self.len_revolute_indices = len(revolute_indices)

    def train(self, total_timesteps=100_000):
        """Treina o agente."""
        self.logger.info("Executando agent.train")

        if self.model is not None:
            callback = TrainingCallback(data_callback=self.data_callback)
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
        else:
            raise ValueError("Modelo não foi inicializado.")

    def get_action(self, obs=None):
        """Obtém uma ação do modelo ou, se não houver modelo, retorna uma ação aleatória."""
        if self.model is not None and obs is not None:
            action, _ = self.model.predict(obs, deterministic=False)
            return action.flatten()  # Garante que é um array 1D
        else:
            # Fallback para ação aleatória
            return [random.uniform(-10, 10) for _ in range(self.len_revolute_indices)]

    def evaluate(self, env, num_episodes=20):
        """
        Avalia o agente treinado em um ambiente.
        Retorna métricas completas incluindo contagem de sucessos.
        """
        self.logger.info("Executando agent.evaluate")

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
                action, _ = self.model.predict(obs, deterministic=True)
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
