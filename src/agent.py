# agent.py
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class Agent:
    def __init__(self, env=None, model_path=None):
        self.revolute_indices = []
        self.len_revolute_indices = 0
        self.model = None

        if env is not None:
            # Criar ambiente vetorizado
            self.env = DummyVecEnv([lambda: env])
            # Criar modelo PPO
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
                tensorboard_log="./logs/"
            )
        elif model_path is not None:
            # Carregar modelo treinado
            self.model = PPO.load(model_path)

    def set_revolute_indices(self, revolute_indices):
        self.revolute_indices = revolute_indices
        self.len_revolute_indices = len(revolute_indices)

    def train(self, total_timesteps=100_000):
        """Treina o agente."""
        if self.model is not None:
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        else:
            raise ValueError("Modelo PPO não foi inicializado.")

    def get_action(self, obs=None):
        """Obtém uma ação do modelo PPO ou, se não houver modelo, retorna uma ação aleatória."""
        if self.model is not None and obs is not None:
            action, _ = self.model.predict(obs, deterministic=False)
            return action.flatten()  # Garante que é um array 1D
        else:
            # Fallback para ação aleatória (útil para testes iniciais)
            return [random.uniform(-10, 10) for _ in range(self.len_revolute_indices)]
    
    def evaluate(self, env, num_episodes=2):
        """
        Avalia o agente treinado em um ambiente.
        Executa `num_episodes` episódios com ações determinísticas e retorna métricas estatísticas.
        """
        if self.model is None:
            raise ValueError("Nenhum modelo PPO treinado carregado para avaliação.")

        total_times = []
        success_count = 0

        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            steps = 0

            while not done:
                # Ação determinística (sem exploração)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                steps += 1

                # Verifica se o episódio terminou por sucesso
                if info.get("success", False):
                    success_count += 1
                    break

            # Calcula duração do episódio em segundos
            episode_time = steps * (1 / 240.0)  # PyBullet usa 240 Hz
            total_times.append(episode_time)

        # Calcula métricas
        avg_time = np.mean(total_times)
        std_time = np.std(total_times)
        success_rate = success_count / num_episodes

        metrics = {
            "avg_time": avg_time,
            "std_time": std_time,
            "success_rate": success_rate,
            "total_times": total_times  # Para análise detalhada, se necessário
        }

        return metrics