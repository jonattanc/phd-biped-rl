# agent.py
import random
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
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log="./logs/tensorboard/"
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