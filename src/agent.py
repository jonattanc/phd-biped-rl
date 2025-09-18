# agent.py
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class AbortOnSignalCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")

        if infos and any(info.get("exit", False) for info in infos):
            return False  # returning False stops training

        return True


class Agent:
    def __init__(self, sim_env, model_path=None):
        if model_path is not None:
            self.model = PPO.load(model_path)

        else:
            self.model = PPO(
                "MlpPolicy",
                sim_env,
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

    def train(self, total_timesteps=100_000):
        """Treina o agente."""
        if self.model is not None:
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=AbortOnSignalCallback())

        else:
            raise ValueError("Modelo não foi inicializado.")

    def get_action(self, obs=None):
        """Obtém uma ação do modelo ou, se não houver modelo, retorna uma ação aleatória."""
        if self.model is not None and obs is not None:
            action, _ = self.model.predict(obs, deterministic=False)
            return action.flatten()  # Garante que é um array 1D
        else:
            # Fallback para ação aleatória (útil para testes iniciais)
            return [random.uniform(-10, 10) for _ in range(self.len_revolute_indices)]

    def evaluate(self, env, num_episodes=20):
        """
        Avalia o agente treinado em um ambiente.
        Retorna métricas completas incluindo contagem de sucessos.
        """
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
