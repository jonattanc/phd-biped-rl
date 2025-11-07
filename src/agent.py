# agent.py
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
    def __init__(self, logger, env=None, model_path=None, algorithm="PPO", device="cpu", initial_episode=0, seed=42):
        self.logger = logger
        self.model = None
        self.algorithm = algorithm
        self.env = env
        self.action_dim = env.action_dim
        self.initial_episode = initial_episode
        self.learning_starts = 10e3
        self.prefill_steps = 100e3
        self.minimum_steps_to_save = self.learning_starts + self.prefill_steps + 100e3

        dummy_env = DummyVecEnv([lambda: env])

        if model_path is None:
            self.env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True)
            self.model = self._create_model(algorithm, device, seed)

        else:
            vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
            self.env = VecNormalize.load(vecnorm_path, dummy_env)
            self.env.training = False
            self.env.norm_reward = False
            self._load_model(model_path)
            self.model.set_env(self.env)

    def _create_model(self, algorithm, device="cpu", seed=42):
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
                learning_starts=self.learning_starts,
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
                seed=seed,
            )
        elif algorithm.upper() == "FASTTD3":
            return FastTD3(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,  # Aumentar LR
                buffer_size=50000,  # Buffer menor para aprendizado mais rápido
                learning_starts=2000,  # Começar a aprender mais cedo
                batch_size=256,  # Batch menor
                gamma=0.98,  # Desconto menor = foco em recompensas imediatas
                train_freq=(4, "step"),  # Treinar com menos frequência
                gradient_steps=1,  # Mais atualizações quando treinar
                policy_delay=2,
                target_policy_noise=0.1,
                target_noise_clip=0.1,
                tensorboard_log="./logs/",
                device=device,
            )
        else:
            raise ValueError(f"Algoritmo {algorithm} não suportado. Use 'PPO', 'TD3' ou 'FastTD3'")

    def save_model(self, model_path):
        """Salva o modelo treinado"""
        if self.model is not None:
            self.model.save(model_path)
            vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
            self.env.save(vecnorm_path)

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

    def set_agent(self, agent):
        """Configura o agente no ambiente"""
        self.agent = agent
        self.logger.info(f"Agente {agent.algorithm} configurado na simulação")

    def learn(self, total_timesteps, reset_num_timesteps=False, callback=None):
        """
        Método learn para compatibilidade com DPG
        """
        # Apenas delega para o modelo
        return self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps, callback=callback)
