# agent.py
import random
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import torch

from utils import PhaseManager


class IntelligentBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.categories = {
            'excellent': [],    # Recompensa > 5.0
            'good': [],         # Recompensa 1.0 - 5.0  
            'neutral': [],      # Recompensa -1.0 - 1.0
            'poor': []          # Recompensa < -1.0
        }
    
    def add_experience(self, experience):
        """Adiciona experiência categorizada"""
        state, action, reward, next_state, done = experience
        
        # Classificar por qualidade
        if reward > 5.0:
            category = 'excellent'
        elif reward > 1.0:
            category = 'good'
        elif reward > -1.0:
            category = 'neutral'
        else:
            category = 'poor'
        
        # Manter balanceamento
        self.categories[category].append(experience)
        if len(self.categories[category]) > self.capacity // 4:
            self.categories[category].pop(0)
    
    def sample(self, batch_size=128):
        """Amostra balanceada do buffer"""
        samples = []
        for category in self.categories:
            if len(self.categories[category]) > 0:
                # 25% de cada categoria
                n_samples = batch_size // 4
                category_samples = random.sample(
                    self.categories[category], 
                    min(n_samples, len(self.categories[category]))
                )
                samples.extend(category_samples)
        
        # Completar com amostras aleatórias se necessário
        if len(samples) < batch_size:
            all_experiences = [exp for cat in self.categories.values() for exp in cat]
            samples.extend(random.sample(all_experiences, batch_size - len(samples)))
        
        return samples
    
    def __len__(self):
        return sum(len(category) for category in self.categories.values())
    

class FastTD3(TD3):
    """
    Fast TD3 com DPG integrado - buffer inteligente e sistema de fases
    """
    
    def __init__(self, policy, env, action_dim, **kwargs):
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim), 
            sigma=0.1 * np.ones(action_dim)
        )
        super().__init__(
            policy,
            env,
            action_noise=action_noise,
            **kwargs
        )
        
        # Sistema DPG integrado
        self.dpg_buffer = IntelligentBuffer(capacity=50000)
        self.phase_manager = PhaseManager() 

        # Configurações específicas por fase
        self.phase_configs = {
            1: {'learning_rate': 1.0e-4, 'noise_sigma': 0.1},  # Fase 1: padrão
            2: {'learning_rate': 8.0e-5, 'noise_sigma': 0.05}, # Fase 2: mais estável
            3: {'learning_rate': 5.0e-5, 'noise_sigma': 0.02}  # Fase 3: mais preciso
        }
        
    def store_dpg_experience(self, state, action, reward, next_state, done):
        """Armazena experiência no buffer DPG"""
        self.dpg_buffer.add_experience((state, action, reward, next_state, done))
    
    def get_dpg_status(self):
        """Retorna status do DPG integrado"""
        return {
            'phase': self.phase_manager.current_phase,
            'buffer_size': len(self.dpg_buffer),
            'phase_ready': len(self.dpg_buffer) > 1000,
            'phase_info': self.phase_manager.get_phase_info()
        }
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas e gerencia transições de fase"""
        self.phase_manager.update_phase_metrics(episode_metrics)
        
        # Verificar transição de fase
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                self._apply_phase_config()
                return True
        return False
    
    def _apply_phase_config(self):
        """Aplica configurações específicas da fase"""
        phase = self.phase_manager.current_phase
        config = self.phase_configs.get(phase, {})
        
        if 'learning_rate' in config:
            # Atualizar learning rate
            for param_group in self.actor.optimizer.param_groups:
                param_group['lr'] = config['learning_rate']
            for param_group in self.critic.optimizer.param_groups:
                param_group['lr'] = config['learning_rate']
        
        if 'noise_sigma' in config:
            # Atualizar noise
            self.action_noise = NormalActionNoise(
                mean=np.zeros(self.action_space.shape[0]),
                sigma=config['noise_sigma'] * np.ones(self.action_space.shape[0])
            )
    
    def should_transition_phase(self):
        """Verifica transição de fase"""
        return self.phase_manager.should_transition_phase()
    
    def get_phase_multiplier(self):
        """Retorna multiplicador da fase"""
        return self.phase_manager.get_phase_weight_multiplier()
    
    def get_phase_info(self):
        """Retorna informações da fase"""
        return self.phase_manager.get_phase_info()

    def get_phase_weight_adjustments(self):
        """Retorna ajustes de peso específicos por componente"""
        return self.phase_manager.get_phase_weight_adjustments()


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
        self.action_dim = env.action_space.shape[0] if env else None
        self.initial_episode = initial_episode
        self.learning_starts = 10e3
        self.prefill_steps = 100e3
        self.minimum_steps_to_save = self.learning_starts + self.prefill_steps + 1e6

        dummy_env = DummyVecEnv([lambda: env])

        if model_path is None:
            self.env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True)
            self.model = self._create_model(algorithm, device, seed)

        else:
            self.logger.info(f"Loading: {model_path}")
            vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
            self.env = VecNormalize.load(vecnorm_path, dummy_env)
            self.env.training = False
            self.env.norm_reward = False
            self.env.norm_obs = True
            self._load_model(model_path)
            self.model.set_env(self.env)
            self.logger.info(f"self.env.obs_rms.mean[:5]: {self.env.obs_rms.mean[:5]}")
            self.logger.info(f"self.env.obs_rms.var[:5]: {self.env.obs_rms.var[:5]}")

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
                target_policy_noise=0.4,
                target_noise_clip=0.5,
                tensorboard_log="./logs/",
                device=device,
                seed=seed,
            )
        elif algorithm.upper() == "FASTTD3":
            action_dim = self.env.action_space.shape[0]
            return FastTD3(
                "MlpPolicy",
                self.env,
                action_dim=action_dim,
                learning_rate=1.0e-4,
                buffer_size=int(1e6),
                learning_starts=self.learning_starts,
                batch_size=256,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_delay=3,
                target_policy_noise=0.4,
                target_noise_clip=0.5,
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
            self.logger.info(f"self.env.obs_rms.mean[:5]: {self.env.obs_rms.mean[:5]}")
            self.logger.info(f"self.env.obs_rms.var[:5]: {self.env.obs_rms.var[:5]}")
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
                self.logger.info(f"self.model.action_space.shape[0]: {self.model.action_space.shape[0]}")
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
