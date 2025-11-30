# agent.py
import random
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from utils import PhaseManager
from collections import deque
from typing import List, Dict, Any


class AdaptiveBuffer:
    """Buffer de experiências adaptativo"""
    
    def __init__(self, capacity=5000, observation_shape=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.current_phase = 1
        self.episode_count = 0
        self.observation_shape = observation_shape

        # CONFIGURAÇÕES DE LIMPEZA DO DPG FUNCIONAL
        self.cleanup_interval = 200  
        self.min_buffer_size = 1000  
        self.last_cleanup_episode = 0
        
    def set_phase(self, phase: int):
        """Altera a fase do buffer"""
        if 1 <= phase <= 3:
            self.current_phase = phase
    
    def _calculate_quality(self, reward: float) -> float:
        """Calcula qualidade baseada na recompensa"""
        if reward > 50: return 0.9
        elif reward > 20: return 0.8
        elif reward > 10: return 0.7
        elif reward > 5: return 0.6
        elif reward > 0: return 0.5
        elif reward > -5: return 0.4
        elif reward > -10: return 0.3
        elif reward > -20: return 0.2
        else: return 0.1

    def add(self, obs, next_obs, action, reward, done):
        """Adiciona experiência"""
        quality = self._calculate_quality(reward)

        # Filtro de qualidade baseado na fase
        if self.current_phase == 2 and quality < 0.05:
            return
        elif self.current_phase == 3 and quality < 0.1:
            return

        try:
            # CONVERSÃO ROBUSTA
            obs_array = np.array(obs, dtype=np.float32).flatten()
            next_obs_array = np.array(next_obs, dtype=np.float32).flatten()
            action_array = np.array(action, dtype=np.float32).flatten()
            
            # Garantir shapes consistentes
            if self.observation_shape is not None:
                if len(obs_array) != self.observation_shape:
                    obs_array = self._adjust_shape(obs_array, self.observation_shape)
                if len(next_obs_array) != self.observation_shape:
                    next_obs_array = self._adjust_shape(next_obs_array, self.observation_shape)
            else:
                # Definir observation_shape na primeira experiência válida
                if len(obs_array) > 0:
                    self.observation_shape = len(obs_array)

        except Exception as e:
            return  # Silenciosamente descarta experiências com erro

        experience = (obs_array, action_array, float(reward), next_obs_array, bool(done), quality, self.episode_count)
        self.buffer.append(experience)

        # LIMPEZA AUTOMÁTICA QUANDO BUFFER CHEIO
        if len(self.buffer) >= self.capacity:
            self._cleanup_aggressive()
    
    def _adjust_shape(self, array, target_shape):
        """Ajusta shape do array para o target"""
        if len(array) < target_shape:
            # Preencher com zeros
            padded = np.zeros(target_shape, dtype=np.float32)
            padded[:len(array)] = array
            return padded
        elif len(array) > target_shape:
            # Cortar excesso
            return array[:target_shape]
        else:
            return array
        
    def sample(self, batch_size: int):
        """Amostragem balanceada"""
        if len(self.buffer) < batch_size:
            return None

        # Estratégia por fase
        if self.current_phase == 1:
            # Fase 1: 80% aleatório, 20% qualidade
            samples = self._stratified_sample(batch_size, [0.2, 0.3, 0.5])
        elif self.current_phase == 2:
            # Fase 2: 60% alta, 30% média, 10% baixa
            samples = self._stratified_sample(batch_size, [0.6, 0.3, 0.1])
        else:  # Fase 3
            # Fase 3: 80% alta, 15% média, 5% baixa  
            samples = self._stratified_sample(batch_size, [0.8, 0.15, 0.05])

        return self._convert_to_td3_format(samples)
    
    def _stratified_sample(self, batch_size, ratios):
        """Amostragem estratificada por qualidade"""
        high_quality = [exp for exp in self.buffer if exp[5] > 0.6]
        medium_quality = [exp for exp in self.buffer if 0.3 <= exp[5] <= 0.6]
        low_quality = [exp for exp in self.buffer if exp[5] < 0.3]
        
        high_count = min(int(batch_size * ratios[0]), len(high_quality))
        medium_count = min(int(batch_size * ratios[1]), len(medium_quality))
        low_count = batch_size - high_count - medium_count
        
        samples = []
        if high_count > 0:
            samples.extend(random.sample(high_quality, high_count))
        if medium_count > 0:
            samples.extend(random.sample(medium_quality, medium_count))
        if low_count > 0 and len(low_quality) >= low_count:
            samples.extend(random.sample(low_quality, low_count))
        
        # Completar se necessário (fallback para aleatório)
        if len(samples) < batch_size:
            needed = batch_size - len(samples)
            additional = random.sample(self.buffer, needed)
            samples.extend(additional)
            
        return samples
    
    def _convert_to_td3_format(self, samples):
        """Conversão para formato TD3"""
        if not samples:
            return None
            
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for exp in samples:
            obs, action, reward, next_obs, done, quality, episode_count = exp
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done)

        try:
            observations_array = np.array(observations, dtype=np.float32)
            actions_array = np.array(actions, dtype=np.float32)
            rewards_array = np.array(rewards, dtype=np.float32).reshape(-1, 1)
            next_observations_array = np.array(next_observations, dtype=np.float32)
            dones_array = np.array(dones, dtype=np.bool_).reshape(-1, 1)
            
            return (observations_array, actions_array, rewards_array, next_observations_array, dones_array)
        except Exception as e:
            return None
    
    def update_episode(self, episode_count: int):
        """Limpeza periódica"""
        self.episode_count = episode_count
        
        # Limpeza a cada 50 episódios se buffer > 80% capacidade
        if episode_count % 50 == 0 and len(self.buffer) > self.capacity * 0.8:
            self._cleanup_low_quality(1000)
    
    def _cleanup_low_quality(self, remove_count: int):
        """Remove as piores experiências"""
        if len(self.buffer) <= remove_count + self.min_buffer_size:
            return
            
        qualities = []
        for i, exp in enumerate(self.buffer):
            quality = exp[5]  
            qualities.append((i, quality))
        
        qualities.sort(key=lambda x: x[1])
        indices_to_remove = {i for i, qual in qualities[:remove_count]}
        
        new_buffer = deque(maxlen=self.capacity)
        for i, exp in enumerate(self.buffer):
            if i not in indices_to_remove:
                new_buffer.append(exp)
                
        self.buffer = new_buffer
    
    def _cleanup_aggressive(self):
        """Limpeza agressiva"""
        current_size = len(self.buffer)
        
        if current_size <= self.min_buffer_size:
            return  
            
        if current_size <= 2000:
            remove_count = 200   
        elif current_size <= 4000:
            remove_count = 500     
        else:
            remove_count = 1000  
            
        self._cleanup_low_quality(remove_count)
    
    def get_status(self):
        """Status do buffer"""
        if not self.buffer:
            return {"size": 0, "phase": self.current_phase, "avg_quality": 0}
        
        qualities = [exp[5] for exp in self.buffer]
        avg_quality = np.mean(qualities) if qualities else 0
        
        return {
            "phase": self.current_phase,
            "size": len(self.buffer),
            "capacity": self.capacity,
            "avg_quality": round(avg_quality, 3),
            "utilization": round(len(self.buffer) / self.capacity * 100, 1),
            "cleanup_interval": self.cleanup_interval,
            "min_buffer_size": self.min_buffer_size,
            "last_cleanup": self.last_cleanup_episode
        }
    
    def __len__(self):
        return len(self.buffer)
    

class FastTD3(TD3):
    """
    Fast TD3 com buffer adaptativo funcional
    """
    
    def __init__(self, policy, env, custom_logger=None, **kwargs):
        kwargs.pop('action_dim', None)
        super().__init__(policy, env, **kwargs)

        self.custom_logger = custom_logger

        # Obter observation_shape do environment
        try:
            if hasattr(env.observation_space, 'shape'):
                observation_shape = env.observation_space.shape[0]
            else:
                # Fallback: tentar obter do env
                observation_shape = getattr(env, 'obs_shape', [None])[0] if hasattr(env, 'obs_shape') else None
        except:
            observation_shape = None

        buffer_size = kwargs.get('buffer_size', 100000)

        # Buffer adaptativo funcional
        self.replay_buffer = AdaptiveBuffer(
            capacity=buffer_size, 
            observation_shape=observation_shape
        )
        self.phase_manager = PhaseManager()

        if self.custom_logger:
            self.custom_logger.info(f"FastTD3 com buffer adaptativo - observation_shape: {observation_shape}")
    
    def _store_transition(self, replay_buffer, action, new_obs, reward, done, infos):
        """Armazena transição"""
        self.replay_buffer.add(self._last_obs, new_obs, action, reward, done)
    
    def train(self, gradient_steps, batch_size=100):
        """
        Treinamento adaptado para nosso buffer
        """
        # Sincronizar fase
        current_phase = self.phase_manager.current_phase
        self.replay_buffer.set_phase(current_phase)

        successful_steps = 0

        for gradient_step in range(gradient_steps):
            # Amostrar do nosso buffer
            batch_data = self.replay_buffer.sample(batch_size)
            
            if batch_data is None:
                continue

            try:
                obs, actions, rewards, next_obs, dones = batch_data
                
                # Verificação final de shapes
                if (obs.shape[0] == batch_size and actions.shape[0] == batch_size and
                    rewards.shape[0] == batch_size and next_obs.shape[0] == batch_size):
                    
                    # Usar método interno do TD3 para treinamento
                    self._train_step(obs, actions, rewards, next_obs, dones)
                    successful_steps += 1
                    
            except Exception as e:
                if self.custom_logger:
                    self.custom_logger.debug(f"Erro no batch: {e}")
                continue

        return successful_steps
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas de fase"""
        self.phase_manager.update_phase_metrics(episode_metrics)
        self.replay_buffer.update_episode(episode_metrics.get('episode_count', 0))
        
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                new_phase = self.phase_manager.current_phase
                self.replay_buffer.set_phase(new_phase)
                return True
        return False
    
    def get_phase_info(self):
        return self.phase_manager.get_phase_info()

    def get_phase_weight_adjustments(self):
        return self.phase_manager.get_phase_weight_adjustments()
    
    def get_buffer_status(self):
        return self.replay_buffer.get_status()


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
            return FastTD3(
                "MlpPolicy",
                self.env,
                custom_logger=self.logger,
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
                if hasattr(self.model, 'custom_logger'):
                    self.model.custom_logger = self.logger
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
