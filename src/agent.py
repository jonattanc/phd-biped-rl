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
    """Buffer de experiências adaptativo com sistema de fases"""
    
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.current_phase = 1
        self.episode_count = 0
        
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
        """Adiciona experiência ao buffer"""
        quality = self._calculate_quality(reward)
        
        # Aplicar filtro de qualidade baseado na fase
        if self.current_phase == 2 and quality < 0.1:  # Fase 2: filtra ruins
            return
        elif self.current_phase == 3 and quality < 0.2:  # Fase 3: filtro rigoroso
            return
            
        # Garantir que todos os elementos sejam arrays numpy
        obs = np.array(obs, dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.float32(reward)
        done = np.bool_(done)
        
        experience = (obs, action, reward, next_obs, done, quality, self.episode_count)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Amostragem balanceada baseada na fase atual"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        # Amostragem estratificada simples
        if self.current_phase == 1:
            # Fase 1: amostragem aleatória
            return random.sample(list(self.buffer), batch_size)
        else:
            # Fases 2 e 3: priorizar experiências de alta qualidade
            high_quality = [exp for exp in self.buffer if exp[5] > 0.7]
            medium_quality = [exp for exp in self.buffer if 0.3 <= exp[5] <= 0.7]
            low_quality = [exp for exp in self.buffer if exp[5] < 0.3]
            
            # Balancear amostras por qualidade
            if self.current_phase == 2:
                high_count = min(int(batch_size * 0.5), len(high_quality))
                medium_count = min(int(batch_size * 0.3), len(medium_quality))
                low_count = batch_size - high_count - medium_count
            else:  # Fase 3
                high_count = min(int(batch_size * 0.7), len(high_quality))
                medium_count = min(int(batch_size * 0.2), len(medium_quality))
                low_count = batch_size - high_count - medium_count
            
            samples = []
            if high_count > 0 and high_quality:
                samples.extend(random.sample(high_quality, high_count))
            if medium_count > 0 and medium_quality:
                samples.extend(random.sample(medium_quality, medium_count))
            if low_count > 0 and low_quality:
                samples.extend(random.sample(low_quality, low_count))
                
            # Completar com amostras aleatórias se necessário
            if len(samples) < batch_size:
                additional = random.sample(list(self.buffer), batch_size - len(samples))
                samples.extend(additional)
                
            return samples
    
    def update_episode(self, episode_count: int):
        """Atualiza contador de episódio"""
        self.episode_count = episode_count
        
        # Limpeza periódica simples (a cada 50 episódios)
        if episode_count % 50 == 0 and len(self.buffer) > self.capacity * 0.8:
            # Manter apenas as melhores 80% das experiências
            sorted_buffer = sorted(self.buffer, key=lambda x: x[5], reverse=True)
            keep_count = int(self.capacity * 0.8)
            self.buffer = deque(sorted_buffer[:keep_count], maxlen=self.capacity)
    
    def get_status(self):
        """Retorna estatísticas do buffer"""
        if not self.buffer:
            return {"size": 0, "phase": self.current_phase, "avg_quality": 0}
        
        qualities = [exp[5] for exp in self.buffer]
        avg_quality = np.mean(qualities)
        
        return {
            "phase": self.current_phase,
            "size": len(self.buffer),
            "capacity": self.capacity,
            "avg_quality": round(avg_quality, 3),
            "utilization": round(len(self.buffer) / self.capacity * 100, 1)
        }
    
    def __len__(self):
        return len(self.buffer)
    

class FastTD3(TD3):
    """
    Fast TD3 com DPG integrado - buffer inteligente e sistema de fases
    """
    
    def __init__(self, policy, env, custom_logger=None, **kwargs):
        kwargs.pop('action_dim', None)
        super().__init__(policy, env, **kwargs)
        
        # Armazenar o logger com um nome diferente para evitar conflito
        self.custom_logger = custom_logger
        
        # Usar nosso buffer adaptativo em vez do buffer padrão
        buffer_size = kwargs.get('buffer_size', 100000)
        self.replay_buffer = AdaptiveBuffer(capacity=buffer_size)
        self.phase_manager = PhaseManager()
    
    def _store_transition(self, replay_buffer, action, new_obs, reward, done, infos):
        """Armazena transição no nosso buffer adaptativo"""
        # Ignorar infos para manter compatibilidade com nosso buffer
        self.replay_buffer.add(self._last_obs, new_obs, action, reward, done)
    
    def train(self, gradient_steps, batch_size=100):
        """
        Treinamento com buffer adaptativo
        """
        # Sincronizar fase atual com o buffer
        current_phase = self.phase_manager.current_phase
        self.replay_buffer.set_phase(current_phase)
        
        for gradient_step in range(gradient_steps):
            # Amostrar do nosso buffer adaptativo
            if len(self.replay_buffer) < batch_size:
                break
                
            replay_data = self.replay_buffer.sample(batch_size)
            
            if not replay_data:
                continue
                
            try:
                # Converter para arrays compatíveis com TD3 - garantir formato consistente
                observations = np.stack([exp[0] for exp in replay_data])
                actions = np.stack([exp[1] for exp in replay_data])
                rewards = np.array([exp[2] for exp in replay_data], dtype=np.float32)
                next_observations = np.stack([exp[3] for exp in replay_data])
                dones = np.array([exp[4] for exp in replay_data], dtype=np.bool_)
                
                # Verificar se todas as observações têm a mesma dimensão
                if observations.shape[1] != self.observation_space.shape[0]:
                    if hasattr(self, 'custom_logger') and self.custom_logger:
                        self.custom_logger.warn(f"Dimensão inconsistente: esperado {self.observation_space.shape[0]}, obtido {observations.shape[1]}")
                    continue
                    
                # Chamar implementação original do TD3
                self._train_step(observations, actions, rewards, next_observations, dones)
                
            except Exception as e:
                if hasattr(self, 'custom_logger') and self.custom_logger:
                    self.custom_logger.warn(f"Erro ao processar batch de treinamento: {e}")
                continue
    
    def update_phase_metrics(self, episode_metrics):
        """Atualiza métricas e gerencia transições de fase"""
        self.phase_manager.update_phase_metrics(episode_metrics)
        
        # Atualizar contador de episódio no buffer
        self.replay_buffer.update_episode(episode_metrics.get('episode_count', 0))
        
        # Verificar transição de fase
        if self.phase_manager.should_transition_phase():
            if self.phase_manager.transition_to_next_phase():
                new_phase = self.phase_manager.current_phase
                self.replay_buffer.set_phase(new_phase)
                return True
        return False
    
    def get_phase_info(self):
        """Retorna informações da fase atual"""
        return self.phase_manager.get_phase_info()

    def get_phase_weight_adjustments(self):
        """Retorna ajustes de peso específicos por componente"""
        return self.phase_manager.get_phase_weight_adjustments()
    
    def get_buffer_status(self):
        """Retorna status do buffer adaptativo"""
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
