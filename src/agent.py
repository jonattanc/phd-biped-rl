# agent.py
import math
import random
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import torch
import torch.nn.functional as F
from utils import PhaseManager
from collections import deque
from typing import List, Dict, Any

    
class SimpleFastBuffer:
    def __init__(self, capacity=10000, observation_shape=None, action_shape=None):
        self.capacity = capacity
        self.target_size_after_cleanup = 8000  
        self.buffer = deque(maxlen=capacity)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.current_phase = 1
        self.min_buffer_size = 1000  
        self.episode_count = 0
        self.cleanup_frequency = 500  
        self.min_quality_for_phase = {1: 0.1, 2: 0.4, 3: 0.8}
        self.prefill_mode = False
        
    def add(self, obs, next_obs, action, reward, done, infos=None):
        """Apenas experiências de ALTA QUALIDADE são aceitas"""
        try:
            obs_array = np.asarray(obs, dtype=np.float32).flatten()
            next_obs_array = np.asarray(next_obs, dtype=np.float32).flatten()
            action_array = np.asarray(action, dtype=np.float32).flatten()

            # Ajustar shapes
            obs_array = self._adjust_shape(obs_array, self.observation_shape)
            next_obs_array = self._adjust_shape(next_obs_array, self.observation_shape)
            action_array = self._adjust_shape(action_array, self.action_shape)

            # CRITÉRIOS DE QUALIDADE MUITO MAIS RESTRITIVOS
            quality = self._calculate_elite_quality(reward, done, infos)
            
            # Apenas aceitar experiências de alta qualidade
            min_quality = self.min_quality_for_phase.get(self.current_phase, 0.5)
            
            if quality < min_quality:
                return False

            experience = {
                'obs': obs_array,
                'next_obs': next_obs_array, 
                'action': action_array,
                'reward': float(reward),
                'done': bool(done),
                'quality': quality
            }

            # Estratégia de substituição agressiva
            if len(self.buffer) >= self.capacity:
                self._replace_worst_experience(experience)
            else:
                self.buffer.append(experience)

            return True
                
        except Exception as e:
            return False
    
    def _calculate_elite_quality(self, reward, done, infos=None):
        """Critério de qualidade RESTRITIVO - apenas experiências elite"""
        if done and reward < -800: return 0.01
        if done and reward < -100: return 0.3
            
        # COMPORTAMENTOS EXCELENTES
        if reward > 200:   return 0.99
        if reward > 100:   return 0.95
        if reward > 50:    return 0.85
        if reward > 20:    return 0.70
        if reward > 10:    return 0.60
        if reward > 5:     return 0.50
        if reward > 0:     return 0.40
        return 0.15
    
    def sample(self, batch_size: int):
        """Amostragem APENAS das melhores experiências"""
        if len(self.buffer) < batch_size:
            return None

        # Amostrar apenas do top 30% do buffer
        elite_size = max(batch_size, int(len(self.buffer) * 0.5))
        elite_pool = sorted(self.buffer, key=lambda x: x['quality'], reverse=True)[:elite_size]
        
        qualities = [exp['quality'] for exp in elite_pool]
        probs = np.array(qualities) / sum(qualities)

        indices = np.random.choice(len(elite_pool), size=batch_size, p=probs, replace=False)
        selected = [elite_pool[i] for i in indices]

        return self._convert_to_arrays(selected)
    
    def _adjust_shape(self, array, target_shape):
        """Ajuste de shape confiável"""
        if len(array) < target_shape:
            return np.pad(array, (0, target_shape - len(array)), 'constant')
        elif len(array) > target_shape:
            return array[:target_shape]
        return array
    
    def _convert_to_arrays(self, batch):
        """Converts batch of experiences to numpy arrays for training"""
        obs = np.array([exp['obs'] for exp in batch])
        next_obs = np.array([exp['next_obs'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        return (obs, actions, rewards, next_obs, dones)
    
    def set_phase(self, phase):
        self.current_phase = phase

    def update_episode(self, episode_count: int):
        """Limpeza agressiva a cada X episódios"""
        self.episode_count = episode_count

        if episode_count % self.cleanup_frequency == 0:
            self._aggressive_cleanup()

    def _aggressive_cleanup(self):
        """Mantém apenas as melhores experiências próximas a target_size_after_cleanup"""
        if len(self.buffer) <= self.target_size_after_cleanup:
            return

        # Ordenar por qualidade e manter as melhores
        sorted_buffer = sorted(self.buffer, key=lambda x: x['quality'], reverse=True)
        keep_count = min(
            self.target_size_after_cleanup + random.randint(-100, 100),  
            len(sorted_buffer)
        )
        
        self.buffer = deque(
            sorted_buffer[:keep_count],
            maxlen=self.capacity
        )

    def _replace_worst_experience(self, new_experience):
        """Substitui a pior experiência se a nova for melhor"""
        if not self.buffer:
            self.buffer.append(new_experience)
            return

        # Encontrar a pior experiência
        worst_quality = float('inf')
        worst_index = -1
        
        for i, exp in enumerate(self.buffer):
            if exp['quality'] < worst_quality:
                worst_quality = exp['quality']
                worst_index = i

        # Substituir apenas se nova experiência for melhor
        if new_experience['quality'] > worst_quality:
            self.buffer[worst_index] = new_experience

    def __len__(self):
        return len(self.buffer)
    

class FastTD3(TD3):
    def __init__(self, policy, env, custom_logger=None, **kwargs):
        kwargs.pop('action_dim', None)
        kwargs['learning_starts'] = 0
        
        super().__init__(policy, env, **kwargs)
        
        # Buffer elite
        observation_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]
        
        self.replay_buffer = SimpleFastBuffer(
            capacity=kwargs.get('buffer_size', 5000),
            observation_shape=observation_shape,
            action_shape=action_shape
        )
        
        self.custom_logger = custom_logger
        self.phase_manager = PhaseManager()

    def _store_transition(self, replay_buffer, action, new_obs, reward, done, infos):
        """Armazena transição"""
        obs = self._last_obs
        if hasattr(obs, 'flatten'):
            obs = obs.flatten()
        if hasattr(new_obs, 'flatten'):
            new_obs = new_obs.flatten()
        if hasattr(action, 'flatten'):
            action = action.flatten()
            
        success = self.replay_buffer.add(obs, new_obs, action, reward, done, infos)
        
        # Log para debug
        if hasattr(self, 'custom_logger') and self.custom_logger:
            if self.num_timesteps == 10000:
                buffer_size = len(self.replay_buffer)
                self.custom_logger.info(f"Buffer size: {buffer_size}, Última adição: {'Sucesso' if success else 'Falha'}")
    
    def _polyak_update(self, params, target_params, tau):
        """Implementação manual do polyak update"""
        with torch.no_grad():
            for param, target_param in zip(params, target_params):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(param.data * tau)

    def train(self, gradient_steps, batch_size=256):
        """Treinamento usando nosso buffer personalizado"""
        current_phase = self.phase_manager.current_phase
        self.replay_buffer.set_phase(current_phase)

        successful_steps = 0
        
        for gradient_step in range(gradient_steps):
            # Amostrar do nosso buffer personalizado
            batch_data = self.replay_buffer.sample(batch_size)
            
            if batch_data is None:
                continue
                
            obs, actions, rewards, next_obs, dones = batch_data
            
            with torch.no_grad():
                # Converter para tensores
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                actions_tensor = torch.FloatTensor(actions).to(self.device)
                next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
                dones_tensor = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
                
                # Noise para regularização
                noise = (torch.randn_like(actions_tensor) * self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                
                # Ações alvo com noise
                next_actions = (self.actor_target(next_obs_tensor) + noise).clamp(-1, 1)
                
                # Q-values alvo (twin critics)
                target_q1, target_q2 = self.critic_target(next_obs_tensor, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q
            
            # Q-values atuais
            current_q1, current_q2 = self.critic(obs_tensor, actions_tensor)
            
            # Loss do critic
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Otimizar critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # Atualizar actor (com policy delay)
            if gradient_step % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(obs_tensor, self.actor(obs_tensor)).mean()
                
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                
                # Atualizar redes alvo usando nossa implementação polyak
                self._polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                self._polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            
            successful_steps += 1

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
