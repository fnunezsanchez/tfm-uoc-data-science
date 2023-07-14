from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from DroneEnv import DroneEnv
from DQNAgentPolicy import DQNAgentPolicy
import numpy as np
import datetime
import time
import os

REWARD_EPISODE_FREQ = 50


# *************************************************
class TensorboardCallback(BaseCallback):

    def __init__(self, envr, check_freq: int, log_dir: str, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.env = envr
        self.episodes_recorded = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

            # Recoger las recompensas de los episodios
            episode_rewards = self.env.envs[0].get_episode_rewards()

            if len(episode_rewards) > 0:

                # Registrar recompensas de cada episodio
                for i in range(self.episodes_recorded, len(episode_rewards)):
                    self.logger.record(f'Propias/rew_ep', float(episode_rewards[i]))

                # Actualizar contador episodios registrados
                self.episodes_recorded = len(episode_rewards)

                # Calcular la recompensa media de los últimos REWARD_EPISODE_FREQ
                mean_reward = np.mean(episode_rewards[-REWARD_EPISODE_FREQ:])
                self.logger.record('Propias/rew_mean_last_' + str(REWARD_EPISODE_FREQ)
                                   + ' episodios',
                                   float(mean_reward))

                # Guardar el modelo con la mejor recompensa
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))

        return True

# *************************************************


# Crear una carpeta única para la ejecución del entrenamiento
unique_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_train_dir = f"./logs/{unique_id}/"
os.makedirs(log_train_dir, exist_ok=True)

# Crear el entorno personalizado
env = DroneEnv()

# Comprobar el entorno
check_env(env)

# Envolver el entorno en un DummyVecEnv para que sea compatible con Stable Baselines 3
env = DummyVecEnv([lambda: env])

# Envolver el entorno en un VecMonitor para monitorizar entrenamiento
env = VecMonitor(env, log_train_dir)

# Configurar log para tensorflow
logger = configure(log_train_dir, ["stdout", "csv", "tensorboard"])

# Configurar callback para gráfica recompensa
callback = TensorboardCallback(envr=env, check_freq=REWARD_EPISODE_FREQ, log_dir=log_train_dir)


# Definir valores hiperparámetros
_buffer_size = 10000
_learning_starts = 5000
_train_freq = 4
_target_update_interval = 1000
_exploration_fraction = 0.45
_exploration_final_eps = 0.01
_batch_size = 32
_gamma = 0.99
_learning_rate = 0.00008
_total_timesteps = 50000


# Crear el modelo de RL usando la política personalizada
model = DQN(policy=DQNAgentPolicy,
            env=env,
            buffer_size=_buffer_size,
            learning_starts=_learning_starts,
            train_freq=_train_freq,
            target_update_interval=_target_update_interval,
            exploration_fraction=_exploration_fraction,
            exploration_final_eps=_exploration_final_eps,
            batch_size=_batch_size,
            gamma=_gamma,
            learning_rate=_learning_rate,
            verbose=1,
            device="cuda")

model.set_logger(logger)

print("Iniciando el entrenamiento...")

# Obtener el tiempo actual antes de iniciar el entrenamiento
start_time = time.time()
start_datetime = datetime.datetime.now()
print("Tiempo de inicio:", start_datetime)

# Entrenar el modelo
model.learn(total_timesteps=_total_timesteps, log_interval=1, callback=callback)

# Obtener el tiempo actual al finalizar el entrenamiento
end_time = time.time()
end_datetime = datetime.datetime.now()
print("Tiempo de finalización:", end_datetime)

# Calcular el tiempo transcurrido en horas y minutos
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
time_string = "{}:{}".format(hours, minutes)
print("Tiempo transcurrido:", time_string)

# Escribir los hiperparámetros y tiempo de entrenamiento en un archivo .txt
with open(log_train_dir + 'hyperparameters.txt', 'w') as f:
    f.write(f'_buffer_size = {_buffer_size}\n')
    f.write(f'_learning_starts = {_learning_starts}\n')
    f.write(f'_train_freq = {_train_freq}\n')
    f.write(f'_target_update_interval = {_target_update_interval}\n')
    f.write(f'_exploration_fraction = {_exploration_fraction}\n')
    f.write(f'_exploration_final_eps = {_exploration_final_eps}\n')
    f.write(f'_batch_size = {_batch_size}\n')
    f.write(f'_gamma = {_gamma}\n')
    f.write(f'_learning_rate = {_learning_rate}\n')
    f.write(f'_total_timesteps = {_total_timesteps}\n')
    f.write(f'_tiempo_entrenamiento = {time_string}\n')
    f.write(f'_fecha_inicio = {start_datetime}\n')
    f.write(f'_fecha_fin = {end_datetime}\n')


# Guardar el último modelo
model.save(f"{log_train_dir}/models/model")
