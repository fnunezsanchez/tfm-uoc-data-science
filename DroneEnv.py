import gym
from gym import spaces
from typing import Dict, Any
from DroneController import DroneController
from MiDaSDepthGenerator import MiDaSDepthGenerator
import numpy as np
import sys
import sqlite3
import datetime


HEIGHT = 300
WIDTH = 300
MAX_STEPS = 300  # 120
DISTANCE_SCALE = 10
DEPTH_THRESHOLD = 220
COLLISION_PENALTY = 100
DISTANCE_REWARD_FACTOR = 1.5  # Factor de recompensa por distancia acumulada

action_names = {
    0: "avance",
    1: "giro izq.",
    2: "giro dcha."
}


class DroneEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.step_count = 0
        self.episode_reward = 0
        self.accumulated_distance = 0.0
        self.episode_rewards = []
        self.done = False
        self.previous_action = None
        self.collided = False

        # Inicializamos la traza para analizar acciones y recompensas

        unique_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trace = []
        self.conn = sqlite3.connect(f'./logs/{unique_id}_traces.db')
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                episode INTEGER,
                step_count INTEGER,                
                distance REAL,
                depth REAL,
                action INTEGER,
                reward REAL,
                done BOOLEAN                
            )
        """)
        self.conn.commit()

        # Acciones disponibles:
        # 0: Avanzar
        # 1: Girar izquierda
        # 2: Girar derecha

        self.action_space = spaces.Discrete(3)

        # Crea un espacio de observaciones que combina la imagen RGB (3 canales),
        # la imagen del mapa de profundidad (3 canales),
        # la medida de profundidad y la distancia recorrida acumulada.

        total_elements = HEIGHT * WIDTH * 3 + HEIGHT * WIDTH * 3 + 1 + 1

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(total_elements,), dtype=np.float32)

        # Inicializa el controlador del dron y el generador de mapas de profundidad
        self.drone_controller = DroneController()
        self.depth_generator = MiDaSDepthGenerator(model_type="DPT_Large")

        # Inicializar la posición anterior del dron
        self.previous_position = np.zeros(3)

    def reset(self):
        # Indicar que no ha finalizado el episodio
        self.done = False

        # Reiniciar valor de acción anterior
        self.previous_action = None

        # Inicializar indicador de colisión
        self.collided = False

        # Inicializar la recompensa del episodio a cero
        self.episode_reward = 0

        # Reiniciar contador de pasos de tiempo
        self.step_count = 0

        # Inicializar contador de distancia recorrida
        self.accumulated_distance = 0.0

        # Iniciar la traza de acciones y recompensas
        self.trace = []

        # Reiniciar el dron
        self.drone_controller.shutdown()
        self.drone_controller.__init__()
        self.drone_controller.takeoff()

        # Tomar una foto con el dron, obtener mapa profundidad y calcular profundidad media
        image_rgb_observation = self.drone_controller.take_photo()
        depth_map_observation = self.depth_generator.get_depth_map(image_rgb_observation)
        mean_depth_observation = self.get_mean_depth(depth_map_observation)

        # Calcular la posición del dron
        previous_position = self.drone_controller.update_estimated_position()
        self.previous_position = previous_position

        # Combinar los elementos de la observación
        observation = {
            'image_rgb': image_rgb_observation,
            'depth_map': depth_map_observation,
            'mean_depth': np.array([float(mean_depth_observation)], dtype=np.float32),
            'distance': np.array([0.0], dtype=np.float32),
        }

        return self.process_observation(observation)

    def step(self, action):
        # Guardar la acción actual para la siguiente iteración
        self.previous_action = action

        # Ejecutar la acción en el entorno
        self.collided = self.drone_controller.execute_action(action)

        # Obtener la observación resultante
        image_rgb_observation = self.drone_controller.take_photo()
        depth_map_observation = self.depth_generator.get_depth_map(image_rgb_observation)
        mean_depth_observation = self.get_mean_depth(depth_map_observation)

        # Obtener la posición anterior del dron
        prev_position = self.previous_position
        prev_position_scaled = prev_position * DISTANCE_SCALE

        # Obtener la posición actual del dron
        estimated_position = self.drone_controller.update_estimated_position()
        estimated_position_scaled = estimated_position * DISTANCE_SCALE

        # Calcular la distancia recorrida por la acción
        distance = np.linalg.norm(estimated_position_scaled - prev_position_scaled)

        # print(f"Posición actual: {estimated_position}")
        # print(f"Posición anterior: {prev_position}")
        # print("Distancia recorrida: ", distance)
        # print("Profundidad: ", mean_depth)

        # Actualizar la distancia acumulada
        if action == 0:
            self.accumulated_distance += float(distance)

        # Actualiza la posición anterior
        self.previous_position = estimated_position

        # Combiar los elementos de la observación
        observation = {
            'image_rgb': image_rgb_observation,
            'depth_map': depth_map_observation,
            'mean_depth': np.array([float(mean_depth_observation)], dtype=np.float32),
            'distance': np.array([float(self.accumulated_distance)], dtype=np.float32),
        }

        # Calcular la recompensa
        reward = self.calculate_reward(observation, action)

        # Sumar la recompensa del paso actual a la recompensa del episodio
        self.episode_reward += reward

        # Aumentar el contador de pasos
        self.step_count += 1

        # Condición para finalizar el episodio si se supera el máximo de pasos por episodio
        if self.step_count >= MAX_STEPS:
            print("Superado número máximo de pasos")
            self.done = True

        # Añadir datos a la traza
        self.trace.append({
            'step_count': self.step_count,
            'distance': float(self.accumulated_distance),
            'depth': float(mean_depth_observation),
            'action': int(action),
            'reward': reward,
            'done': self.done
        })

        # Si el episodio ha terminado, almacenar la recompensa del episodio
        if self.done:
            self.episode_rewards.append(self.episode_reward)
            self.save_trace(episode=len(self.episode_rewards) - 1)
            print("RECOMPENSA TOTAL EPISODIO: ", round(self.episode_reward, 2))

        # Vaciar buffer salida consola
        sys.stdout.flush()

        return self.process_observation(observation), reward, self.done, {}

    def calculate_reward(self, observation, action):
        reward = 0

        # Obtener la distancia recorrida acumulada
        distance = observation['distance']

        # Obtener la profundidad media en el centro de la imagen
        mean_depth = observation['mean_depth']

        # Obtener el nombre de la acción
        action_name = action_names.get(action)

        if self.collided:
            reward -= COLLISION_PENALTY  # Penalización por colisión
            self.done = True  # Terminar episodio por colisión
        else:
            if action == 0:
                reward = (1 + distance.item() * DISTANCE_REWARD_FACTOR) / (mean_depth.item() + 1)
            else:
                reward = 1 / (mean_depth.item() + 1)

        print(
            action_name.upper(),
            round(reward, 2),
            "distancia acumulada:",
            round(distance.item(), 2),
            "profundidad:",
            round(mean_depth.item(), 2)
        )

        # Devolver recompensa obtenida
        return float(reward)

    def render(self, mode='human'):
        # No es necesario implementarla
        pass

    def close(self):
        self.drone_controller.shutdown()
        self.cur.close()
        self.conn.close()

    def get_mean_depth(self,
                       depth_map,
                       roi_height_ratio_horizontal=0.55,
                       roi_height_ratio_vertical=0.4,
                       roi_width_ratio_horizontal=0.8,
                       roi_width_ratio_vertical=0.5,
                       offset_ratio_horizontal=0.1,
                       offset_ratio_vertical=0.1):

        # Inicializar variable retorno
        depth_value = 0.0

        # Determinar las dimensiones de la imagen del mapa de profundidad
        img = depth_map
        h, w = img.shape[:2]

        # Definir los parámetros de la franja horizontal de la ROI
        strip_height_horizontal_extension = int(roi_height_ratio_horizontal * h)
        strip_start_y_horizontal = (h - strip_height_horizontal_extension) // 2
        strip_start_y_horizontal += int(offset_ratio_horizontal * h)  # Desplazamiento hacia abajo
        strip_end_y_horizontal = strip_start_y_horizontal + strip_height_horizontal_extension
        strip_width_horizontal = int(roi_width_ratio_horizontal * w)
        strip_start_x_horizontal = (w - strip_width_horizontal) // 2
        strip_end_x_horizontal = strip_start_x_horizontal + strip_width_horizontal

        # Definir los parámetros de la franja vertical de la ROI
        strip_width_vertical_extension = int(roi_width_ratio_vertical * w)
        strip_start_x_vertical = (w - strip_width_vertical_extension) // 2
        strip_end_x_vertical = strip_start_x_vertical + strip_width_vertical_extension
        strip_height_vertical = int(roi_height_ratio_vertical * h)
        strip_start_y_vertical = (h - strip_height_vertical) // 2
        strip_start_y_vertical += int(offset_ratio_vertical * h)  # Desplazamiento hacia abajo
        strip_end_y_vertical = strip_start_y_vertical + strip_height_vertical

        # Recortar las ROI de la imagen
        roi_horizontal = img[strip_start_y_horizontal:strip_end_y_horizontal,
                         strip_start_x_horizontal:strip_end_x_horizontal]
        roi_vertical = img[strip_start_y_vertical:strip_end_y_vertical, strip_start_x_vertical:strip_end_x_vertical]

        # Encontrar el valor medio en las ROI
        # mean_value_horizontal = roi_horizontal.mean()
        mean_value_vertical = roi_vertical.mean()

        # Encontrar el valor máximo en las ROI
        max_value_horizontal = roi_horizontal.max()
        # max_value_vertical = roi_vertical.max()

        if max_value_horizontal >= DEPTH_THRESHOLD:
            depth_value = max_value_horizontal
        else:
            depth_value = mean_value_vertical

        return float(depth_value/10)

    def process_observation(self, observations: Dict[str, Any]):
        # Preprocesar las observaciones
        np_image_rgb = np.array(observations['image_rgb'], dtype=np.float32).flatten()
        np_depth_map = np.array(observations['depth_map'], dtype=np.float32).flatten()
        np_mean_depth = np.expand_dims(observations['mean_depth'], axis=0).flatten()
        np_distance = np.expand_dims(observations['distance'], axis=0).flatten()

        # Concatenar las observaciones preprocesadas
        processed_observations = np.concatenate([np_image_rgb, np_depth_map, np_mean_depth, np_distance], axis=0)

        # Asegurar que la forma de la observación coincide con la forma del espacio de observación
        assert processed_observations.shape == self.observation_space.shape

        # Devuelve la observación como un único array
        return processed_observations

    def get_episode_rewards(self):
        # print("** EPISODE REWARDS ** ", self.episode_rewards)
        return self.episode_rewards

    def save_trace(self, episode):
        for trace in self.trace:
            self.cur.execute("""
                    INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (episode, trace['step_count'], trace['distance'], trace['depth'],
                      trace['action'], trace['reward'], trace['done']))
        self.conn.commit()
