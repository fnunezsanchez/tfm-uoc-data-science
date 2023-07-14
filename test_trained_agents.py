from sb3_contrib.qrdqn import QRDQN
from DroneEnv import DroneEnv
import csv
import time
import os

TOTAL_EPISODIOS = 50

# Directorio para guardar los resultados
dir_path = './logs/csv_episodios_prueba/comedor/'

# Verificar si el directorio existe, y si no, crearlo
os.makedirs(dir_path, exist_ok=True)

# Lista de modelos a probar
models_dir = [
    "./logs/QRDQN_50k_1/models/model",
    "./logs/QRDQN_50k_2/models/model",
    "./logs/QRDQN_50k_3/models/model"
]

count = 0

# Bucle a través de cada modelo
for model_dir in models_dir:

    # Cargar el modelo
    model = QRDQN.load(model_dir)
    env = DroneEnv()

    # Desactivar el dropout en la política

    # Crear el entorno
    model.policy.set_training_mode(False)

    # Variables para rastrear las estadísticas
    collision_count = 0
    no_collision_count = 0
    total_duration = 0

    # Preparar el archivo CSV
    with open(f'{dir_path}resultados_modelo{count}.csv', 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(["Episodio", "Duracion (segundos)", "Recompensa total"])

        # Ejecutar episodios de prueba
        for episode in range(TOTAL_EPISODIOS):

            start_time = time.time()
            obs = env.reset()
            done = False

            total_reward = 0
            collision = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = action.item()
                obs, reward, done, info = env.step(action)
                total_reward += reward

                # Comprueba si ha habido colisión
                if reward == -100:
                    collision = True

            duration = time.time() - start_time

            # Actualiza las estadísticas
            if collision:
                collision_count += 1
            else:
                no_collision_count += 1

            total_duration += duration

            writer.writerow([episode, duration, total_reward])

            # Muestra las estadísticas hasta el momento
            print("*********************************************")
            print("MODELO ACTUAL: ", model_dir)
            print(f'Episodios terminados: {episode + 1}')
            print(f'Episodios finalizados por colisión: {collision_count}')
            print(f'Episodios finalizados sin colisión: {no_collision_count}')
            print(f'Duración media de los episodios: {total_duration / (episode + 1)} segundos\n')
            print("********************************************")

            count = count + 1
