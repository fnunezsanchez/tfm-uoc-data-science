import airsim
from airsim import LandedState
import math
from PIL import Image
import io
import numpy as np
import time


# Dimensiones de las imágenes capturadas por la cámara del dron
HEIGHT = 300
WIDTH = 300


class DroneController:

    def __init__(self):

        # Conectar al cliente de AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.drive_train = airsim.DrivetrainType.ForwardOnly
        self.estimated_position = np.array([0.0, 0.0, 0.0])
        self.last_update_time = time.time()

    def takeoff(self, max_height=-0.6):

        # Despegar y elevar el dron a la altitud indicada
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(max_height, 1).join()  # Cambiar a la altitud deseada
        return self.has_collided()

    def advance(self, speed=1, duration=0.7):

        # Obtener la posición y orientación actuales del dron
        state = self.client.getMultirotorState()
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        height = state.kinematics_estimated.position.z_val

        # Calcular las velocidades en los ejes X e Y basándose en el ángulo de giro actual (yaw)
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)

        # Mantener la altura actual (z) y la orientación actual en el eje Yaw
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityZAsync(
            vx,
            vy,
            height,
            duration=duration,
            drivetrain=self.drive_train,
            yaw_mode=yaw_mode
        ).join()

        return self.has_collided()

    def turn_left(self, angle, duration=1):
        current_yaw = self.get_current_yaw()
        self.client.rotateToYawAsync(math.degrees(current_yaw) - angle, duration, margin=0.5).join()
        return self.has_collided()

    def turn_right(self, angle, duration=1):
        current_yaw = self.get_current_yaw()
        self.client.rotateToYawAsync(math.degrees(current_yaw) + angle, duration, margin=0.5).join()
        return self.has_collided()

    def turn_180(self, duration=3):
        current_yaw = self.get_current_yaw()
        self.client.rotateToYawAsync(math.degrees(current_yaw) + 180, duration, margin=0.5).join()
        return self.has_collided()

    def ascend(self, speed, duration=1):
        z = self.client.simGetVehiclePose().position.z_val
        self.client.moveByVelocityZAsync(0, 0, z - speed, duration).join()
        return self.has_collided()

    def descend(self, speed, duration=1):
        z = self.client.simGetVehiclePose().position.z_val
        self.client.moveByVelocityZAsync(0, 0, z + speed, duration).join()
        return self.has_collided()

    def stop(self):
        self.client.hoverAsync().join()

    def shutdown(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def get_current_yaw(self):
        quaternion = self.client.simGetVehiclePose().orientation
        _, _, yaw = airsim.to_eularian_angles(quaternion)
        return yaw

    def execute_action(self, action):

        collided = False

        if action == 0:
            # print("Avanzar")
            collided = self.advance(1)  # Avanza a una velocidad de 0.5 m/s
            # Necesario para estabilizar y eliminar errores yaw en AirSim
            collided |= self.descend(.01)
            collided |= self.ascend(.01)

        elif action == 1:
            # print("Girar izquierda")
            collided = self.turn_left(45)  # Gira a la izquierda 10 grados

        elif action == 2:
            # print("Girar derecha")
            collided = self.turn_right(45)  # Gira a la derecha 10 grados

        elif action == 3:
            collided = self.turn_180()

        elif action == 4:
            # print("Ascender")
            collided = self.ascend(0.4)

        elif action == 5:
            # print("Descender")
            collided = self.descend(0.4)

        else:
            raise ValueError("Acción no válida: {}".format(action))

        return collided

    def take_photo(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene)
        ])

        image_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        resized_image = image.resize((WIDTH, HEIGHT))
        resized_image_np = np.array(resized_image)

        return resized_image_np

    def update_estimated_position(self):

        current_time = time.time()
        dt = current_time - self.last_update_time

        # print("Diferencia de tiempo en el cálculo de la posición: ", dt)

        # Obtener las velocidades lineales actuales
        kinematics = self.client.getMultirotorState().kinematics_estimated
        linear_velocity = np.array([kinematics.linear_velocity.x_val,
                                    kinematics.linear_velocity.y_val,
                                    kinematics.linear_velocity.z_val])

        # print("Velocidades lineales actuales: ", linear_velocity)

        # Actualizar la posición estimada
        self.estimated_position += linear_velocity * dt
        # Actualizar el último tiempo de actualización
        self.last_update_time = current_time

        # print("Posición estimada actualizada: ", self.estimated_position)

        return self.estimated_position

    def has_collided(self):
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided

    def has_stopped(self):

        stopped = False

        state = self.client.getMultirotorState()
        landed_state = state.landed_state

        # Si el dron está en tierra, significa que se ha detenido
        if landed_state == LandedState.Landed:
            stopped = True

        # Obtener la altura del dron
        state = self.client.simGetVehiclePose()
        altitude = -state.position.z_val

        #print("Altura de vuelo: ", altitude)

        # Comprobar si se ha producido una colisión con el terreno
        if altitude <= 0.09:
            print("Colisión con el suelo detectada.")
            stopped = True

        return stopped

