import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from stable_baselines3.dqn.policies import DQNPolicy
from torchvision import transforms
import numpy as np


# Dimensiones de las imágenes capturadas por la cámara del dron
HEIGHT = 300
WIDTH = 300

# Número de filtros y dimensiones de los mapas de características
NUM_FILTERS = 32
TARGET_HEIGHT = 50
TARGET_WIDTH = 50


class DQNAgentPolicy(DQNPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Guardar valor "learning rate" recibido
        self.lr_schedule = lr_schedule

        # Cargar la red preentrenada DeepLabV3
        self.deeplabv3 = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self._device)

        # Congelar todas las capas
        for param in self.deeplabv3.parameters():
            param.requires_grad = False

        # Eliminar clasificador de DeepLabV3
        self.deeplabv3.classifier = nn.Identity()

        # Reducción del número de filtros de las imágenes procesadas
        self.conv_image = nn.Conv2d(2048, 32, kernel_size=1)

        # Reducción de la dimensionalidad de las imágenes procesadas
        self.maxpool = nn.AdaptiveMaxPool2d((TARGET_HEIGHT, TARGET_WIDTH))

        # MaxPool2d(8) -> 32 canales * 520/8 * 520/8
        # AdaptativeMaxPool2d -> definimos directamente las dimensiones de salida (100x100)

        num_features_flatten = NUM_FILTERS * TARGET_HEIGHT * TARGET_WIDTH

        # Definir la subred para procesar conjuntamente la imagen RGB y el mapa de profundidad
        self.subnet_image = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features_flatten * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        ).to(self._device)

        num_features_fc = 256 + 2

        # Definir la capa FC final
        self.fc_final = nn.Sequential(
            nn.Linear(num_features_fc, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_space.n)
        ).to(self._device)

        # Definir el normalizador para la información de profundidad y distancia
        self.distance_depth_normalizer = nn.BatchNorm1d(2).to(self._device)

        # Definir el optimizador para entrenar las capas añadidas
        self.optimizer = torch.optim.Adam([
            {'params': self.subnet_image.parameters()},
            {'params': self.fc_final.parameters()}
        ], lr=self.lr_schedule(1))

    def forward(self, observation):

        # Procesar la imagen RGB y el mapa de profundidad a través de DeepLabV3
        x_image = observation[:, :HEIGHT * WIDTH * 3].view(-1, 3, HEIGHT, WIDTH)
        x_depth = observation[:, HEIGHT * WIDTH * 3: 2 * HEIGHT * WIDTH * 3].view(-1, 3, HEIGHT, WIDTH)

        # Preprocesado de las imágenes
        x_image = self.preprocess_image(x_image)
        x_image = self.deeplabv3(x_image)["out"]

        x_depth = self.preprocess_image(x_depth)
        x_depth = self.deeplabv3(x_depth)["out"]

        # Aplicar la capa de convolución y reducir la dimensionalidad
        x_image = self.conv_image(x_image)
        x_image = self.maxpool(x_image)

        x_depth = self.conv_image(x_depth)
        x_depth = self.maxpool(x_depth)

        # Procesar ambas imágenes en su subred
        x_proc_images = self.subnet_image(torch.cat((x_image, x_depth), dim=1))

        # Procesar la información de profundidad y distancia
        x_sub = observation[:, 2 * HEIGHT * WIDTH * 3:]

        x_sub = self.distance_depth_normalizer(x_sub)

        # Concatenar las salidas de las subredes con la información de profundidad y distancia
        x = torch.cat((x_proc_images, x_sub), dim=1)

        # Calcular la salida de la capa fully-connected
        q_values = self.fc_final(x)

        return q_values

    def _predict(self, observation, deterministic=True):

        q_values = self.forward(observation)
        if deterministic:
            action = q_values.argmax(dim=1).detach().to('cpu').numpy()
        else:
            action = torch.multinomial(F.softmax(q_values, dim=1), num_samples=1).detach().to('cpu').numpy()

        return torch.as_tensor(action.flatten(), device=self._device)

    def set_training_mode(self, training):

        self.training = training
        self.deeplabv3.train(training)
        self.subnet_image.train(training)
        self.fc_final.train(training)
        self.distance_depth_normalizer.train(training)

    def preprocess_image(self, image):

        image = image.detach().cpu().numpy()
        image = np.squeeze(image, axis=0)

        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = preprocess(torch.tensor(image, dtype=torch.float32))
        tensor = tensor.to(self.device)

        return tensor.unsqueeze(0)
