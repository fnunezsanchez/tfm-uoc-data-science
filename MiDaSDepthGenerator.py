import torch
import numpy as np
import cv2

HEIGHT = 300
WIDTH = 300

class MiDaSDepthGenerator:

    def __init__(self, model_type="DPT_Large"):

        # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas, self.transform = self.init_midas(model_type)

    def init_midas(self, model_type):
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(self.device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        return midas, transform

    def get_depth_map(self, image_rgb):
        with torch.no_grad():
            input_batch = self.transform(image_rgb).to(self.device)
            prediction = self.midas(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        # Redimensionar a 300x300
        depth_map_resized = cv2.resize(depth_map, (HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)

        # Normalizar el mapa de profundidad
        depth_map_normalized = (depth_map_resized - depth_map_resized.min()) / (
                    depth_map_resized.max() - depth_map_resized.min()) * 255
        depth_map_normalized = depth_map_normalized.astype(np.uint8)

        # Convertir a RGB
        depth_map_rgb = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2RGB)

        return depth_map_rgb
