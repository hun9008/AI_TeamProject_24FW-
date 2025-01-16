import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize
from sklearn.decomposition import PCA

"""
그대로 transforms에 집어넣어서 사용하시면 됩니다.
Macenko Normalization을 먼저 다 수행한 뒤에 Tensor로 변경하여
이후 다른 처리들 (flip, noise, blur, ...)을 진행하세요.
"""


class MacenkoNormalization:
    """
    Custom transform for Macenko normalization in PyTorch pipelines.
    """
    def __init__(self, reference_image_path, output_shape=None):
        self.reference_image = cv2.imread(reference_image_path)
        self.output_shape = output_shape
        
        if self.reference_image is None:
            raise ValueError("Reference image could not be loaded.")

    def __call__(self, image):
        """
        Apply Macenko normalization to the input image.

        Parameters:
        - image: Input image (PIL Image or Tensor).

        Returns:
        - Normalized image (Tensor).
        """
        if isinstance(image, torch.Tensor):
            image = ToPILImage()(image)

        image = np.array(image)[:, :, ::-1]  # Convert PIL (RGB) to OpenCV (BGR)

        # Perform normalization
        normalized_image = self.macenko_normalization(image, self.reference_image, self.output_shape)

        # Convert back to PIL and then to Tensor
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
        return ToTensor()(normalized_image)

    def macenko_normalization(self, image, reference_image, output_shape=None):
        def rgb_to_od(rgb):
            rgb = rgb.astype(np.float32)
            rgb[rgb == 0] = 1  # Avoid log(0)
            return -np.log(rgb / 255 + 1e-6)

        def od_to_rgb(od):
            return (255 * np.exp(-od)).clip(0, 255).astype(np.uint8)

        def stain_separation(od, n_components=2):
            flat_od = od.reshape(-1, 3)
            flat_od = flat_od[~np.all(flat_od < 0.15, axis=1)]
            pca = PCA(n_components=n_components)
            return pca.fit(flat_od).components_[:n_components]

        def normalize_stains(od, stain_matrix, target_concentrations):
            concentrations = np.linalg.lstsq(stain_matrix, od.T, rcond=None)[0]
            normalized_od = np.dot(target_concentrations, stain_matrix).T
            return normalized_od.reshape(od.shape)

        if output_shape:
            image = cv2.resize(image, output_shape)
            reference_image = cv2.resize(reference_image, output_shape)

        od = rgb_to_od(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        reference_od = rgb_to_od(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))

        stain_matrix = stain_separation(od)
        reference_stain_matrix = stain_separation(reference_od)

        target_concentrations = np.dot(reference_stain_matrix, reference_stain_matrix.T)
        normalized_od = normalize_stains(od, stain_matrix, target_concentrations)

        return od_to_rgb(normalized_od)
