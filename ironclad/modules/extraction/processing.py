# ironclad/modules/extraction/processing.py

from PIL import Image
import torchvision.transforms as transforms

# Preprocessing pipeline for face images
class Preprocessor:
    def __init__(self, image_size=(160, 160)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize to common dimensions
            transforms.CenterCrop(image_size),  # Center crop
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(  # Normalize using common face model values
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    def preprocess(self, image_path):
        """
        Preprocesses an image and returns a tensor suitable for embedding extraction.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - tensor (torch.Tensor): Preprocessed image tensor.
        """
        image = Image.open(image_path).convert("RGB")  # Open as RGB
        return self.transform(image)
