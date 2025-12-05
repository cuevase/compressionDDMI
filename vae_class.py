from PIL import Image
import torch
import numpy as np
from efficientvit.ae_model_zoo import DCAE_HF # Model we will be using
from typing import Tuple, Optional

# This is the class that contains functions related to the vae, including the encoder and the decoder

class VAE:

    def __init__(self, model, quant: int, model_dims: Tuple[int, int], device: str):

        self.device = device
        self.model = model
        self.quant = quant
        self.model_dims = model_dims

    # loads image and returns loaded image_tensor and original size of image dimensions for future reference 
    def load_image(self, path_to_image: str):
        image = Image.open(path_to_image).convert("RGB")
        original_size = image.size
        image_resized = image.resize(self.model_dims, Image.LANCZOS) # Resize to VAE required size 

        # Create tensor accepted by DC-AE vae
        torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float() # Reorder dimensions, add batch dimension, convert to float32 and normalize to [0-1]# Reorder dimensions, add batch dimension, convert to float32 and normalize to [0-1]
        image_tensor = image_tensor.to(self.device) # Pass on the tensor to the device 

        return image_tensor, original_size

    # Encode with quantization 
    def encode_with_quant(self, image_tensor):
        with torch.no_grad():
            latents = self.model.encode(image_tensor)
            latent_min = latents.min()
            latent_max = latents.max()
            latents_int3 = ((latents - latent_min) / (latent_max - latent_min) * 7).round().to(torch.uint8)

            return latents_int3, latent_min, latent_max

    # Encode without quantization 
    def encode(self, image_tensor):

        with torch.no_grad():
            latents = self.model.encode(image_tensor)
        return latents 

    def decode_and_save(self, latents, quant: bool, latent_max: Optional[float], latent_min: Optional[float], path_to_save: str, original_size):

        # if quantized first dequantize
        if quant: 
            latents_dequant = latents.float() / 7.0 * (latent_max - latent_min) + latent_min
            latents = latents_dequant

        with torch.no_grad():
            recon = self.model.decode(latents)
            recon = recon.clamp(0, 1)
            recon_np = recon.squeeze().permute(1, 2, 0).cpu().numpy()
            recon_np = (recon_np * 255).astype(np.uint8)
            output_image = Image.fromarray(recon_np)
            output_image = output_image.resize(original_size, Image.LANCZOS)
            output_image.save(path_to_save, quality=95)

            print(f"Saved: {path_to_save}")



        





