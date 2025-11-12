from diffusers import DDPMPipeline, DDIMScheduler
from helpers import generate_noise
import torch
import os
from PIL import Image
from diffusers.utils import PIL_INTERPOLATION
import numpy as np


def main():

    # Check CUDA availability, raise print error if not
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Sets the precision used for the model weights and computations
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else: 
        print("Cuda or GPU is not set up")
        device = "cpu"
        dtype = torch.float32

    # Load pixel-space model (NO VAE!)
    model_id = "google/ddpm-celebahq-256"
    
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

    print("Model loaded successfully!")

    # Replace scheduler with DDIM 
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    shared_seed = 42

    # Prepare image - resize to 256x256 (model expects this size)
    image = Image.open("output.png")
    image = image.convert("RGB")  # Ensure that image is RGB
    image = image.resize((256, 256), resample=PIL_INTERPOLATION["lanczos"])  # 256x256 for celebahq model
    image = np.array(image).astype(np.float32) / 255.0  # Converts image to numpy array and normalizes values from 0 to 1
    image_array = image[None].transpose(0, 3, 1, 2)  # Changes shape to make it ready for PyTorch with (batch, channels, height, width)
    image_tensor = torch.from_numpy(image_array)  # Converts numpy to PyTorch tensor
    image_tensor = image_tensor.to(device=device, dtype=dtype)  # Moves tensor to GPU and with correct dtype
    
    # NO VAE ENCODING - work directly on pixels!
    # Normalize to [-1, 1] range (typical for diffusion models)
    image_tensor = (image_tensor * 2.0 - 1.0)

    print(f"Image tensor shape: {image_tensor.shape}")  # Should be (1, 3, 256, 256)

    # Set up scheduler for noise addition
    num_inference_steps = 5 # Reasonable number of steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    max_timestep = timesteps[0].item()  # Get maximum timestep from scheduler

    print(f"Using timestep {max_timestep} for noise addition")
    print(f"Using {num_inference_steps} steps for denoising")

    # Generate noise with seed for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_seed)
    noise = torch.randn(image_tensor.shape, generator=generator, device=device, dtype=dtype)

    # Add noise directly on pixels (NO VAE!)
    noisy_image = pipe.scheduler.add_noise(
        image_tensor, 
        noise, 
        torch.tensor([max_timestep], device=device)
    )

    print("Decoding noisy image (no VAE needed - already pixels!)")

    with torch.no_grad():
        # Convert from [-1, 1] to [0, 1]
        noisy_image_display = (noisy_image / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy and PIL
        noisy_image_display = noisy_image_display.cpu().permute(0, 2, 3, 1).numpy()
        noisy_image_display = (noisy_image_display * 255).round().astype("uint8")
        noisy_pil = Image.fromarray(noisy_image_display[0])

        noisy_pil.save("output_noised.png")
        print("Saved noisy image to output_noised.png")
    
    # NO TEXT CONDITIONING - DDPM is unconditional!
    print(f"Starting denoising with {num_inference_steps} steps...")
    print("Note: This model is unconditional (no text prompts)")

    # Set timesteps for denoising (already set above, but can reset if needed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Start with noisy image (already in pixel space!)
    latents = noisy_image

    # Denoising loop - NO TEXT CONDITIONING!
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # UNet prediction - NO encoder_hidden_states needed!
            noise_pred = pipe.unet(latents, t, return_dict=False)[0]

            # Compute previous noisy sample x_t -> x_{t-1}
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
                print(f"Step {i+1}/{len(timesteps)}: timestep {t.item()}")
    
    print("Denoising complete")

    # Decode denoised image (no VAE - already pixels!)
    print("Converting denoised image (no VAE decoding needed)")

    with torch.no_grad():
        # Convert from [-1, 1] to [0, 1]
        denoised_image = (latents / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy and PIL
        denoised_image = denoised_image.cpu().permute(0, 2, 3, 1).numpy()
        denoised_image = (denoised_image * 255).round().astype("uint8")
        denoised_pil = Image.fromarray(denoised_image[0])

        denoised_pil.save("output_denoised.png")
        print("Saved denoised image to output_denoised.png")

    # Also save original for comparison
    original_image_display = (image_tensor / 2 + 0.5).clamp(0, 1)
    original_image_display = original_image_display.cpu().permute(0, 2, 3, 1).numpy()
    original_image_display = (original_image_display * 255).round().astype("uint8")
    original_pil = Image.fromarray(original_image_display[0])
    original_pil.save("output_original.png")
    print("Saved original image to output_original.png for comparison")


if __name__ == "__main__":
    main()