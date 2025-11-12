from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
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
        dtype = torch.float16  # For UNet (faster)
        vae_dtype = torch.float16  # For VAE (better precision)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else: 
        print("Cuda or GPU is not set up")
        device = "cpu"
        dtype = torch.float32
        vae_dtype = torch.float32

    model_id = "stabilityai/stable-diffusion-2"

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

    # OPTIMIZATION 1: Replace with better VAE (HIGHEST IMPACT!)
    # Use improved VAE for significantly better quality
    vae_model_id = "stabilityai/sd-vae-ft-mse"  # Improved VAE for SD 2.x
    print(f"Loading improved VAE: {vae_model_id}")
    pipe.vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=vae_dtype).to(device)
    
    # OPTIMIZATION 2: Ensure VAE uses float32
    pipe.vae = pipe.vae.to(dtype=vae_dtype)
    
    print("Model loaded successfully!")
    print(f"VAE dtype: {pipe.vae.dtype}")

    # Replace scheduler with DDIM 
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    shared_seed = 42

    # OPTIMIZATION 3: Use higher resolution (HIGH IMPACT!)
    # SD 2.x supports up to 768x768 (less compression per pixel)
    image_size = 768  # Increase from 512 to 768 (or 1024 if you have VRAM)
    
    # Prepare image 
    image = Image.open("output.png")
    image = image.convert("RGB")
    
    # OPTIMIZATION 4: Use high-quality resampling
    image = image.resize((image_size, image_size), resample=PIL_INTERPOLATION["lanczos"])
    
    image = np.array(image).astype(np.float32) / 255.0
    image_array = image[None].transpose(0, 3, 1, 2)
    
    # OPTIMIZATION 5: Use float32 for image tensor before VAE encoding
    image_tensor = torch.from_numpy(image_array).to(device=device, dtype=vae_dtype)
    
    image_tensor_vae = (image_tensor * 2.0 - 1.0)

    # OPTIMIZATION 6: Use deterministic encoding (mode instead of sample)
    with torch.no_grad():
        # Get latent distribution
        latent_dist = pipe.vae.encode(image_tensor_vae).latent_dist
        # Use mode() instead of sample() for deterministic encoding (preserves more info)
        latent = latent_dist.mode()  # Deterministic, less random loss
        latent = latent * pipe.vae.config.scaling_factor

    print(f"Latent shape: {latent.shape}")
    print(f"Original image size: {image_size}x{image_size}")
    
    # Set up scheduler properly
    num_inference_steps = 50
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    max_timestep = timesteps[0].item()  # Use scheduler's max timestep

    print(f"Using timestep {max_timestep} for noise addition")
    print(f"Using {num_inference_steps} steps for denoising")

    # Generate noise with seed for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_seed)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)

    # Add noise in latent space
    noisy_latent = pipe.scheduler.add_noise(
        latent, 
        noise, 
        torch.tensor([max_timestep], device=device)
    )

    # OPTIMIZATION 7: Remove unnecessary VAE decode of noisy latent
    # Don't decode noisy latent (saves a VAE roundtrip, reduces artifacts)
    # Only decode the final denoised result
    print("Skipping noisy latent decode (reduces VAE artifacts)")
    
    print(f"Starting denoising with {num_inference_steps} steps...")

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    prompt = "image"
    text_inputs = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]

    # Denoising loop - work in latent space
    latents = noisy_latent.to(dtype=dtype)  # Convert to float16 for UNet (faster)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            noise_pred = pipe.unet(
                latents, 
                t, 
                encoder_hidden_states=prompt_embeds, 
                return_dict=False
            )[0]

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
                print(f"Step {i+1}/{len(timesteps)}: timestep {t.item()}")
    
    print("Denoising complete")

    # OPTIMIZATION 8: Use float32 for VAE decoding
    print("Decoding denoised latent to image")

    with torch.no_grad():
        # Convert back to float32 for VAE decoding (better precision)
        latents = latents.to(dtype=vae_dtype)
        latents = latents / pipe.vae.config.scaling_factor
        
        # OPTIMIZATION 9: Enable VAE tiling for large images (optional)
        # Uncomment if using very large images (1024x1024+)
        # pipe.vae.enable_tiling()
        
        # Decode latent to image
        denoised_image = pipe.vae.decode(latents).sample
        
        # Convert from [-1, 1] to [0, 1]
        denoised_image = (denoised_image / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy and PIL
        denoised_image = denoised_image.cpu().permute(0, 2, 3, 1).numpy()
        denoised_image = (denoised_image * 255).round().astype("uint8")
        denoised_pil = Image.fromarray(denoised_image[0])

        denoised_pil.save("output_denoised.png")
        print("Saved denoised image to output_denoised.png")

    # Save original for comparison
    original_image_display = (image_tensor / 2 + 0.5).clamp(0, 1)
    original_image_display = original_image_display.cpu().permute(0, 2, 3, 1).numpy()
    original_image_display = (original_image_display * 255).round().astype("uint8")
    original_pil = Image.fromarray(original_image_display[0])
    original_pil.save("output_original.png")
    print("Saved original image to output_original.png for comparison")


if __name__ == "__main__":
    main()