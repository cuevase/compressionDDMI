from diffusers import StableDiffusionPipeline, DDIMScheduler
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

    model_id = "stabilityai/stable-diffusion-2"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

    print("Model loaded successfully!")

    # Replace scheduler with DDIM 
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    #shared_seed = 42
   # noise_sender = generate_noise(shared_seed, (1, 3 , 64, 64)) # Create noisy image that is same for receiver
   # noise_receiver = generate_noise(shared_seed, (1, 3, 64, 64)) # Create noisy image that is same as sender
   # print(torch.equal(noise_sender, noise_receiver))
   # print(noise_sender)
    shared_seed = 42

    # Prepare image 
    image = Image.open("output.png")
    image = image.convert("RGB") # Ensure that image is RGB
    image = image.resize((512,512), resample = PIL_INTERPOLATION["lanczos"]) # Resizes to size that stable diffusion is expecting
    image = np.array(image).astype(np.float32) / 255.0 # Converts image to numpy array and normalizes values from 0 to 1
    image_array = image[None].transpose(0, 3, 1, 2) # Changes shape to make it ready for PyTorch with (batch, channels, height, width)
    image_tensor = torch.from_numpy(image_array) # Converts numpy to PyTorch tensor
    image_tensor = image_tensor.to(device = device, dtype = dtype) # Moves tensor to GPU and with correct dtype
    
    image_tensor_vae = (image_tensor* 2.0 - 1.0)

    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor_vae).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor

    print(f"Latent shape: {latent.shape}")
    num_inference_steps = 200
    timestep = num_inference_steps - 1 

    # Generate noise with seed for reproducibility
    generator = torch.Generator(device = device)
    generator.manual_seed(shared_seed)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)


    noisy_latent = pipe.scheduler.add_noise(latent, noise, torch.tensor([timestep], device = device))

    print("Decoding noisy latent to image")

    with torch.no_grad():

        noisy_latent_decoded = noisy_latent / pipe.vae.config.scaling_factor 

        noisy_image = pipe.vae.decode(noisy_latent_decoded).sample
        
        noisy_image = (noisy_image/2 + 0.5).clamp(0,1)

        noisy_image = noisy_image.cpu().permute(0,2,3,1).numpy()
        noisy_image = (noisy_image * 255).round().astype("uint8")
        noisy_pil = Image.fromarray(noisy_image[0])

        noisy_pil.save("output_noised.png")
    
    print(f"Starting denoising with {num_inference_steps} steps...")

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    prompt = "yellow city with a lot of sun"
    text_inputs = tokenizer(prompt, padding = "max_length", max_length = tokenizer.model_max_length, truncation = True, return_tensors = "pt")
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]

    prompt_embeds = prompt_embeds

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = noisy_latent

    with torch.no_grad():
        for i, t in enumerate(timesteps):

            noise_pred = pipe.unet(latents, t, encoder_hidden_states = prompt_embeds, return_dict = False)[0]

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict = False)[0]

            if (i+1) % 10 == 0 or i == len(timesteps) - 1: 
                print(f"Step {i+1}/{len(timesteps)}: timestep {t.item()}")
    print ("Denoising complete")


    print("Decoding denoised latent to image")

    with torch.no_grad():
        latents = latents/ pipe.vae.config.scaling_factor
        denoised_image = pipe.vae.decode(latents).sample
        denoised_image = (denoised_image / 2 + 0.5).clamp(0,1)
        denoised_image = denoised_image.cpu().permute(0,2,3,1).numpy()
        denoised_image = (denoised_image * 255).round().astype("uint8")
        denoised_pil = Image.fromarray(denoised_image[0])

        denoised_pil.save("output_denoised.png")
        print("Saved denoised image to output_denoised.png")

  



if __name__ == "__main__":
    main()

