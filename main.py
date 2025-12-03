"""
Diffusion-Based Image Compression - Sender (Optimization)

This script optimizes a conditioning vector c* that, when used to guide
the denoising process, reconstructs the original image.
"""
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import torch
import torch.nn.functional as F
import os
import time
from PIL import Image
import numpy as np


def main():
    print("="*60)
    print(" Diffusion-Based Image Compression - Sender")
    print("="*60)

    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # For UNet (faster)
        vae_dtype = torch.float16  # For VAE
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else: 
        print("CUDA not available, using CPU (will be very slow)")
        device = "cpu"
        dtype = torch.float32
        vae_dtype = torch.float32

    # =========================================================================
    # CONFIGURATION - Must match between sender and receiver!
    # =========================================================================
    model_id = "stabilityai/stable-diffusion-2"
    vae_model_id = "stabilityai/sd-vae-ft-mse"
    shared_seed = 42
    image_size = 768
    num_inference_steps = 50
    
    # Optimization parameters
    num_iterations = 50  # Number of optimization steps (increase for better quality)
    learning_rate = 0.05
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Denoising steps: {num_inference_steps}")
    print(f"  Optimization iterations: {num_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Shared seed: {shared_seed}")

    # =========================================================================
    # LOAD MODELS
    # =========================================================================
    print("\n[1/6] Loading models...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # Load improved VAE
    print(f"Loading improved VAE: {vae_model_id}")
    pipe.vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=vae_dtype).to(device)
    
    # Freeze all model parameters - we only optimize the conditioning!
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    print("Models loaded and frozen")

    # Replace scheduler with DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # =========================================================================
    # LOAD AND ENCODE ORIGINAL IMAGE
    # =========================================================================
    print("\n[2/6] Loading and encoding image...")
    
    image = Image.open("output.png").convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.LANCZOS)
    
    # Convert to tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)  # Keep float32 for loss
    
    # Encode to latent space
    image_for_vae = (image_tensor * 2.0 - 1.0).to(dtype=vae_dtype)
    
    with torch.no_grad():
        latent_dist = pipe.vae.encode(image_for_vae).latent_dist
        original_latent = latent_dist.mode() * pipe.vae.config.scaling_factor

    print(f"Original image shape: {image_tensor.shape}")
    print(f"Latent shape: {original_latent.shape}")

    # =========================================================================
    # GENERATE NOISY LATENT (Deterministic with shared seed)
    # =========================================================================
    print("\n[3/6] Generating noisy latent...")
    
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_seed)
    noise = torch.randn(original_latent.shape, generator=generator, device=device, dtype=dtype)

    max_timestep = pipe.scheduler.timesteps[0]
    noisy_latent = pipe.scheduler.add_noise(
        original_latent.to(dtype=dtype),
        noise,
        max_timestep
    )
    
    print(f"Max timestep: {max_timestep}")
    print(f"Noisy latent shape: {noisy_latent.shape}")

    # =========================================================================
    # INITIALIZE LEARNABLE CONDITIONING
    # =========================================================================
    print("\n[4/6] Initializing learnable conditioning...")
    
    # SD 2.x uses 1024 hidden dim, 77 tokens
    conditioning_shape = (1, 77, 1024)
    
    # Option 1: Random initialization
    conditioning = torch.randn(
        conditioning_shape,
        device=device,
        dtype=torch.float32,  # Float32 for optimization stability
        requires_grad=True
    )
    
    # Option 2: Initialize from text (sometimes converges faster)
    # Uncomment to use:
    # with torch.no_grad():
    #     text_inputs = pipe.tokenizer(
    #         "high quality photograph",
    #         padding="max_length",
    #         max_length=77,
    #         return_tensors="pt"
    #     ).to(device)
    #     conditioning = pipe.text_encoder(text_inputs.input_ids)[0].clone()
    #     conditioning = conditioning.to(dtype=torch.float32).requires_grad_(True)
    
    print(f"Conditioning shape: {conditioning.shape}")
    print(f"Conditioning requires_grad: {conditioning.requires_grad}")

    # =========================================================================
    # OPTIMIZATION LOOP - THE KEY PART!
    # =========================================================================
    print("\n[5/6] Starting optimization...")
    print("="*70)
    
    optimizer = torch.optim.Adam([conditioning], lr=learning_rate)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=learning_rate * 0.1
    )
    
    best_loss = float('inf')
    best_conditioning = None
    losses = []
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        optimizer.zero_grad()
        
        # ----- Forward pass: Denoise with current conditioning -----
        latents = noisy_latent.clone().to(dtype=dtype)
        cond = conditioning.to(dtype=dtype)
        
        # Denoising loop (must be differentiable - NO torch.no_grad!)
        for t in pipe.scheduler.timesteps:
            noise_pred = pipe.unet(
                latents,
                t,
                encoder_hidden_states=cond,
                return_dict=False
            )[0]
            
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # ----- Decode to image space -----
        latents_for_decode = latents.to(dtype=vae_dtype) / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents_for_decode).sample
        reconstructed = (decoded.float() / 2 + 0.5).clamp(0, 1)
        
        # ----- Compute loss -----
        loss = F.mse_loss(reconstructed, image_tensor)
        
        # ----- Backward pass -----
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([conditioning], max_norm=1.0)
        
        optimizer.step()
        scheduler_lr.step()
        
        # ----- Track progress -----
        loss_val = loss.item()
        losses.append(loss_val)
        
        improved = ""
        if loss_val < best_loss:
            improvement = (best_loss - loss_val) / best_loss * 100 if best_loss != float('inf') else 100
            best_loss = loss_val
            best_conditioning = conditioning.clone().detach()
            improved = f" ⬇ NEW BEST! (-{improvement:.1f}%)"
        
        iter_time = time.time() - iter_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Iter {iteration+1:3d}/{num_iterations}] "
              f"Loss: {loss_val:.6f} | "
              f"Best: {best_loss:.6f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {iter_time:.1f}s"
              f"{improved}")
    
    total_time = time.time() - start_time
    print("="*70)
    print(f"Optimization complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/num_iterations:.1f}s per iteration)")
    print(f"Final best loss: {best_loss:.6f}")
    print(f"Loss reduction: {(losses[0] - best_loss) / losses[0] * 100:.1f}%")

    # =========================================================================
    # SAVE COMPRESSED DATA
    # =========================================================================
    print("\n[6/6] Saving compressed data...")
    
    # Save the optimized conditioning (this IS the compressed file!)
    conditioning_np = best_conditioning.cpu().numpy().astype(np.float16)
    
    # Save as .npz with metadata
    np.savez_compressed(
        "compressed.npz",
        conditioning=conditioning_np,
        # Metadata for receiver
        model_id=model_id,
        vae_model_id=vae_model_id,
        shared_seed=shared_seed,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        final_loss=best_loss
    )
    
    # Calculate compression ratio
    original_size = image_size * image_size * 3  # RGB uint8
    compressed_size = conditioning_np.nbytes
    ratio = original_size / compressed_size
    
    print(f"\nCompression Results:")
    print(f"  Original size: {original_size / 1024:.1f} KB ({image_size}x{image_size} RGB)")
    print(f"  Compressed size: {compressed_size / 1024:.1f} KB (conditioning vector)")
    print(f"  Compression ratio: {ratio:.1f}x")
    print(f"  Saved to: compressed.npz")
    
    # =========================================================================
    # GENERATE FINAL RECONSTRUCTION FOR COMPARISON
    # =========================================================================
    print("\nGenerating final reconstruction...")
    
    with torch.no_grad():
        # Regenerate using best conditioning
        latents = noisy_latent.clone().to(dtype=dtype)
        cond = best_conditioning.to(dtype=dtype)
        
        for t in pipe.scheduler.timesteps:
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=cond, return_dict=False)[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        latents = latents.to(dtype=vae_dtype) / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents).sample
        final_image = (decoded / 2 + 0.5).clamp(0, 1)
        
        # Save reconstructed image
        final_np = final_image[0].permute(1, 2, 0).cpu().numpy()
        final_np = (final_np * 255).astype(np.uint8)
        Image.fromarray(final_np).save("reconstructed.png")
        
        # Save original (resized) for comparison
        orig_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        orig_np = (orig_np * 255).astype(np.uint8)
        Image.fromarray(orig_np).save("original_resized.png")
    
    # Calculate quality metrics
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    orig_for_metrics = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    recon_for_metrics = final_np
    
    psnr = peak_signal_noise_ratio(orig_for_metrics, recon_for_metrics)
    ssim = structural_similarity(orig_for_metrics, recon_for_metrics, channel_axis=2)
    
    print(f"\nQuality Metrics:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    print(f"\nOutput files:")
    print(f"  • compressed.npz      - Compressed data (send this!)")
    print(f"  • original_resized.png - Original image")
    print(f"  • reconstructed.png   - Reconstructed image")
    
    print("\n" + "="*60)
    print(" Compression complete! ")
    print("="*60)


if __name__ == "__main__":
    main()


