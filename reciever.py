from PIL import Image
import torch
import numpy as np
import struct
from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline

# ============================================
# RECEIVER: Reconstruct image
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load compressed payload
input_file = "compressed_payload.bin"
print(f"Loading {input_file}...")

with open(input_file, "rb") as f:
    orig_w, orig_h = struct.unpack('ii', f.read(8))
    latent_min, latent_max = struct.unpack('ff', f.read(8))
    num_channels = struct.unpack('i', f.read(4))[0]
    spatial_size = struct.unpack('i', f.read(4))[0]
    scaling_factor = struct.unpack('f', f.read(4))[0]
    
    # Read embedding
    embed_size = struct.unpack('i', f.read(4))[0]
    embeds_np = np.frombuffer(f.read(embed_size * 4), dtype=np.float32)
    
    # Read packed latent
    latents_packed = np.frombuffer(f.read(), dtype=np.uint8)

print(f"Original size: {orig_w}x{orig_h}")
print(f"Latent shape: (1, {num_channels}, {spatial_size}, {spatial_size})")
print(f"Scaling factor: {scaling_factor}")
print(f"Embedding size: {embed_size}")

# 2. Unpack and dequantize
latents_unpacked = np.zeros(len(latents_packed) * 2, dtype=np.uint8)
latents_unpacked[0::2] = latents_packed >> 4
latents_unpacked[1::2] = latents_packed & 0x0F

latents_int4 = torch.from_numpy(latents_unpacked).reshape(1, num_channels, spatial_size, spatial_size).to(device)
latents_dequant = latents_int4.float() / 15.0 * (latent_max - latent_min) + latent_min

# 3. VAE decode
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

print("Decoding with VAE...")
with torch.no_grad():
    recon = vae.decode(latents_dequant / scaling_factor).sample

# Convert from [-1, 1] to [0, 1]
recon = (recon + 1) / 2
recon = recon.clamp(0, 1)

recon_image = recon.squeeze().permute(1, 2, 0).cpu().numpy()
recon_image = (recon_image * 255).astype(np.uint8)
recon_pil = Image.fromarray(recon_image)
recon_pil = recon_pil.resize((orig_w, orig_h), Image.LANCZOS)
recon_pil.save("step1_vae_decoded.png")
print("Saved: step1_vae_decoded.png")

# 4. Diffusion enhancement with CLIP embedding
print("\nLoading diffusion model...")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

# Load IP-Adapter for image conditioning
print("Loading IP-Adapter...")
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)
pipe.set_ip_adapter_scale(0.5)

# Prepare embedding for IP-Adapter
clip_embedding = torch.from_numpy(embeds_np).unsqueeze(0).to(device, dtype=torch.float16)
# IP-Adapter expects [negative, positive] concatenated
negative_embedding = torch.zeros_like(clip_embedding)
combined_embedding = torch.cat([negative_embedding, clip_embedding], dim=0).unsqueeze(1)

print("Enhancing with diffusion model...")
enhanced = pipe(
    prompt="",
    negative_prompt="",
    image=recon_pil.resize((1024, 1024)),
    ip_adapter_image_embeds=[combined_embedding],
    strength=0.35,
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]

enhanced = enhanced.resize((orig_w, orig_h), Image.LANCZOS)
enhanced.save("step2_diffusion_enhanced.png")
print("Saved: step2_diffusion_enhanced.png")

print("\n=== RECONSTRUCTION COMPLETE ===")
print("Output files:")
print("  - step1_vae_decoded.png (VAE only)")
print("  - step2_diffusion_enhanced.png (VAE + Diffusion)")
