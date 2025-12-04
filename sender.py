from PIL import Image
import torch
import numpy as np
import struct
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# ============================================
# SENDER: Compress image + Semantic Embedding
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load VAE & CLIP Image Encoder
print("Loading Models...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

# CLIP encoder for SDXL (BigG-14)
encoder_repo = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
print(f"Loading encoder from {encoder_repo}...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    encoder_repo, torch_dtype=torch.float16
).to(device)
clip_processor = CLIPImageProcessor.from_pretrained(encoder_repo)

# 2. Load image
print("Loading image...")
image_path = "before.jpg"
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: {image_path} not found.")
    exit()

original_size = image.size
print(f"Original size: {original_size}")

# 3. Extract Semantic Embedding
print("Extracting semantic embedding...")
with torch.no_grad():
    clip_input = clip_processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.float16)
    image_embeds = image_encoder(clip_input).image_embeds
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    embeds_np = image_embeds.cpu().float().numpy().flatten()

print(f"Embedding shape: {embeds_np.shape}")

# 4. Encode with VAE (SD VAE expects [-1, 1] range)
print("Encoding with VAE...")
image_resized = image.resize((512, 512))
image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image_tensor = image_tensor.to(device)
image_tensor = image_tensor * 2 - 1  # Convert [0,1] to [-1,1]

with torch.no_grad():
    latent_dist = vae.encode(image_tensor)
    latents = latent_dist.latent_dist.sample() * vae.config.scaling_factor

print(f"Latent shape: {latents.shape}")  # (1, 4, 64, 64)

# 5. Quantize to 4-bit
latent_min = latents.min().item()
latent_max = latents.max().item()
latents_normalized = ((latents - latent_min) / (latent_max - latent_min) * 15).round().to(torch.uint8)

latents_flat = latents_normalized.flatten().cpu().numpy()
latents_packed = ((latents_flat[0::2] << 4) | latents_flat[1::2]).astype(np.uint8)

# 6. Save compressed payload
output_file = "compressed_payload.bin"
print(f"Saving to {output_file}...")

with open(output_file, "wb") as f:
    # Header
    f.write(struct.pack('ii', original_size[0], original_size[1]))
    f.write(struct.pack('ff', latent_min, latent_max))
    f.write(struct.pack('i', latents.shape[1]))  # channels (4)
    f.write(struct.pack('i', latents.shape[2]))  # spatial (64)
    f.write(struct.pack('f', vae.config.scaling_factor))  # scaling factor
    
    # Save Embedding Size and data
    f.write(struct.pack('i', len(embeds_np)))
    f.write(embeds_np.tobytes())
    
    # Write Packed Latents
    f.write(latents_packed.tobytes())

import os
file_size = os.path.getsize(output_file)
original_bytes = original_size[0] * original_size[1] * 3
embed_bytes = len(embeds_np) * 4

print(f"\n=== COMPRESSION SUMMARY ===")
print(f"Original image: {original_bytes:,} bytes")
print(f"Image Embedding: {embed_bytes/1024:.2f} KB")
print(f"Packed Latent: {len(latents_packed)/1024:.2f} KB")
print(f"Total Payload: {file_size/1024:.2f} KB")
print(f"Compression Ratio: {original_bytes / file_size:.0f}x")
print(f"\nSaved: {output_file}")
