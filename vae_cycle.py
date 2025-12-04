from PIL import Image
import torch
import numpy as np
from efficientvit.ae_model_zoo import DCAE_HF

# ============================================
# DC-AE ENCODE/DECODE CYCLE + 3-BIT QUANTIZATION
# ~256x compression!
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load DC-AE (MIT Han Lab - high compression VAE)
print("Loading DC-AE...")
vae = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f32c32-in-1.0").to(device).eval()

# 2. Load image
print("Loading lake-michigan.png...")
image = Image.open("lake-michigan.png").convert("RGB")
original_size = image.size
print(f"Original size: {original_size}")

# Resize to 512x512 for VAE
image_resized = image.resize((512, 512), Image.LANCZOS)

# DC-AE expects [0, 1] range (NOT [-1, 1] like SD VAE)
image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image_tensor = image_tensor.to(device)

# 3. Encode
print("Encoding...")
with torch.no_grad():
    latents = vae.encode(image_tensor)

print(f"Latent shape: {latents.shape}")  # (1, 32, 16, 16)

# 4. Quantize to 3-bit (8 levels: 0-7)
print("Quantizing to 3-bit...")
latent_min = latents.min()
latent_max = latents.max()
latents_int3 = ((latents - latent_min) / (latent_max - latent_min) * 7).round().to(torch.uint8)

# Calculate compression with 3-bit packing (8 values per 3 bytes)
packed_size = (latents_int3.numel() * 3) // 8
original_size_bytes = 512 * 512 * 3

print(f"Compression breakdown:")
print(f"  Original image: {original_size_bytes:,} bytes")
print(f"  Latent (float32): {latents.numel() * 4:,} bytes")
print(f"  Latent (3-bit packed): {packed_size:,} bytes")
print(f"  Total compression: {original_size_bytes / packed_size:.0f}x")

# 5. Dequantize back to float
print("Dequantizing...")
latents_dequant = latents_int3.float() / 7.0 * (latent_max - latent_min) + latent_min

# 6. Decode (using dequantized latent)
print("Decoding...")
with torch.no_grad():
    recon = vae.decode(latents_dequant)

# DC-AE outputs [0, 1] already
recon = recon.clamp(0, 1)

# 7. Save output
recon_np = recon.squeeze().permute(1, 2, 0).cpu().numpy()
recon_np = (recon_np * 255).astype(np.uint8)
output_image = Image.fromarray(recon_np)

# Resize back to original size
output_image = output_image.resize(original_size, Image.LANCZOS)
output_image.save("output_dcae.jpg", quality=95)

print("Saved: output_dcae.jpg")
print("Done!")



