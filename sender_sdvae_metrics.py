"""
sender_sdvae_metrics.py
-----------------------
• Encodes an image with Stable‑Diffusion’s pre‑trained VAE
• Quantises latent to float‑16, gzip‑compresses, and sends to receiver
• Prints PSNR & SSIM so you can claim "little loss"
• Command‑line flags let you choose the image and host/port

Usage examples:
    python sender_sdvae_metrics.py --image face.jpg
    python sender_sdvae_metrics.py --image sat.jpg --host 127.0.0.1 --port 7000
"""

import os, time, socket, struct, gzip, argparse
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from diffusers.models import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ------------------------------------------------------------------
# 1) CLI ARGUMENTS
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Send SD‑VAE latent with metrics")
parser.add_argument("--image", default="hd-image.jpeg", help="Path to input image")
parser.add_argument("--host",  default="127.0.0.1",  help="Receiver host")
parser.add_argument("--port",  type=int, default=65432, help="Receiver port")
args = parser.parse_args()

HOST, PORT   = args.host, args.port
IMAGE_PATH   = args.image
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

# ------------------------------------------------------------------
# 2) LOAD PRE‑TRAINED VAE  (40 MB download first run)
# ------------------------------------------------------------------
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE).eval()

# ------------------------------------------------------------------
# 3) PRE‑PROCESS  (auto‑resize long side -> 256, keep aspect)
# ------------------------------------------------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img = ImageOps.fit(img, (256, 256), method=Image.Resampling.LANCZOS)
tensor = T.ToTensor()(img).unsqueeze(0).to(DEVICE) * 2 - 1  # (1,3,256,256)

# ------------------------------------------------------------------
# 4) ENCODE
# ------------------------------------------------------------------
t_enc = time.time()
with torch.no_grad():
    latent = vae.encode(tensor).latent_dist.mean  # float32, (1,4,32,32)
encode_time = time.time() - t_enc

# ------------------------------------------------------------------
# 5) LOCAL RECONSTRUCTION -> METRICS
# ------------------------------------------------------------------
with torch.no_grad():
    recon_local = vae.decode(latent).sample

orig = ((tensor.squeeze().cpu() + 1) / 2).numpy()          # (3,H,W) in [0,1]
reco = ((recon_local.squeeze().cpu().clamp(-1,1) + 1) / 2).numpy()

psnr_val = psnr(orig, reco, data_range=1)
ssim_val = ssim(
    orig.transpose(1, 2, 0),
    reco.transpose(1, 2, 0),
    data_range=1.0,
    channel_axis=-1
)

print(f"Quality metrics   : PSNR {psnr_val:.2f} dB  |  SSIM {ssim_val:.3f}")

# ------------------------------------------------------------------
# 6) FLOAT16  +  GZIP  +  HEADER
# ------------------------------------------------------------------
lat16   = latent.cpu().half().numpy().tobytes()          # float16
payload = gzip.compress(lat16)                          # ~15 KB
header  = struct.pack("!IBHH", len(payload), 2, 32, 32) # flag 2 = f16 SD‑VAE

# ------------------------------------------------------------------
# 7) SEND
# ------------------------------------------------------------------
t_send = time.time()
with socket.create_connection((HOST, PORT), timeout=10) as sock:
    sock.sendall(header + payload)
send_time = time.time() - t_send

# ------------------------------------------------------------------
# 8) REPORT
# ------------------------------------------------------------------
orig_kb = os.path.getsize(IMAGE_PATH) / 1024
pay_kb  = len(payload) / 1024
print("\n===== Sender Report =====")
print(f"Original file size : {orig_kb:.2f} KB")
print(f"Latent payload     : {pay_kb:.2f} KB")
print(f"Compression ratio  : {orig_kb / pay_kb:.1f}×")
print(f"Encoding time      : {encode_time:.3f} s")
print(f"Transmission time  : {send_time:.3f} s")
