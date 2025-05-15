import os, time, socket, struct, gzip, torch
from PIL import Image
import torchvision.transforms as T
from diffusers.models import AutoencoderKL

# ------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 65432
IMAGE_PATH = "denseresidential97.tif"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE).eval()

# Load and preprocess image
img = Image.open(IMAGE_PATH).convert("RGB").resize((256, 256))
tensor = T.ToTensor()(img).unsqueeze(0).to(DEVICE) * 2 - 1  # (1,3,256,256)

# Save original image as numpy array [0,1]
original_image_np = T.ToTensor()(img).numpy()  # (3,256,256), dtype float32

# Encode to latent
t_enc = time.time()
with torch.no_grad():
    latent = vae.encode(tensor).latent_dist.mean
encode_time = time.time() - t_enc

# Compress latent as float16 + gzip
lat16 = latent.cpu().half().numpy().tobytes()
payload = gzip.compress(lat16)
header = struct.pack("!IBHH", len(payload), 2, 32, 32)  # flag 2 = f16

# Send both latent + original image
t_send = time.time()
with socket.create_connection((HOST, PORT)) as sock:
    sock.sendall(header + payload)
    sock.sendall(original_image_np.astype("float32").tobytes())
send_time = time.time() - t_send

# Report
orig_kb = os.path.getsize(IMAGE_PATH) / 1024
pay_kb = len(payload) / 1024
print("\n===== Sender Report =====")
print(f"Original file size : {orig_kb:.2f} KB")
print(f"Latent payload     : {pay_kb:.2f} KB")
print(f"Compression ratio  : {orig_kb / pay_kb:.1f}×")
print(f"Encoding time      : {encode_time:.3f} s")
print(f"Transmission time  : {send_time:.3f} s")
