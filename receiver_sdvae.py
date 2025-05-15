"""
receiver_sdvae.py
-----------------
Receives gzip-compressed latent + original image from sender_sdvae.py,
reconstructs the image, and displays both original and reconstruction
side-by-side for visual comparison.
"""
import socket, struct, gzip, numpy as np, time, matplotlib.pyplot as plt
import torch
from diffusers.models import AutoencoderKL

# ------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 65432
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema") \
                   .to(DEVICE).eval()

# helper to read exact n bytes
def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        pkt = conn.recv(n - len(buf))
        if not pkt:
            raise ConnectionError("Socket closed early")
        buf += pkt
    return buf

# ------------------------------------------------------------------
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Receiver listening on {HOST}:{PORT} ...")

    conn, addr = srv.accept()
    print(f"Connected from {addr}")

    # Receive: [len (4B)] [flag (1B)] [h (2B)] [w (2B)]
    meta = recv_exact(conn, 9)
    length, flag, h, w = struct.unpack("!IBHH", meta)

    # Receive: latent payload
    body = recv_exact(conn, length)

    # Receive: original image (assume float32 [0,1], shape 3x256x256)
    raw_image_bytes = recv_exact(conn, 3 * 256 * 256 * 4)  # 786,432 bytes
    conn.close()

# ------------------------------------------------------------------
t0 = time.time()

if flag == 0:  # int8 SD-VAE latent
    lat8 = np.frombuffer(gzip.decompress(body), dtype=np.int8) \
                .reshape(1, 4, h, w)
    latent = torch.from_numpy(lat8.astype("float32") / 127).to(DEVICE)
elif flag == 2:  # float16 SD-VAE latent
    lat16 = np.frombuffer(gzip.decompress(body), dtype=np.float16) \
                .reshape(1, 4, h, w)
    latent = torch.from_numpy(lat16.astype("float32")).to(DEVICE)

# Reconstruct
with torch.no_grad():
    recon = vae.decode(latent).sample

elapsed = time.time() - t0

# Prepare images
recon_img = ((recon.clamp(-1, 1) + 1) / 2).squeeze().permute(1, 2, 0).cpu().numpy()
orig_img = np.frombuffer(raw_image_bytes, dtype=np.float32) \
                .reshape(3, 256, 256).transpose(1, 2, 0)

# Show both images side-by-side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(orig_img)
axs[0].set_title("Original")
axs[0].axis("off")
axs[1].imshow(recon_img)
axs[1].set_title("Reconstructed")
axs[1].axis("off")
plt.tight_layout()
plt.show()

print("\n===== Receiver Report =====")
print(f"Payload size     : {length/1024:.2f} KB")
print(f"Decode + display : {elapsed:.3f} s")
