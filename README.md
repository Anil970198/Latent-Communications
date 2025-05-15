# 🔄 Latent Communications — Deep Image Transmission Using Autoencoders

This project presents a bandwidth-efficient image transmission framework that leverages deep latent representations learned by pretrained autoencoders. Instead of transmitting raw image data over the network, the system compresses the image into a low-dimensional latent vector using a pretrained Variational Autoencoder (VAE), transmits it over a TCP connection, and reconstructs the original image on the receiver side using the decoder.

This approach reduces transmission size while preserving visual quality, offering a practical method for real-time, resource-constrained communication scenarios.

---

## 📌 Project Objectives

* Utilize a pretrained **Variational Autoencoder (VAE)** for semantic compression.
* Transmit only the **compressed latent representation** rather than full-resolution images.
* Reduce **bandwidth and latency** during image transmission.
* Reconstruct and visualize original vs. decoded image side-by-side on the receiver.
* Lay the groundwork for **multimodal and real-time transmission systems**.

---

## 🧱 Project File Structure

![Architecture Diagram](https://github.com/AnilGit9701/Latent-Communications/blob/7c66ce7242b464ee53b16f6d55fb1997fe48ff8d/arch.png)

```text
Latent-Communications/
├── sender_sdvae.py                 # Encodes image to latent, compresses, and sends via socket
├── receiver_sdvae.py              # Receives latent, decompresses, decodes, and displays images
├── sender_sdvae_metrics.py        # Optional: logs and compares payload vs. original image
├── requirements.txt               # All required Python libraries
├── README.md                      # Project overview and setup instructions
├── temp/
│   └── apple.jpeg                 # Sample test image (256x256 RGB)
├── models/
│   ├── vqgan_imagenet_f16_16384.yaml      # Required config (optional VQGAN extension)
│   └── taming-transformers/               # (Optional) Git submodule or helper code
└── gfpgan/
    └── weights/                           # Store optional downloaded weights here
```

---

## 📦 Software Requirements

Install all required Python libraries with:

```bash
pip install -r requirements.txt
```

Ensure your environment includes:

* Python 3.10+
* PyTorch (with support for `cuda`, `cpu`, or `mps`)
* diffusers
* numpy, Pillow, matplotlib
* Standard libraries: `socket`, `struct`, `gzip` (no need to install separately)

---

## 📁 Required External Files (Must Be Downloaded Separately)

GitHub restricts uploads of files larger than 100MB. To keep the repository lightweight, you’ll need to download these files manually if you plan to use optional modules:

### 🧠 1. Stable Diffusion VAE (AutoencoderKL)

Used for compressing and reconstructing images.

```python
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
```

✅ Downloads automatically the first time you run the code.

---

### 🧠 2. VQGAN Checkpoint (Optional — for future expansion)

For using VQGAN-based compression instead of VAEs:

* Download: [`vqgan_imagenet_f16_16384.ckpt`](https://heibox.uni-heidelberg.de/f/140747ba53464f318eaf/?dl=1)
* Place in:

```bash
models/vqgan_imagenet_f16_16384.ckpt
```

* Also download the config file:

```bash
models/vqgan_imagenet_f16_16384.yaml
```

---

### 🧠 3. GFPGAN Weights (Optional — for post-processing and face enhancement)

* [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases)
* [detection\_Resnet50\_Final.pth](https://github.com/TencentARC/GFPGAN)
* [parsing\_parsenet.pth](https://github.com/TencentARC/GFPGAN)

Place them inside:

```bash
gfpgan/weights/
```

---

## 🚀 How to Run the System

### 1. Start the receiver:

```bash
python receiver_sdvae.py
```

* Opens socket on `127.0.0.1:65432`
* Waits for incoming compressed latent data

### 2. Then run the sender:

```bash
python sender_sdvae.py
```

* Loads image from `temp/apple.jpeg`
* Compresses using pretrained VAE
* Sends latent + original image to receiver

### ✅ Output:

![Original vs Reconstructed Output](https://github.com/AnilGit9701/Latent-Communications/blob/7c66ce7242b464ee53b16f6d55fb1997fe48ff8d/results.png)

* Matplotlib display:

  * Left: Original image
  * Right: Reconstructed image
* Console output:

  * Payload size
  * Decode + display time

---

## 🧪 Example Use Case

| Mode              | Size       |
| ----------------- | ---------- |
| Raw image (PNG)   | \~196 KB   |
| Compressed latent | \~12–20 KB |

Over 10× compression achieved with minimal visual loss.

Ideal for:

* IoT communication
* Remote image monitoring
* Real-time robotics vision

---

## 🌱 Future Scope

* Extend to multimodal compression (image + text)
* Add real-time encryption of latent vectors
* Optimize models for edge devices
* Enable adaptive compression based on bandwidth
* Web or mobile dashboard integration

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute with credit.

---

## 🙌 Acknowledgements

* 🫧 [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* 🧠 [Stable Diffusion VAE (AutoencoderKL)](https://huggingface.co/stabilityai/sd-vae-ft-ema)
* 🌀 [VQGAN by CompVis](https://github.com/CompVis/taming-transformers)
* 🔧 [GFPGAN by TencentARC](https://github.com/TencentARC/GFPGAN)

---

## ✉️ Contact

For questions or suggestions, feel free to open an issue or contact [@Anil970198](https://github.com/Anil970198).
![arch.png](../../PyCharm%20Projects/encoders/arch.png)
![results.png](../../PyCharm%20Projects/encoders/results.png)
