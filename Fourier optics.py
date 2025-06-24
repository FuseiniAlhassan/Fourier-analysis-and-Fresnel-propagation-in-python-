# 📸 Digital Holography: Fourier Analysis and Fresnel Propagation in Python

## 📝 Project Summary
This project explores the reconstruction of digital holograms using Fourier optics principles and Fresnel diffraction simulation. Starting from real `.tif` hologram images, the notebook performs:

- ✅ Square padding and grayscale conversion  
- ✅ 2D Fourier Transform and frequency domain visualization  
- ✅ Frequency filtering via customizable masks  
- ✅ Inverse Fourier reconstruction of complex fields  
- ✅ Fresnel propagation using scalar diffraction theory  
- ✅ Normalized TIFF export of intensity maps

The code is modular and annotated for educational and research reuse, and it serves as a computational optics tool for analyzing interference patterns, simulating diffraction, and comparing holographic quality.

📌 **Use cases:**  
Digital holography, physics/optics education, phase retrieval, beam propagation simulation, and research prototyping in Fourier imaging.

📂 Output: `reconstructed_output.tif`

---

# Author: Alhassan Kpahambang (Refined and Annotated with GPT Support)

# Description: This notebook performs hologram loading, Fourier transform, frequency filtering, and
# Fresnel diffraction propagation on a digital hologram using Python. It visualizes phase and intensity
# maps and saves results for further analysis.

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image
import tifffile
import os

# === STEP 1: Load and Prepare Hologram ===
def load_hologram(image_path):
    """Loads and pads the hologram image to square dimensions."""
    image = np.array(Image.open(image_path).convert('L'))
    max_dim = max(image.shape)
    pad_width = ((0, max_dim - image.shape[0]), (0, max_dim - image.shape[1]))
    padded = np.pad(image, pad_width, mode='constant')
    return padded

# Example hologram file path (replace with actual path if needed)
hologram = load_hologram('/kaggle/input/bak-obj2/objektas3.tif')

plt.title("Loaded Hologram")
plt.imshow(hologram, cmap='gray')
plt.axis('off')
plt.show()

# === STEP 2: Fourier Transform and Spectrum Visualization ===
def display_spectrum(field):
    return np.log(np.abs(field) + 1)

fft = fft2(hologram)
fft_shifted = fftshift(fft)

plt.title("Fourier Spectrum")
plt.imshow(display_spectrum(fft_shifted), cmap='gray')
plt.axis('off')
plt.show()

# === STEP 3: Frequency Filtering with Square Mask ===
def make_square_mask(size, mask_size, center_x, center_y):
    mask = np.zeros((size, size))
    mask[center_y:center_y+mask_size, center_x:center_x+mask_size] = 1
    return mask

def crop_and_center_image(image, mask):
    rows, cols = np.nonzero(mask)
    cropped = image[min(rows):max(rows)+1, min(cols):max(cols)+1]
    centered = np.zeros_like(image)
    start_r = (centered.shape[0] - cropped.shape[0]) // 2
    start_c = (centered.shape[1] - cropped.shape[1]) // 2
    centered[start_r:start_r+cropped.shape[0], start_c:start_c+cropped.shape[1]] = cropped
    return centered

mask = make_square_mask(fft_shifted.shape[0], 1200, 1270, 10)
filtered_freq = crop_and_center_image(fft_shifted, mask)

plt.title("Filtered Spectrum")
plt.imshow(display_spectrum(filtered_freq), cmap='gray')
plt.axis('off')
plt.show()

# === STEP 4: Inverse Transform to Get Complex Field ===
complex_field = ifftshift(filtered_freq)
reconstructed_image = np.abs(ifft2(complex_field))

plt.title("Reconstructed Field Intensity")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()

# === STEP 5: Fresnel Propagation ===
def propagate_field(complex_field, z, ps, lambda0):
    n = complex_field.shape[1]
    grid_size = ps * n
    fx = np.linspace(-(n-1)/2*(1/grid_size), (n-1)/2*(1/grid_size), n)
    fy = np.linspace(-(n-1)/2*(1/grid_size), (n-1)/2*(1/grid_size), n)
    Fx, Fy = np.meshgrid(fx, fy)
    H = np.exp(1j * np.pi * lambda0 * z * (Fx**2 + Fy**2))
    G = fftshift(complex_field) * H
    propagated = np.exp(1j * 2 * np.pi * z / lambda0) / (1j * lambda0 * z) * ifft2(ifftshift(G))
    return propagated

# Constants (adjust as needed)
ps = 10e-6           # pixel size in meters
lambda0 = 632.8e-9   # wavelength in meters (HeNe laser)
z = 0.01             # propagation distance in meters

propagated_field = propagate_field(complex_field, z, ps, lambda0)
intensity = np.abs(propagated_field)

plt.title("Propagated Field Intensity")
plt.imshow(intensity, cmap='gray')
plt.axis('off')
plt.show()

# === STEP 6: Normalize and Save as TIFF ===
def save_tiff_image(array, filename):
    norm = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    uint8_array = norm.astype(np.uint8)
    tifffile.imwrite(filename, uint8_array)

save_tiff_image(intensity, 'reconstructed_output.tif')

print("✅ Process complete. Reconstructed field saved as 'reconstructed_output.tif'")
