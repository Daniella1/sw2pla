import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def compress_image_rgb(image_path, k):
    # Load the image (RGB mode)
    img = Image.open(image_path)
    
    # Convert to numpy array (shape: height x width x 3 for RGB)
    img_matrix = np.array(img)
    
    # Split into R, G, B channels
    R = img_matrix[:, :, 0]
    G = img_matrix[:, :, 1]
    B = img_matrix[:, :, 2]
    
    # Perform SVD on each channel
    U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
    
    # Reduce the rank (number of singular values) to k
    U_r_k, S_r_k, Vt_r_k = U_r[:, :k], np.diag(S_r[:k]), Vt_r[:k, :]
    U_g_k, S_g_k, Vt_g_k = U_g[:, :k], np.diag(S_g[:k]), Vt_g[:k, :]
    U_b_k, S_b_k, Vt_b_k = U_b[:, :k], np.diag(S_b[:k]), Vt_b[:k, :]
    
    # Reconstruct each channel
    R_compressed = np.dot(U_r_k, np.dot(S_r_k, Vt_r_k))
    G_compressed = np.dot(U_g_k, np.dot(S_g_k, Vt_g_k))
    B_compressed = np.dot(U_b_k, np.dot(S_b_k, Vt_b_k))
    
    # Merge the channels back into a single RGB image
    compressed_img_matrix = np.stack([R_compressed, G_compressed, B_compressed], axis=-1)
    
    # Clip the values to be in the valid range [0, 255] and convert to uint8
    compressed_img_matrix = np.clip(compressed_img_matrix, 0, 255).astype(np.uint8)
    
    # Create a PIL image from the compressed matrix
    compressed_img = Image.fromarray(compressed_img_matrix)
    compressed_img.save('compressed_rgb_image.png')

    # Show the original and compressed images side by side for comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title('Original Image (14.41 MB)')
    axes[0].axis('off')

    axes[1].imshow(compressed_img)
    axes[1].set_title(f'Compressed Image (5.85 MB)')
    axes[1].axis('off')

    plt.show()

    # Calculate and display the file size difference
    original_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    compressed_size = os.path.getsize('compressed_rgb_image.png') / (1024 * 1024)  # MB
    print(f"Original image size: {original_size:.2f} MB")
    print(f"Compressed image size: {compressed_size:.2f} MB")
    print(f"Size reduction: {original_size - compressed_size:.2f} MB")

# Example usage:
image_path = 'image_muffin_square.png'
k = 100  # Set the rank (number of singular values) for compression
compress_image_rgb(image_path, k)
