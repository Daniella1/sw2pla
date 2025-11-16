import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compress_image(image_path, k):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_matrix = np.array(img)  # Convert to numpy array
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(img_matrix, full_matrices=False)
    
    # Reduce the rank by keeping only the top k singular values
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # Reconstruct the image from the compressed matrices
    compressed_img_matrix = np.dot(U_k, np.dot(S_k, Vt_k))
    
    # Convert the reconstructed image back to a PIL image and save it
    compressed_img = Image.fromarray(np.uint8(compressed_img_matrix))
    compressed_img.save('compressed_image.png')
    
    # Show the original and compressed images side by side for comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_matrix, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(compressed_img_matrix, cmap='gray')
    axes[1].set_title(f'Compressed Image (k={k})')
    axes[1].axis('off')
    
    plt.show()

# Example usage:
image_path = 'image_muffin_square.png'
k = 50  # Set the rank (number of singular values) for compression
compress_image(image_path, k)
