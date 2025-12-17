import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import Autoencoder
import os

def add_noise(img, noise_factor=0.4):
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

def visualize_reconstruction(model, dataloader, device, denoising=False):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    if denoising:
        inputs = add_noise(images, 0.4).to(device)
    else:
        inputs = images

    with torch.no_grad():
        _, outputs = model(inputs)
    
    images = images.cpu().numpy()
    inputs = inputs.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2: ax.set_title("Original")

        # Display input (noisy or original)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(inputs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2: ax.set_title("Input (Noisy)" if denoising else "Input")

        # Display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(outputs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2: ax.set_title("Reconstructed")
    
    filename = 'results/denoising_reconstruction.png' if denoising else 'results/simple_reconstruction.png'
    plt.savefig(filename)
    print(f"Saved reconstruction visualization to {filename}")
    plt.close()

def visualize_latent_space(model, dataloader, device):
    model.eval()
    all_encoded = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            encoded, _ = model(images)
            all_encoded.append(encoded.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_encoded = np.concatenate(all_encoded, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_encoded[:, 0], all_encoded[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=2)
    plt.colorbar(scatter)
    plt.title("Latent Space (Bottleneck dim=2)")
    plt.savefig('results/latent_space.png')
    print("Saved latent space visualization to results/latent_space.png")
    plt.close()

def visualize_sampling(model, device):
    model.eval()
    # Uniform sampling in latent space
    # Based on latent space plot, we can estimate range. Usually around -10 to 10 or similar depending on activation.
    # Since we didn't use activation on bottleneck, it's linear.
    # Let's assume a range based on typical distribution or check min/max later. 
    # For now, let's try grid from -30 to 30 (adjust if needed after seeing latent plot)
    
    n = 20
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-50, 50, n) # Adjusted range, might need tuning
    grid_y = np.linspace(-50, 50, n)
    
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                decoded = model.decoder(z_sample)
                digit = decoded[0].cpu().numpy().reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
                       
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title("Uniform Sampling of Latent Space")
    plt.axis('off')
    plt.savefig('results/latent_sampling.png')
    print("Saved latent sampling visualization to results/latent_sampling.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    
    # 1. Visualize Simple AE
    print("--- Visualizing Simple AE ---")
    model = Autoencoder().to(device)
    if os.path.exists('results/simple_ae.pth'):
        model.load_state_dict(torch.load('results/simple_ae.pth'))
        visualize_reconstruction(model, test_loader, device, denoising=False)
        visualize_latent_space(model, test_loader, device)
        visualize_sampling(model, device)
    else:
        print("Simple AE model not found. Run training first.")

    # 2. Visualize Denoising AE
    print("\n--- Visualizing Denoising AE ---")
    model_denoise = Autoencoder().to(device)
    if os.path.exists('results/denoising_ae.pth'):
        model_denoise.load_state_dict(torch.load('results/denoising_ae.pth'))
        visualize_reconstruction(model_denoise, test_loader, device, denoising=True)
    else:
        print("Denoising AE model not found. Run training first.")
