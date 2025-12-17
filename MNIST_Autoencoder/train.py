import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from model import Autoencoder

def add_noise(img, noise_factor=0.4):
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 20
    noise_factor = 0.4

    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Denoising or Standard AE
            if args.denoising:
                noisy_images = add_noise(images, noise_factor).to(device)
                inputs = noisy_images
            else:
                inputs = images
            
            # Forward pass
            _, outputs = model(inputs)
            loss = criterion(outputs, images) # Target is always clean image
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    os.makedirs('results', exist_ok=True)
    model_path = 'results/denoising_ae.pth' if args.denoising else 'results/simple_ae.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoising', action='store_true', help='Train Denoising Autoencoder')
    args = parser.parse_args()
    train(args)
