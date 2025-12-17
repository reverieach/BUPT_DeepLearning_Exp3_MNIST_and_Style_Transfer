import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Imports
code_1 = """\
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for results
os.makedirs('results', exist_ok=True)
"""

# Cell 2: Data Loading
code_2 = """\
# MNIST dataset
batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaded successfully.")
"""

# Cell 3: Model Definition
code_3 = """\
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Bottleneck layer, dim=2
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()  # Output pixels between 0 and 1
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, 28, 28)
        return encoded, decoded

model = Autoencoder().to(device)
print(model)
"""

# Cell 4: Training Function
code_4 = """\
def train_model(model, train_loader, num_epochs=10, learning_rate=1e-3, denoising=False, noise_factor=0.4):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            
            if denoising:
                noisy_img = img + noise_factor * torch.randn_like(img)
                noisy_img = torch.clamp(noisy_img, 0., 1.)
                inputs = noisy_img
            else:
                inputs = img
            
            # Forward
            _, output = model(inputs)
            loss = criterion(output, img) # Target is always the clean image
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
    return loss_history

# Train Standard AE
print("Training Standard Autoencoder...")
model_standard = Autoencoder().to(device)
loss_standard = train_model(model_standard, train_loader, num_epochs=10, denoising=False)

# Plot Loss
plt.plot(loss_standard)
plt.title('Standard AE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
"""

# Cell 5: Denoising AE Training
code_5 = """\
# Train Denoising AE
print("Training Denoising Autoencoder...")
model_denoising = Autoencoder().to(device)
loss_denoising = train_model(model_denoising, train_loader, num_epochs=10, denoising=True, noise_factor=0.4)

# Plot Loss
plt.plot(loss_denoising)
plt.title('Denoising AE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
"""

# Cell 6: Visualization - Denoising Results
code_6 = """\
# Visualize Original, Noisy, Reconstructed for digits 0-9
def visualize_denoising(model, test_loader):
    model.eval()
    
    # Get one batch
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    # Find one image for each digit 0-9
    digit_indices = []
    for i in range(10):
        idx = (labels == i).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            digit_indices.append(idx[0].item())
        else:
            digit_indices.append(0) # Fallback
            
    selected_images = images[digit_indices]
    
    # Add noise
    noise_factor = 0.4
    noisy_images = selected_images + noise_factor * torch.randn_like(selected_images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    
    # Reconstruct
    with torch.no_grad():
        _, reconstructed = model(noisy_images)
    
    # Plot
    selected_images = selected_images.cpu()
    noisy_images = noisy_images.cpu()
    reconstructed = reconstructed.cpu()
    
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    for i in range(10):
        # Original
        axes[0, i].imshow(selected_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 5: axes[0, i].set_title("Original")
        
        # Noisy
        axes[1, i].imshow(noisy_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 5: axes[1, i].set_title("Noisy")
        
        # Reconstructed
        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 5: axes[2, i].set_title("Reconstructed")
    
    plt.show()

print("Visualizing Denoising Results (0-9)...")
visualize_denoising(model_denoising, test_loader)
"""

# Cell 7: Latent Space Visualization
code_7 = """\
def visualize_latent_space(model, loader):
    model.eval()
    all_encoded = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            encoded, _ = model(images)
            all_encoded.append(encoded.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_encoded = np.concatenate(all_encoded, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_encoded[:, 0], all_encoded[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=2)
    plt.colorbar(scatter)
    plt.title("Latent Space Distribution (Bottleneck dim=2)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.show()

print("Visualizing Latent Space (Standard AE)...")
visualize_latent_space(model_standard, test_loader)
"""

# Cell 8: Uniform Sampling
code_8 = """\
def visualize_sampling(model, range_val=50, n=20):
    model.eval()
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Create a grid of latent variables
    grid_x = np.linspace(-range_val, range_val, n)
    grid_y = np.linspace(-range_val, range_val, n)
    
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
    plt.title(f"Uniform Sampling (Range: -{range_val} to {range_val})")
    plt.axis('off')
    plt.show()

print("Visualizing Uniform Sampling (Standard AE)...")
# Note: You might need to adjust 'range_val' based on the scatter plot above
visualize_sampling(model_standard, range_val=30) 
"""

nb.cells = [
    nbf.v4.new_markdown_cell("# 实验三：基于 MNIST 数据集的自编码器实现\n\n本 Notebook 包含了实验三第一部分的所有内容：\n1. 数据读写与模型搭建\n2. 训练普通自编码器\n3. 训练降噪自编码器并可视化\n4. 潜空间可视化与均匀采样重构"),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell("## 1. 数据加载"),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell("## 2. 模型定义 (Bottleneck dim=2)"),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell("## 3. 训练普通自编码器"),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell("## 4. 训练降噪自编码器 (Noise Factor = 0.4)"),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell("## 5. 结果可视化：降噪效果 (0-9)"),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell("## 6. 结果可视化：潜空间分布"),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_markdown_cell("## 7. 结果可视化：潜空间均匀采样"),
    nbf.v4.new_code_cell(code_8)
]

with open('MNIST_Autoencoder.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
