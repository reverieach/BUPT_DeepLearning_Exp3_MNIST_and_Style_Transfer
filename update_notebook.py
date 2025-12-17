
# Cell 9: Standard AE Reconstruction (0-9)
code_9 = """\
# Visualize Original vs Reconstructed for digits 0-9 (Standard AE)
def visualize_standard_reconstruction(model, test_loader):
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
    
    # Reconstruct
    with torch.no_grad():
        _, reconstructed = model(selected_images)
    
    # Plot
    selected_images = selected_images.cpu()
    reconstructed = reconstructed.cpu()
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        # Original
        axes[0, i].imshow(selected_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 5: axes[0, i].set_title("Original")
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 5: axes[1, i].set_title("Reconstructed")
    
    plt.show()

print("Visualizing Standard AE Reconstruction (0-9)...")
visualize_standard_reconstruction(model_standard, test_loader)
"""

# Append to notebook
import nbformat as nbf
nb = nbf.read('MNIST_Autoencoder/MNIST_Autoencoder.ipynb', as_version=4)
nb.cells.append(nbf.v4.new_markdown_cell("## 8. 补充：普通自编码器重构效果 (0-9)"))
nb.cells.append(nbf.v4.new_code_cell(code_9))

with open('MNIST_Autoencoder/MNIST_Autoencoder.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated with Standard AE visualization.")
