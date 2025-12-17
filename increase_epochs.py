import nbformat as nbf

notebook_path = 'MNIST_Autoencoder/MNIST_Autoencoder.ipynb'
nb = nbf.read(notebook_path, as_version=4)

changed = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Find the cell calling train_model
        if 'train_model(' in cell.source and 'num_epochs=10' in cell.source:
            # Replace 10 with 50 for better convergence
            cell.source = cell.source.replace('num_epochs=10', 'num_epochs=50')
            changed = True

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Successfully increased epochs to 50 in the notebook.")
else:
    print("Could not find the target code to update, or it was already updated.")
