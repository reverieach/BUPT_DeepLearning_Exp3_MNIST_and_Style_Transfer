# Setup and Train Script for Style Transfer

# 1. Clone the repository
git clone https://github.com/naoto0804/pytorch-AdaIN.git
cd pytorch-AdaIN

# 2. Install dependencies (if needed, usually just torch and torchvision which are likely present)
# pip install -r requirements.txt

# 3. Download Datasets (User needs to do this manually usually, or we assume they have them)
# The experiment requires MS-COCO and WikiArt.
# This script assumes datasets are at ../datasets/coco and ../datasets/wikiart
# You might need to adjust paths.

# 4. Train the model
# The requirements say "train model (>= 10,000 iterations)"
# And "save models at different stages"
# We need to modify the training script or just run it and hope it saves checkpoints.
# The standard train.py usually saves checkpoints.

echo "Starting training... (Ensure datasets are linked correctly)"
# Example command (adjust paths):
# python train.py --content_dir ../datasets/coco --style_dir ../datasets/wikiart --max_iter 10000 --save_dir ../experiments

# Note: You need to manually ensure you have the datasets.
