import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

# Note: This script assumes 'pytorch-AdaIN' is cloned in the same directory
# and we can import from it or use its model definition.
# Since we might not be able to easily import from a subdir without sys.path hacks,
# I will assume the user runs this FROM the pytorch-AdaIN directory or I'll add it to path.

import sys
sys.path.append('pytorch-AdaIN')

# Try importing model from the repo. 
# If this fails, the user needs to clone the repo first.
try:
    import net
    from function import adaptive_instance_normalization, coral
except ImportError:
    print("Error: Could not import 'net' from pytorch-AdaIN.")
    print("Please ensure you have cloned the repository into 'style_transfer/pytorch-AdaIN'")
    print("and run this script from 'style_transfer/' directory.")
    sys.exit(1)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, help='File path to the content image')
    parser.add_argument('--style', type=str, help='File path to the style image')
    parser.add_argument('--vgg', type=str, default='pytorch-AdaIN/models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='pytorch-AdaIN/models/decoder.pth')
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--student_id', type=str, default=None, help='Student ID for alpha blending task')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(512, False)
    style_tf = test_transform(512, False)

    # Task 3: BUPT Scenic Spots (User provides 2 content, 1 style)
    # This script handles one pair at a time. User should run it twice.

    # Task 4: Alpha Blending
    if args.student_id:
        last_digit = int(args.student_id[-1])
        if last_digit % 2 == 0:
            alphas = [0.3, 0.6, 0.9]
            print(f"Student ID {args.student_id} ends in even number. Alphas: {alphas}")
        else:
            alphas = [0.2, 0.5, 0.8]
            print(f"Student ID {args.student_id} ends in odd number. Alphas: {alphas}")
        
        content = content_tf(Image.open(args.content)).unsqueeze(0).to(device)
        style = style_tf(Image.open(args.style)).unsqueeze(0).to(device)
        
        for a in alphas:
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, a)
            
            output_name = f"{os.path.splitext(args.output)[0]}_alpha_{a}.png"
            save_image(output, output_name)
            print(f"Saved {output_name}")
            
    else:
        # Standard Transfer
        content = content_tf(Image.open(args.content)).unsqueeze(0).to(device)
        style = style_tf(Image.open(args.style)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, args.alpha)

        save_image(output, args.output)
        print(f"Saved {args.output}")

def generate_training_progress(checkpoints_dir, output_dir):
    """
    Generates style transfer results for 10%, 50%, 80%, 100% of 10000 iterations.
    Uses default images: input/content/cornell.jpg and input/style/woman_with_hat_matisse.jpg
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = net.vgg
    vgg.load_state_dict(torch.load('pytorch-AdaIN/models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    vgg.eval()

    decoder = net.decoder
    decoder.to(device)
    decoder.eval()

    # Paths from repo structure
    content_path = 'pytorch-AdaIN/input/content/cornell.jpg'
    style_path = 'pytorch-AdaIN/input/style/woman_with_hat_matisse.jpg'
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("Error: Default images not found in pytorch-AdaIN/input/")
        return

    content_tf = test_transform(512, False)
    style_tf = test_transform(512, False)
    
    content = content_tf(Image.open(content_path)).unsqueeze(0).to(device)
    style = style_tf(Image.open(style_path)).unsqueeze(0).to(device)

    # Required iterations for 10000 total
    iters = [1000, 5000, 8000, 10000]
    percents = ["10%", "50%", "80%", "100%"]

    for it, pct in zip(iters, percents):
        ckpt_name = f'decoder_iter_{it}.pth.tar'
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_name} not found. Skipping {pct}.")
            continue
            
        try:
            decoder.load_state_dict(torch.load(ckpt_path))
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, 1.0)
            
            out_name = os.path.join(output_dir, f'training_progress_{pct}.png')
            save_image(output, out_name)
            print(f"Saved {out_name} (Iteration {it})")
        except Exception as e:
            print(f"Error processing {ckpt_name}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='transfer', choices=['transfer', 'progress'], help='Mode: transfer or progress')
    parser.add_argument('--content', type=str, help='File path to the content image')
    parser.add_argument('--style', type=str, help='File path to the style image')
    parser.add_argument('--vgg', type=str, default='pytorch-AdaIN/models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='pytorch-AdaIN/models/decoder.pth')
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--student_id', type=str, default=None, help='Student ID for alpha blending task')
    parser.add_argument('--checkpoints_dir', type=str, default='pytorch-AdaIN/experiments', help='Dir with training checkpoints')
    
    args = parser.parse_args()

    if args.mode == 'progress':
        generate_training_progress(args.checkpoints_dir, '.')
    else:
        # Existing main logic for single transfer
        main()
