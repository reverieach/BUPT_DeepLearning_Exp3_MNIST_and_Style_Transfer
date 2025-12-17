import os
import subprocess
import sys

def check_models():
    models_dir = os.path.join("pytorch-AdaIN", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    decoder = os.path.join(models_dir, "decoder.pth")
    vgg = os.path.join(models_dir, "vgg_normalised.pth")
    
    missing = []
    if not os.path.exists(decoder): missing.append("decoder.pth")
    if not os.path.exists(vgg): missing.append("vgg_normalised.pth")
    
    if missing:
        print("!" * 60)
        print(f"MISSING MODELS: {', '.join(missing)}")
        print(f"Please download them from: https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0")
        print(f"And place them in: {os.path.abspath(models_dir)}")
        print("!" * 60)
        return False
    return True

def run_task_3(bupt_images, style_image):
    print("\n[Running Task 3: BUPT Scenic Spot Transfer]")
    for i, img_path in enumerate(bupt_images):
        if not os.path.exists(img_path):
            print(f"Error: Content image not found: {img_path}")
            continue
            
        output_name = f"result_bupt{i+1}.png"
        cmd = [
            sys.executable, "custom_tasks.py",
            "--content", img_path,
            "--style", style_image,
            "--output", output_name
        ]
        print(f"Processing {img_path} with style {style_image}...")
        subprocess.run(cmd)

def run_task_4(bupt_image, style_image, student_id):
    print("\n[Running Task 4: Alpha Blending]")
    if not os.path.exists(bupt_image):
        print(f"Error: Content image not found: {bupt_image}")
        return

    output_name = "alpha_test.png"
    cmd = [
        sys.executable, "custom_tasks.py",
        "--content", bupt_image,
        "--style", style_image,
        "--student_id", student_id,
        "--output", output_name
    ]
    print(f"Processing Alpha Blending for ID {student_id}...")
    subprocess.run(cmd)

def run_training():
    print("\n[Running Training Task]")
    # Hardcoded paths as requested
    base_dir = r"D:\study\深度学习\实验三\datasets"
    coco_path = os.path.join(base_dir, "coco")
    wikiart_path = os.path.join(base_dir, "wikiart", "archive")
    
    print(f"Using hardcoded dataset paths:")
    print(f"  Content (COCO):   {coco_path}")
    print(f"  Style (WikiArt):  {wikiart_path}")
    
    if not os.path.exists(coco_path):
        print(f"Warning: COCO path does not exist: {coco_path}")
        # Fallback to checking just 'datasets' if they put images there, but warn
        if os.path.exists(base_dir):
             print(f"Falling back to base datasets dir: {base_dir}")
             coco_path = base_dir
        else:
             print("Error: Dataset directory not found.")
             return
             
    if not os.path.exists(wikiart_path):
        # Fallback to just wikiart if archive missing
        alt_wikiart = os.path.join(base_dir, "wikiart")
        if os.path.exists(alt_wikiart):
            print(f"WikiArt 'archive' subdir not found, using: {alt_wikiart}")
            wikiart_path = alt_wikiart
        else:
             print(f"Error: WikiArt path does not exist: {wikiart_path}")
             return

    # Using 1000 iter save interval to get 10% (1000/10000) checkpoint
    # Correcting VGG path to be absolute or relative to run_lab.py execution
    vgg_path = os.path.join("pytorch-AdaIN", "models", "vgg_normalised.pth")
    
    if not os.path.exists(vgg_path):
        print(f"Error: VGG model not found at {vgg_path}")
        return

    cmd = [
        sys.executable, "pytorch-AdaIN/train.py",
        "--content_dir", coco_path,
        "--style_dir", wikiart_path,
        "--vgg", vgg_path,  # Explicitly pass correct path
        "--max_iter", "10000",
        "--save_model_interval", "1000",
        "--save_dir", "pytorch-AdaIN/experiments",
        "--log_dir", "pytorch-AdaIN/logs"
    ]
    print("Starting training (this may take a while)...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_progress_report():
    print("\n[Generating Training Progress Report]")
    print("This requires checkpoints at 1k, 5k, 8k, 10k iterations.")
    cmd = [
        sys.executable, "custom_tasks.py",
        "--mode", "progress",
        "--checkpoints_dir", "pytorch-AdaIN/experiments"
    ]
    subprocess.run(cmd)

def main():
    if not check_models():
        input("Press Enter after you have placed the model files to continue...")
        if not check_models():
            print("Models still missing. Exiting.")
            return

    print("=== Deep Learning Experiment 3 - Part 2 Runner ===")
    
    mode = input("Select Mode:\n1. Full Experiment (Training + Inference)\n2. Inference Only (Custom Tasks)\n3. Training Only\nChoice [2]: ").strip() or "2"

    if mode in ["1", "3"]:
        run_training()
        run_progress_report()
    
    if mode in ["1", "2"]:
        # Task 3 Inputs
        print("\n--- Task 3 Setup: BUPT Scenic Spots ---")
        print("Requirement: Two photos of YOURSELF with BUPT scenic spots.")
        bupt1 = input("Enter path to first BUPT/Selfie image (default: bupt1.jpg): ").strip() or "bupt1.jpg"
        bupt2 = input("Enter path to second BUPT/Selfie image (default: bupt2.jpg): ").strip() or "bupt2.jpg"
        style1 = input("Enter path to style image for Task 3 (default: style.jpg): ").strip() or "style.jpg"
        
        run_task_3([bupt1, bupt2], style1)

        # Task 4 Inputs
        print("\n--- Task 4 Setup: Alpha Blending ---")
        print("Requirement: Use one of the above photos but a DIFFERENT style.")
        style2 = input("Enter path to SECOND style image (default: style2.jpg): ").strip() or "style2.jpg"
        student_id = input("Enter your Student ID (for Alpha logic): ").strip()
        
        if student_id:
            # Re-using bupt1 as per requirement "Pick one of the above photos"
            run_task_4(bupt1, style2, student_id)
        else:
            print("Skipping Task 4 (No Student ID provided).")
    
    print("\n=== All Tasks Completed ===")
    print("Check the output files in this directory.")

if __name__ == "__main__":
    main()
