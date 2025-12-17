import pypdf
import os

def extract_text(pdf_path, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"--- Extracting from {os.path.basename(pdf_path)} ---\n")
        try:
            reader = pypdf.PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            f.write(text)
        except Exception as e:
            f.write(f"Error reading {pdf_path}: {e}\n")
        f.write("\n--------------------------------------------------\n\n")

files = [
    "minist/2025本科生《神经网络与深度学习》课程实验（3）.pdf",
    "minist/2025本科生《神经网络与深度学习》课程实验（三）.pdf"
]

output_path = "requirements.txt"
if os.path.exists(output_path):
    os.remove(output_path)

for f in files:
    if os.path.exists(f):
        extract_text(f, output_path)
    else:
        print(f"File not found: {f}")
print("Extraction complete.")
