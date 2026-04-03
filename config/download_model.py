#!/usr/bin/env python3
"""
Download Qwen3.5-4B Q4_K_M GGUF + mmproj from public Hugging Face repo.
No authentication needed.
"""

import os
import requests
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DOWNLOAD_DIR = r"C:\Users\dario\.cache\eve"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Model file (4-bit, good balance for CPU)
MODEL_URL = "https://huggingface.co/mradermacher/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B.Q4_K_M.gguf"
MODEL_FILENAME = "Qwen3.5-4B.Q4_K_M.gguf"

# Vision encoder file (required for multimodal)
MMPROJ_URL = "https://huggingface.co/mradermacher/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B.mmproj-Q8_0.gguf"
MMPROJ_FILENAME = "Qwen3.5-4B.mmproj-Q8_0.gguf"
# ------------------------------------------------------------

def download_file(url, destination):
    """Download with progress bar and user-agent header."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()   # 404/401 will raise here
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024      # 1 MiB chunks

    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

if __name__ == "__main__":
    print("📥 Downloading Qwen3.5-4B model and vision encoder...")
    print(f"   Target folder: {DOWNLOAD_DIR}\n")
    
    try:
        # Download main model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_FILENAME)
        print(f"⬇️  {MODEL_FILENAME} (~5.3 GB)")
        download_file(MODEL_URL, model_path)
        print(f"✅ Model saved to {model_path}\n")
        
        # Download mmproj (vision encoder)
        mmproj_path = os.path.join(DOWNLOAD_DIR, MMPROJ_FILENAME)
        print(f"⬇️  {MMPROJ_FILENAME} (~645 MB)")
        download_file(MMPROJ_URL, mmproj_path)
        print(f"✅ mmproj saved to {mmproj_path}\n")
        
        print("🎉 Download complete! Now update your config/settings.py:")
        print(f'   MODEL_PATH = r"{model_path}"')
        print(f'   (and ensure the mmproj file is in the same folder)')
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        print("\n🔧 Manual alternative: Download the files yourself from your browser:")
        print(f"   Model:   {MODEL_URL}")
        print(f"   mmproj:  {MMPROJ_URL}")
        print(f"   Then move them to {DOWNLOAD_DIR}")