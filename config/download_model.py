#!/usr/bin/env python3
"""
RWKV-7 G1a Agentic Script
Downloads model with HF authentication, then runs an interactive chat loop.
Requires: torch, huggingface_hub, rwkv
Install:  pip install torch huggingface_hub rwkv
"""

import os
import sys
import torch
from huggingface_hub import hf_hub_download, login
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# ── 1. Configuration ──────────────────────────────────────────
MODEL_REPO = "BlinkDL/rwkv-7-g1a-1.5b"
MODEL_FILE = "RWKV-7-G1a-1.5B-v20251015.pth"
LOCAL_DIR  = "./models"          # where to save the model
MODEL_PATH = os.path.join(LOCAL_DIR, MODEL_FILE)

# Tokenizer / pipeline settings
TOKENIZER_FILE = "20B_tokenizer.json"  # will be downloaded from repo automatically
STRATEGY = "cuda fp16"                # use "cpu fp32" if no GPU

# ── 2. Authentication & Download ─────────────────────────────
def ensure_model():
    """Log in and download the model if it doesn't exist locally."""
    # Check if already downloaded
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model found at {MODEL_PATH}")
        return

    print("Model not found. Downloading from Hugging Face...")
    # Log in using stored token (run `huggingface-cli login` once)
    # Or set the HF_TOKEN environment variable.
    try:
        login()  # will prompt for token if not already logged in
    except Exception as e:
        print("Login failed. Make sure you have run 'huggingface-cli login' or set HF_TOKEN.")
        sys.exit(1)

    # Download with resume support
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Model downloaded to {downloaded_path}")

# ── 3. Load Model and Pipeline ────────────────────────────────
def load_model_and_pipeline():
    print("Loading model... (this may take a while)")
    model = RWKV(model=MODEL_PATH, strategy=STRATEGY)
    pipeline = PIPELINE(model, TOKENIZER_FILE)
    return model, pipeline

# ── 4. Interactive Chat Loop ──────────────────────────────────
def agent_chat_loop(model, pipeline):
    print("\n" + "="*60)
    print("RWKV-7 G1a Agent Chat – type 'quit' to exit")
    print("="*60 + "\n")

    # Get generation parameters
    args = PIPELINE_ARGS(
        temperature=1.0,
        top_p=0.3,
        top_k=0,
        alpha_frequency=0.2,
        alpha_presence=0.2,
        token_ban=[0],        # ban the EOS token to avoid early stop
        token_stop=[],        # optional stop tokens
        chunk_len=256         # split input for long prompts
    )

    # The model expects a specific chat format (adjust if needed)
    # Here we use a simple "User: ...\n\nAssistant:" format (common for RWKV)
    conversation = ""

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ["quit", "exit"]:
            break

        # Append user turn to conversation
        conversation += f"User: {user_input}\n\nAssistant:"

        # Generate response
        output = pipeline.generate(conversation, token_count=256, args=args)
        # Remove the original prompt from output
        response = output[len(conversation):].strip()
        # Often the model will produce up to the next "User:"; we can optionally cut there
        if "\n\nUser:" in response:
            response = response.split("\n\nUser:")[0].strip()

        print(f"Assistant: {response}\n")

        # Update conversation with the assistant's reply
        conversation += response + "\n\n"

# ── 5. Main ───────────────────────────────────────────────────
def main():
    ensure_model()
    model, pipeline = load_model_and_pipeline()
    agent_chat_loop(model, pipeline)

if __name__ == "__main__":
    main()