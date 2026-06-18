#!/usr/bin/env python3
"""
Quick fix script for MasterMind model loading issues
Run this to try different fixes for the 0xc000001d error
"""
import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and show the result"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║        MasterMind Model Loading Fix Script                 ║
║  Error: 0xc000001d (Illegal Instruction)                   ║
╚════════════════════════════════════════════════════════════╝

This error usually means llama-cpp-python is built with CPU 
instructions your processor doesn't support.

Choose a fix:

1. Reinstall without AVX/AVX2 (safest, slower)
2. Try pre-built binary wheels (easiest)
3. Install GPU version (if you have NVIDIA GPU)
4. Update .env for CPU-only mode (no reinstall)
5. Download a different model quantization
6. Show diagnostic info
0. Exit

""")

    choice = input("Enter your choice (0-6): ").strip()
    
    if choice == "1":
        print("\n⚠️  This will uninstall and rebuild llama-cpp-python without AVX.")
        print("This is slower but works on older CPUs.")
        confirm = input("Continue? (y/n): ").lower()
        if confirm == 'y':
            run_command(
                "pip uninstall llama-cpp-python -y",
                "Uninstalling llama-cpp-python"
            )
            
            if sys.platform == "win32":
                # Windows
                os.environ["CMAKE_ARGS"] = "-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF"
                run_command(
                    "pip install llama-cpp-python --no-cache-dir --force-reinstall",
                    "Reinstalling without AVX"
                )
            else:
                # Linux/Mac
                run_command(
                    'CMAKE_ARGS="-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF" pip install llama-cpp-python --no-cache-dir --force-reinstall',
                    "Reinstalling without AVX"
                )
    
    elif choice == "2":
        run_command(
            "pip uninstall llama-cpp-python -y",
            "Uninstalling current version"
        )
        run_command(
            "pip install llama-cpp-python --prefer-binary",
            "Installing pre-built wheel"
        )
    
    elif choice == "3":
        print("\n⚠️  This will install the CUDA-enabled version.")
        print("Make sure you have an NVIDIA GPU and CUDA installed.")
        cuda_version = input("Enter CUDA version (e.g., 121 for 12.1, 118 for 11.8): ").strip()
        if cuda_version:
            run_command(
                "pip uninstall llama-cpp-python -y",
                "Uninstalling current version"
            )
            run_command(
                f"pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu{cuda_version}",
                "Installing GPU version"
            )
    
    elif choice == "4":
        print("\n📝 Updating .env file for CPU-only mode...")
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            # Update or add settings
            settings = {
                'N_GPU_LAYERS': '0',
                'FLASH_ATTN': '0',
                'KV_CACHE_TYPE': '0',
                'CONTEXT_SIZE': '8192'  # Reduce context to be safer
            }
            
            new_lines = []
            for line in lines:
                key = line.split('=')[0].strip()
                if key in settings:
                    new_lines.append(f"{key}={settings[key]}\n")
                    del settings[key]
                else:
                    new_lines.append(line)
            
            # Add any remaining settings
            for key, value in settings.items():
                new_lines.append(f"{key}={value}\n")
            
            with open(env_path, 'w') as f:
                f.writelines(new_lines)
            
            print("✓ Updated .env with CPU-only settings:")
            for key, value in settings.items():
                print(f"  {key}={value}")
        else:
            print("✗ .env file not found!")
    
    elif choice == "5":
        print("""
📦 Model Quantization Guide:

Current model: Agent.Nano.Coder-Q4_K_M.gguf

Try these alternative quantizations (in order of compatibility):
  
  1. Q3_K_S  - Smallest, fastest, most compatible
  2. Q4_0    - Simple quantization, very compatible
  3. Q4_K_S  - Smaller than Q4_K_M, more compatible
  4. Q5_0    - Better quality than Q4
  5. Q8_0    - Best quality, larger file

Download from Hugging Face and update MODEL_PATH in .env

Popular models:
  - Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
  - bartowski/Llama-3.2-3B-Instruct-GGUF
  - unsloth/gemma-2-2b-it-GGUF
""")
    
    elif choice == "6":
        run_command(
            "python diagnose_cpu.py",
            "Running diagnostics"
        )
    
    elif choice == "0":
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice!")
        return

    print("\n✓ Done! Try running 'python main.py' again.")

if __name__ == "__main__":
    main()