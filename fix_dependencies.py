#!/usr/bin/env python3
"""
Quick fix script to install missing dependencies in the current conda environment.
Run this if the setup script didn't install dependencies in the right environment.
"""

import subprocess
import sys
import os


def install_missing_dependencies():
    """Install missing dependencies in the current environment."""
    print("🔧 Installing missing dependencies in current environment...")
    
    try:
        # Install PyTorch first (it's often the most problematic)
        print("Installing PyTorch...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'
        ], check=True)
        
        # Install the rest of the requirements
        print("Installing other dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        
        print("✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def verify_installation():
    """Verify that key dependencies are installed."""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except ImportError:
        print("❌ LangChain not found")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    print("✅ All key dependencies verified!")
    return True


def main():
    """Main function."""
    print("🚀 LLM Context Effects - Dependency Fix")
    print("="*50)
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"Current conda environment: {conda_env}")
    
    if conda_env == 'base':
        print("⚠️  Warning: You're in the base conda environment.")
        print("   Consider activating the LLM_Context_Effects environment first:")
        print("   conda activate LLM_Context_Effects")
    
    # Install dependencies
    if install_missing_dependencies():
        # Verify installation
        if verify_installation():
            print("\n🎉 Dependencies fixed successfully!")
            print("\nYou can now run:")
            print("  python example_experiment.py")
            print("  python similarity_effect_single_prompt.py")
        else:
            print("\n⚠️  Some dependencies may still be missing.")
    else:
        print("\n❌ Failed to install dependencies. Please check the error messages above.")


if __name__ == "__main__":
    main()

