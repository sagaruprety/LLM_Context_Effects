#!/usr/bin/env python3
"""
Setup script for LLM Context Effects research project.
This script helps set up the environment and verify dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Conda found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Conda not found. Please install Anaconda or Miniconda.")
        return False

def create_conda_environment():
    """Create conda environment for the project."""
    env_name = "LLM_Context_Effects"
    
    print(f"Creating conda environment: {env_name}")
    try:
        subprocess.run([
            'conda', 'create', '-n', env_name, 
            'python=3.9', '-y'
        ], check=True)
        print(f"✅ Conda environment '{env_name}' created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create conda environment: {e}")
        return False


def install_requirements():
    """Install Python requirements in the conda environment."""
    env_name = "LLM_Context_Effects"
    print(f"Installing Python dependencies in conda environment: {env_name}")
    try:
        # Use conda run to execute pip install in the specific environment
        subprocess.run([
            'conda', 'run', '-n', env_name, 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("✅ Dependencies installed successfully in conda environment")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Ollama found: {result.stdout.strip()}")
        
        # Check if Ollama is running
        try:
            subprocess.run(['ollama', 'list'], 
                          capture_output=True, text=True, check=True)
            print("✅ Ollama is running")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Ollama is installed but not running. Start it with: ollama serve")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False


def create_env_template():
    """Create a template .env file for API keys."""
    env_file = Path('.env.template')
    if not env_file.exists():
        env_content = """# API Keys for LLM Context Effects Project
# Copy this file to .env and fill in your actual API keys

# OpenAI API Key (required for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face API Token (required for open-source models)
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here

# LangChain tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=LLM_Context_Effects
"""
        env_file.write_text(env_content)
        print("✅ Created .env.template file")
    else:
        print("✅ .env.template already exists")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed! Next steps:")
    print("="*60)
    print("1. Activate the conda environment (IMPORTANT!):")
    print("   conda activate LLM_Context_Effects")
    print("   # You should see (LLM_Context_Effects) in your prompt")
    print()
    print("2. Verify the installation:")
    print("   python -c 'import torch; print(\"PyTorch version:\", torch.__version__)'")
    print("   python -c 'import langchain; print(\"LangChain installed successfully\")'")
    print()
    print("3. Set up your API keys:")
    print("   cp .env.template .env")
    print("   # Edit .env with your actual API keys")
    print()
    print("4. Install Ollama models (if using local models):")
    print("   ollama pull llama3.2:3b-instruct-fp16")
    print("   ollama pull llama3.1:8b-instruct-fp16")
    print("   ollama pull llama3.1:70b-instruct-fp16")
    print()
    print("5. Run a test experiment:")
    print("   python example_experiment.py")
    print("   # or: python similarity_effect_single_prompt.py")
    print()
    print("6. Explore the analysis notebooks:")
    print("   jupyter notebook analyse_data.ipynb")
    print()
    print("📚 For more details, see the README.md file")


def main():
    """Main setup function."""
    print("🚀 Setting up LLM Context Effects Research Environment")
    print("="*60)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_conda():
        print("❌ Conda is required for this setup. Please install Anaconda or Miniconda first.")
        sys.exit(1)
    
    # Create conda environment
    if not create_conda_environment():
        print("❌ Failed to create conda environment. Exiting.")
        sys.exit(1)
    
    # Install dependencies in the conda environment
    if not install_requirements():
        print("❌ Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Check Ollama
    check_ollama()
    
    # Create environment template
    create_env_template()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
