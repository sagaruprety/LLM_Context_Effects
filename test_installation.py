#!/usr/bin/env python3
"""
Test script to verify that all dependencies are installed correctly.
This script doesn't require API keys and tests basic functionality.
"""

import sys
import importlib


def test_import(module_name, package_name=None):
    """Test if a module can be imported successfully."""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✅ {module_name}: {getattr(module, '__version__', 'installed')}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key libraries."""
    print("🧪 Testing basic functionality...")
    
    # Test pandas
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        print("✅ Pandas: Basic DataFrame operations work")
    except Exception as e:
        print(f"❌ Pandas: {e}")
        return False
    
    # Test torch
    try:
        import torch
        x = torch.tensor([1, 2, 3])
        y = x * 2
        assert y.tolist() == [2, 4, 6]
        print("✅ PyTorch: Basic tensor operations work")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    # Test langchain
    try:
        from langchain_core.prompts import ChatPromptTemplate
        template = ChatPromptTemplate.from_template("Hello {name}")
        assert template is not None
        print("✅ LangChain: Basic prompt template creation works")
    except Exception as e:
        print(f"❌ LangChain: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    print("🚀 LLM Context Effects - Installation Test")
    print("="*60)
    
    # Test core dependencies
    print("\n📦 Testing core dependencies...")
    core_modules = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'torch',
        'transformers',
        'langchain',
        'langchain_core',
        'langchain_openai',
        'langchain_ollama',
        'scipy',
        'sklearn',
        'statsmodels',
        'plotly'
    ]
    
    success_count = 0
    for module in core_modules:
        if test_import(module):
            success_count += 1
    
    print(f"\n📊 Import test results: {success_count}/{len(core_modules)} modules imported successfully")
    
    # Test basic functionality
    if test_basic_functionality():
        print("\n🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Set up your API keys:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   # or create a .env file")
        print("2. Run the example experiment:")
        print("   python example_experiment.py")
        print("3. Explore the analysis notebooks:")
        print("   jupyter notebook analyse_data.ipynb")
    else:
        print("\n❌ Some functionality tests failed. Please check the error messages above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

