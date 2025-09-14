#!/bin/bash

set -e

echo "🚀 Setting up ML Project Environment..."

# Check for conda and initialize if needed
if command -v conda &> /dev/null; then
    echo "✅ Conda found, using existing installation"
else
    echo "🔍 Checking for existing Anaconda installation..."
    if [ -d "$HOME/anaconda3" ]; then
        echo "✅ Found Anaconda at $HOME/anaconda3"
        export PATH="$HOME/anaconda3/bin:$PATH"
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "❌ Anaconda not found, please install manually"
        exit 1
    fi
fi

# Initialize conda for current shell
eval "$(conda shell.bash hook)"

# Remove existing environment if it exists
if conda env list | grep -q "^ml-project\s"; then
    echo "🗑️ Removing existing ml-project environment..."
    conda env remove -n ml-project -y
fi

# Create new conda environment
echo "📦 Creating conda environment: ml-project"
conda create -y -n ml-project python=3.10

# Activate environment
echo "🔌 Activating environment..."
conda activate ml-project

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r dev_requirements.txt

# Install additional packages needed for your project
echo "📥 Installing additional ML packages..."
pip install torch fastapi uvicorn streamlit requests

# Install Jupyter kernel
echo "📓 Installing Jupyter kernel..."
python -m ipykernel install --user --name=ml-project --display-name="Python (ml-project)"

echo "✅ Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. conda activate ml-project"
echo "2. streamlit run app/Home.py --server.address 0.0.0.0 --server.port 8501"
